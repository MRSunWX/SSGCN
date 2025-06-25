import os
import random
import argparse
import numpy as np
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_utils import Tokenizer, DependencyGraph, ABSADataset, collate_fn
from ssgcn import SSGCN
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            for key in batch:
                if isinstance(batch[key], list):
                    batch[key] = [item.to(device) for item in batch[key]]
                else:
                    batch[key] = batch[key].to(device)

            logits = model(batch)
            labels = batch['label']
            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    return total_loss / len(dataloader), acc, f1

def build_dataloader_from_json(data_path, tokenizer, dep_parser, batch_size=16, shuffle=True, split='train'):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if split in ['train', 'valid']:
        train_data, valid_data = train_test_split(data, test_size=0.1, random_state=1000)
        data = train_data if split == 'train' else valid_data

    dataset = ABSADataset(data, tokenizer, dep_parser)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == 'train'), collate_fn=collate_fn)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./dataset/Restaurants_corenlp')
    parser.add_argument('--bert_model', type=str, default='bert-base-uncased')
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--hidden_dim', type=int, default=300)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--max_hop', type=int, default=3)
    parser.add_argument('--alpha', type=float, default=0.8)

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr_bert', type=float, default=2e-5)
    parser.add_argument('--lr_other', type=float, default=1e-3)
    parser.add_argument('--l2reg', type=float, default=1e-5)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)

    parser.add_argument('--seed', type=int, default=1000)
    parser.add_argument('--output_dir', type=str, default='./outputs/ssgcn-Restaurants')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    tokenizer = Tokenizer(pretrained_model=args.bert_model)
    dep_parser = DependencyGraph()

    train_path = os.path.join(args.data_dir, 'train.json')
    test_path = os.path.join(args.data_dir, 'test.json')

    train_loader = build_dataloader_from_json(train_path, tokenizer, dep_parser, batch_size=args.batch_size, split='train')
    valid_loader = build_dataloader_from_json(train_path, tokenizer, dep_parser, batch_size=args.batch_size, split='valid')
    test_loader = build_dataloader_from_json(test_path, tokenizer, dep_parser, batch_size=args.batch_size, split='test')

    model = SSGCN(args).to('cuda' if torch.cuda.is_available() else 'cpu')
    device = next(model.parameters()).device

    criterion = nn.CrossEntropyLoss()

    no_decay = ['bias', 'LayerNorm.weight']
    bert_params = list(model.bert.named_parameters())
    other_params = [p for n, p in model.named_parameters() if not n.startswith('bert')]

    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in bert_params if not any(nd in n for nd in no_decay)],
            'weight_decay': args.l2reg,
            'lr': args.lr_bert
        },
        {
            'params': [p for n, p in bert_params if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
            'lr': args.lr_bert
        },
        {
            'params': other_params,
            'weight_decay': args.l2reg,
            'lr': args.lr_other
        }
    ]

    optimizer = AdamW(optimizer_grouped_parameters)
    total_steps = len(train_loader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps
    )

    best_f1 = 0
    patience_counter = 0
    save_path = os.path.join(args.output_dir, 'best_model.pt')

    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
            for key in batch:
                if isinstance(batch[key], list):
                    batch[key] = [item.to(device) for item in batch[key]]
                else:
                    batch[key] = batch[key].to(device)

            logits = model(batch)
            labels = batch['label']
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        val_loss, val_acc, val_f1 = evaluate(model, valid_loader, criterion, device)

        print(f'>> Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}')

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f'>> Best model saved at epoch {epoch+1}')
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print('>> Early stopping triggered.')
                break

    print('>> Loading best model for final evaluation...')
    model.load_state_dict(torch.load(save_path))
    test_loss, test_acc, test_f1 = evaluate(model, test_loader, criterion, device)
    print(f'>> Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | Test F1: {test_f1:.4f}')

    print('\n>> Detailed Test Classification Report:')
    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            for key in batch:
                if isinstance(batch[key], list):
                    batch[key] = [item.to(device) for item in batch[key]]
                else:
                    batch[key] = batch[key].to(device)
            logits = model(batch)
            labels = batch['label']
            preds = torch.argmax(logits, dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())
    print(classification_report(all_labels, all_preds, target_names=['negative', 'neutral', 'positive']))

if __name__ == '__main__':
    main()
