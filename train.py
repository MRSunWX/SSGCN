import os
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from ssgcn import SSGCN  # 从 ssgcn.py 导入 SSGCN
from data_utils import ABSADataset, Tokenizer, DependencyGraph
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm
import pickle

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def collate_fn(batch):
    batch_out = {}
    keys = batch[0].keys()
    for key in keys:
        if key in ['edge_index', 'edge_weight', 'aspect_pos']:
            batch_out[key] = [item[key] for item in batch]
        else:
            batch_out[key] = torch.stack([item[key] for item in batch])
    return batch_out

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

def main():
    parser = argparse.ArgumentParser(description="Train SSGCN model on SemEval dataset")
    parser.add_argument('--data-dir', type=str, default='./dataset/processed', help='Directory containing .pkl files')
    parser.add_argument('--bert-model', type=str, default='bert-base-uncased', help='Pretrained BERT model')
    parser.add_argument('--num-classes', type=int, default=3, help='Number of classes (negative, neutral, positive)')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--hidden-dim', type=int, default=300, help='Hidden dimension size')
    parser.add_argument('--num-layers', type=int, default=2, help='Number of GCN layers')
    parser.add_argument('--max-hop', type=int, default=3, help='Maximum dependency hop distance')
    parser.add_argument('--alpha', type=float, default=0.8, help='DSW distance decay factor')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--lr-bert', type=float, default=2e-5, help='Learning rate for BERT parameters')
    parser.add_argument('--lr-other', type=float, default=1e-3, help='Learning rate for other parameters')
    parser.add_argument('--l2reg', type=float, default=1e-5, help='L2 regularization weight')
    parser.add_argument('--num-epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
    parser.add_argument('--warmup-ratio', type=float, default=0.1, help='Warmup ratio for learning rate scheduler')
    parser.add_argument('--seed', type=int, default=1000, help='Random seed')
    parser.add_argument('--output-dir', type=str, default='./outputs/ssgcn', help='Output directory for model')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    # 加载数据集
    train_files = [
        os.path.join(args.data_dir, 'Restaurants_Train.pkl'),
        os.path.join(args.data_dir, 'Laptops_Train.pkl')
    ]
    test_files = [
        os.path.join(args.data_dir, 'Restaurants_Test.pkl'),
        os.path.join(args.data_dir, 'Laptops_Test.pkl')
    ]

    train_datasets = []
    test_datasets = []

    for train_file in train_files:
        if os.path.exists(train_file):
            with open(train_file, 'rb') as f:
                dataset = pickle.load(f)
                train_datasets.append(dataset)
        else:
            print(f"Warning: {train_file} not found")

    for test_file in test_files:
        if os.path.exists(test_file):
            with open(test_file, 'rb') as f:
                dataset = pickle.load(f)
                test_datasets.append(dataset)
        else:
            print(f"Warning: {test_file} not found")

    if not train_datasets:
        raise ValueError("No training datasets found")
    if not test_datasets:
        raise ValueError("No test datasets found")

    # 从训练集中划分验证集（10%）
    train_dataset = ConcatDataset(train_datasets)
    test_dataset = ConcatDataset(test_datasets)
    train_size = int(0.9 * len(train_dataset))
    valid_size = len(train_dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, valid_size], generator=torch.Generator().manual_seed(args.seed)
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # 初始化模型
    model = SSGCN(args).to('cuda' if torch.cuda.is_available() else 'cpu')
    device = next(model.parameters()).device

    criterion = nn.CrossEntropyLoss()

    # 优化器和学习率调度
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

    # 训练循环
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

    # 最终测试
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