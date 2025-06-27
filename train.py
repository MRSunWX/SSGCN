import os
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ssgcn import SSGCN
from data_utils import ABSADataset, Tokenizer, DependencyGraph, collate_fn
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm
import pickle
import optuna
from torch.utils.tensorboard import SummaryWriter
import logging

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

def train_trial(trial, args, train_loader, valid_loader, test_loader, device, writer, trial_log_dir):
    # Optuna 超参数建议
    args.lr_bert = trial.suggest_float('lr_bert', 1e-5, 5e-5, log=True)
    args.lr_other = trial.suggest_float('lr_other', 5e-4, 5e-3, log=True)
    args.dropout = trial.suggest_float('dropout', 0.3, 0.7)
    args.hidden_dim = trial.suggest_int('hidden_dim', 100, 400, step=50)
    args.num_layers = trial.suggest_int('num_layers', 1, 3)
    args.batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
    args.alpha = trial.suggest_float('alpha', 0.5, 1.0)

    # 更新 DataLoader 的 batch_size
    train_loader = DataLoader(train_loader.dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_loader.dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_loader.dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # 初始化模型
    model = SSGCN(args).to(device)
    criterion = nn.CrossEntropyLoss()

    # 优化器和学习率调度
    no_decay = ['bias', 'LayerNorm.weight']
    bert_params = list(model.bert.named_parameters())
    other_params = [p for n, p in model.named_parameters() if not n.startswith('bert')]

    optimizer_grouped_parameters = [
        {'params': [p for n, p in bert_params if not any(nd in n for nd in no_decay)],
         'weight_decay': args.l2reg, 'lr': args.lr_bert},
        {'params': [p for n, p in bert_params if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, 'lr': args.lr_bert},
        {'params': other_params, 'weight_decay': args.l2reg, 'lr': args.lr_other}
    ]

    optimizer = AdamW(optimizer_grouped_parameters)
    total_steps = len(train_loader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * args.warmup_ratio), num_training_steps=total_steps
    )

    # 训练循环
    best_f1 = 0
    patience_counter = 0
    trial_save_path = os.path.join(trial_log_dir, f'trial_{trial.number}_best_model.pt')

    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f'Trial {trial.number} Epoch {epoch+1}')):
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

            # 记录学习率到 TensorBoard
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar(f'Trial_{trial.number}/LR_BERT', optimizer.param_groups[0]['lr'], global_step)
            writer.add_scalar(f'Trial_{trial.number}/LR_Other', optimizer.param_groups[2]['lr'], global_step)

        avg_train_loss = total_loss / len(train_loader)
        val_loss, val_acc, val_f1 = evaluate(model, valid_loader, criterion, device)

        # 记录到 TensorBoard
        writer.add_scalar(f'Trial_{trial.number}/Train_Loss', avg_train_loss, epoch)
        writer.add_scalar(f'Trial_{trial.number}/Val_Loss', val_loss, epoch)
        writer.add_scalar(f'Trial_{trial.number}/Val_Acc', val_acc, epoch)
        writer.add_scalar(f'Trial_{trial.number}/Val_F1', val_f1, epoch)

        # 记录到日志文件
        logging.info(f'Trial {trial.number} Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}')

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), trial_save_path)
            logging.info(f'Trial {trial.number} | Best model saved at epoch {epoch+1}')
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logging.info(f'Trial {trial.number} | Early stopping triggered.')
                break

    # 加载最佳模型并测试
    model.load_state_dict(torch.load(trial_save_path))
    test_loss, test_acc, test_f1 = evaluate(model, test_loader, criterion, device)
    writer.add_scalar(f'Trial_{trial.number}/Test_Loss', test_loss, args.num_epochs)
    writer.add_scalar(f'Trial_{trial.number}/Test_Acc', test_acc, args.num_epochs)
    writer.add_scalar(f'Trial_{trial.number}/Test_F1', test_f1, args.num_epochs)

    logging.info(f'Trial {trial.number} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | Test F1: {test_f1:.4f}')

    # 详细分类报告
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
            preds = torch.argmax(logits, dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(batch['label'].cpu().tolist())  # 修正此处：labels 改为 batch['label']
    report = classification_report(all_labels, all_preds, target_names=['negative', 'neutral', 'positive'], output_dict=True)
    logging.info(f'Trial {trial.number} | Detailed Test Classification Report:\n{report}')

    # 记录超参数和最终 F1 分数到 TensorBoard
    writer.add_hparams(
        {'lr_bert': args.lr_bert, 'lr_other': args.lr_other, 'dropout': args.dropout,
         'hidden_dim': args.hidden_dim, 'num_layers': args.num_layers, 'batch_size': args.batch_size, 'alpha': args.alpha},
        {'test_f1': test_f1, 'test_acc': test_acc}
    )

    return test_f1

def objective(trial, args, train_dataset, valid_dataset, test_dataset, output_dir, log_dir):
    # 设置 TensorBoard 写入器
    trial_log_dir = os.path.join(log_dir, f'trial_{trial.number}')
    os.makedirs(trial_log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=trial_log_dir)

    # 设置日志
    logging.basicConfig(
        filename=os.path.join(trial_log_dir, 'trial.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # 在此处设置初始 batch_size
    args.batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_f1 = train_trial(trial, args, train_loader, valid_loader, test_loader, device, writer, trial_log_dir)
    writer.close()
    return test_f1

def main():
    parser = argparse.ArgumentParser(description="Train Span-based Syntax Graph Convolutional Network (SSGCN) with Optuna hyperparameter tuning")
    parser.add_argument('--data-dir', type=str, default='./dataset/processed', help='Directory containing .pkl files')
    parser.add_argument('--dataset', type=str, default='restaurants', choices=['restaurants', 'laptops'], help='Dataset to use: restaurants or laptops')
    parser.add_argument('--bert-model', type=str, default='bert-base-uncased', help='Pretrained BERT model')
    parser.add_argument('--num-classes', type=int, default=3, help='Number of classes (negative, neutral, positive)')
    parser.add_argument('--max-hop', type=int, default=3, help='Maximum dependency hop distance')
    parser.add_argument('--l2reg', type=float, default=1e-5, help='L2 regularization weight')
    parser.add_argument('--num-epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
    parser.add_argument('--warmup-ratio', type=float, default=0.1, help='Warmup ratio for learning rate scheduler')
    parser.add_argument('--seed', type=int, default=1000, help='Random seed')
    parser.add_argument('--output-dir', type=str, default='./outputs/ssgcn', help='Output directory for model')
    parser.add_argument('--n-trials', type=int, default=10, help='Number of Optuna trials')
    parser.add_argument('--log-dir', type=str, default='./logs/tensorboard', help='TensorBoard log directory')
    parser.add_argument('--alpha', type=float, default=1.0, help='Alpha parameter for dependency graph weights')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    set_seed(args.seed)

    # 加载数据集
    train_file = os.path.join(args.data_dir, f"{args.dataset.capitalize()}_Train.pkl")
    test_file = os.path.join(args.data_dir, f"{args.dataset.capitalize()}_Test.pkl")

    if os.path.exists(train_file):
        with open(train_file, 'rb') as f:
            train_dataset = pickle.load(f)
    else:
        raise FileNotFoundError(f"Training dataset {train_file} not found")

    if os.path.exists(test_file):
        with open(test_file, 'rb') as f:
            test_dataset = pickle.load(f)
    else:
        raise FileNotFoundError(f"Test dataset {test_file} not found")

    # 划分验证集
    train_size = int(0.9 * len(train_dataset))
    valid_size = len(train_dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, valid_size], generator=torch.Generator().manual_seed(args.seed)
    )

    # 创建 Optuna 学习
    study = optuna.create_study(direction='maximize', study_name=f'ssgcn_{args.dataset}')
    study.optimize(
        lambda trial: objective(trial, args, train_dataset, valid_dataset, test_dataset, args.output_dir, args.log_dir),
        n_trials=args.n_trials
    )

    # 输出最佳试验结果
    best_trial = study.best_trial
    print(f'Best trial: {best_trial.number}')
    print(f'Best test F1: {best_trial.value:.4f}')
    print('Best hyperparameters:')
    for key, value in best_trial.params.items():
        print(f'  {key}: {value}')

    # 保存最佳模型到 output_dir
    best_model_path = os.path.join(args.log_dir, f'trial_{best_trial.number}', f'trial_{best_trial.number}_best_model.pt')
    final_model_path = os.path.join(args.output_dir, 'best_model.pt')
    os.rename(best_model_path, final_model_path)
    print(f'Best model saved to {final_model_path}')

if __name__ == '__main__':
    main()