Span-based Syntax Graph Convolutional Network (SSGCN) for Aspect-Based Sentiment Analysis
This project implements a Span-based Syntax Graph Convolutional Network (SSGCN) for aspect-based sentiment analysis on the SemEval-2014 dataset (restaurants and laptops). It combines multi-view Span Attention and syntax-guided Graph Convolutional Networks (GCN), with data preprocessing, hyperparameter tuning using Optuna, and visualization via TensorBoard.
Project Structure
├── train.py              # Training script with Optuna and TensorBoard
├── train.sh              # Bash script to run training for both datasets
├── data_utils.py         # Data preprocessing and dataset creation
├── ssgcn.py              # SSGCN model definition
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── dataset/
│   ├── semeval14/        # Input JSON files (Restaurants_Train.json, etc.)
│   ├── processed/        # Processed .pkl files
├── outputs/
│   ├── ssgcn_restaurants/  # Models for restaurants dataset
│   ├── ssgcn_laptops/      # Models for laptops dataset
├── logs/
│   ├── tensorboard_restaurants/  # TensorBoard logs for restaurants
│   ├── tensorboard_laptops/      # TensorBoard logs for laptops

Requirements

Python 3.8+
GPU (recommended for faster training and data preprocessing)

Install dependencies:
pip install -r requirements.txt
python -m spacy download en_core_web_sm

Setup

Prepare Dataset:

Place SemEval-2014 JSON files in dataset/semeval14/ (e.g., Restaurants_Train.json, Restaurants_Test.json, Laptops_Train.json, Laptops_Test.json).

Generate processed .pkl files:
python data_utils.py --input-dir ./dataset/semeval14 --max-seq-len 128 --alpha 1.0 --max-dep-dist 3


This creates dataset/processed/Restaurants_Train.pkl, Restaurants_Test.pkl, Laptops_Train.pkl, Laptops_Test.pkl.



Run Training:

Use train.sh to train on both datasets with hyperparameter tuning:
chmod +x train.sh
./train.sh


Windows (manual commands):
mkdir logs\tensorboard_restaurants
mkdir logs\tensorboard_laptops
mkdir outputs\ssgcn_restaurants
mkdir outputs\ssgcn_laptops
python train.py --dataset restaurants --data-dir ./dataset/processed --output-dir ./outputs/ssgcn_restaurants --log-dir ./logs/tensorboard_restaurants --n-trials 10 --num-epochs 20 --seed 1000 --max-hop 3 --alpha 1.0 > logs\tensorboard_restaurants\optuna.log 2>&1
python train.py --dataset laptops --data-dir ./dataset/processed --output-dir ./outputs/ssgcn_laptops --log-dir ./logs/tensorboard_laptops --n-trials 10 --num-epochs 20 --seed 1000 --max-hop 3 --alpha 1.0 > logs\tensorboard_laptops\optuna.log 2>&1




View Results:

Check training logs in logs/tensorboard_restaurants/optuna.log and logs/tensorboard_laptops/optuna.log.

Best models are saved to outputs/ssgcn_restaurants/best_model.pt and outputs/ssgcn_laptops/best_model.pt.

Visualize training metrics with TensorBoard:
tensorboard --logdir ./logs/tensorboard_restaurants --port 6006
tensorboard --logdir ./logs/tensorboard_laptops --port 6007

Open http://localhost:6006 or http://localhost:6007 in your browser.




Inference
Use the best model for prediction (example for restaurants):
from ssgcn import SSGCN
import torch
from torch.utils.data import DataLoader
import pickle
from data_utils import ABSADataset, collate_fn
import argparse

args = argparse.Namespace(
    dataset='restaurants',
    bert_model='bert-base-uncased',
    num_classes=3,
    dropout=0.45,  # Use best hyperparameters from tuning
    hidden_dim=250,
    num_layers=2,
    max_hop=3,
    alpha=0.75
)
model = SSGCN(args).to('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load('./outputs/ssgcn_restaurants/best_model.pt'))
model.eval()

with open('./dataset/processed/Restaurants_Test.pkl', 'rb') as f:
    dataset = pickle.load(f)
test_loader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

predictions = []
with torch.no_grad():
    for batch in test_loader:
        for key in batch:
            if isinstance(batch[key], list):
                batch[key] = [item.to(model.device) for item in batch[key]]
            else:
                batch[key] = batch[key].to(model.device)
        logits = model(batch)
        preds = torch.argmax(logits, dim=1).cpu().tolist()
        predictions.extend(preds)
print("Predictions:", predictions)

Debugging

Missing .pkl Files: Re-run data preprocessing:
python data_utils.py --input-dir ./dataset/semeval14


Training Errors: Check logs/tensorboard_restaurants/trial_n/trial.log or optuna.log for details.

Verify Dataset:
import pickle
from data_utils import ABSADataset
with open('./dataset/processed/Restaurants_Train.pkl', 'rb') as f:
    dataset = pickle.load(f)
    print(f"Dataset size: {len(dataset)}")
    print(f"Sample: {dataset[0]}")


Aspect Misalignment: Check [Warning] Aspect未对齐 in logs. If frequent, adjust data_utils.py thresholds (fuzzy: 80, semantic: 0.7).


Notes

Performance: Use a GPU for faster training and data preprocessing. Reduce --batch-size (e.g., 8) or --max-seq-len (e.g., 64) if memory is limited.
Hyperparameters: Best hyperparameters are logged in optuna.log and TensorBoard (HPARAMS tab).
Customization:
Modify train.py to adjust Optuna search ranges or add new parameters.
Update train.sh for different --n-trials or --num-epochs.


