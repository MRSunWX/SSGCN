# Span-based Syntax Graph Convolutional Network (SSGCN)

This project implements the **Span-based Syntax Graph Convolutional Network (SSGCN)** for aspect-based sentiment analysis on the SemEval-2014 dataset, covering `restaurants` and `laptops` domains. The model integrates multi-view Span Attention and syntax-guided Graph Convolutional Networks (GCN) for enhanced sentiment classification. It includes data preprocessing, hyperparameter tuning with Optuna, and training visualization with TensorBoard.

## Project Structure

- `train.py`: Script for training with Optuna hyperparameter tuning and TensorBoard logging.
- `train.sh`: Bash script to execute training for both datasets.
- `data_utils.py`: Utilities for data preprocessing and dataset creation.
- `ssgcn.py`: SSGCN model definition.
- `requirements.txt`: Python dependencies.
- `README.md`: Project documentation.
- `dataset/semeval14/`: Directory for input JSON files (`Restaurants_Train.json`, `Restaurants_Test.json`, `Laptops_Train.json`, `Laptops_Test.json`).
- `dataset/processed/`: Directory for processed pickle files (`Restaurants_Train.pkl`, `Restaurants_Test.pkl`, `Laptops_Train.pkl`, `Laptops_Test.pkl`).
- `outputs/ssgcn_restaurants/`: Directory for restaurant dataset model outputs.
- `outputs/ssgcn_laptops/`: Directory for laptop dataset model outputs.
- `logs/tensorboard_restaurants/`: Directory for restaurant dataset TensorBoard logs.
- `logs/tensorboard_laptops/`: Directory for laptop dataset TensorBoard logs.

## Requirements

- Python 3.8 or higher
- GPU (recommended for faster training and preprocessing)

Install dependencies:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Setup and Usage

### 1. Prepare Dataset

Place SemEval-2014 JSON files in `dataset/semeval14/`. Then, generate processed pickle files:

```bash
python data_utils.py --input-dir ./dataset/semeval14 --max-seq-len 128 --alpha 1.0 --max-dep-dist 3
```

This creates:
- `dataset/processed/Restaurants_Train.pkl`
- `dataset/processed/Restaurants_Test.pkl`
- `dataset/processed/Laptops_Train.pkl`
- `dataset/processed/Laptops_Test.pkl`

### 2. Train the Model

Run training with hyperparameter tuning for both datasets:

```bash
chmod +x train.sh
./train.sh
```

For Windows, execute manually:

```cmd
mkdir logs\tensorboard_restaurants
mkdir logs\tensorboard_laptops
mkdir outputs\ssgcn_restaurants
mkdir outputs\ssgcn_laptops
python train.py --dataset restaurants --data-dir ./dataset/processed --output-dir ./outputs/ssgcn_restaurants --log-dir ./logs/tensorboard_restaurants --n-trials 10 --num-epochs 20 --seed 1000 --max-hop 3 --alpha 1.0 > logs\tensorboard_restaurants\optuna.log 2>&1
python train.py --dataset laptops --data-dir ./dataset/processed --output-dir ./outputs/ssgcn_laptops --log-dir ./logs/tensorboard_laptops --n-trials 10 --num-epochs 20 --seed 1000 --max-hop 3 --alpha 1.0 > logs\tensorboard_laptops\optuna.log 2>&1
```

### 3. View Results

- **Training Logs**: Check `logs/tensorboard_restaurants/optuna.log` and `logs/tensorboard_laptops/optuna.log` for performance metrics and best hyperparameters.
- **Model Weights**: Best models are saved to `outputs/ssgcn_restaurants/best_model.pt` and `outputs/ssgcn_laptops/best_model.pt`.
- **Visualization**: Use TensorBoard to view training metrics:

```bash
tensorboard --logdir ./logs/tensorboard_restaurants --port 6006
tensorboard --logdir ./logs/tensorboard_laptops --port 6007
```

Access via `http://localhost:6006` or `http://localhost:6007` in a browser.

## Inference

Perform inference with the trained model (example for `restaurants`):

```python
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
    dropout=0.45,
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
```

## Debugging

- **Missing Pickle Files**: Regenerate with:

```bash
python data_utils.py --input-dir ./dataset/semeval14
```

- **Training Errors**: Review logs in `logs/tensorboard_restaurants/trial_n/trial.log` or `logs/tensorboard_laptops/optuna.log`.

- **Verify Dataset**:

```python
import pickle
from data_utils import ABSADataset
with open('./dataset/processed/Restaurants_Train.pkl', 'rb') as f:
    dataset = pickle.load(f)
    print(f"Dataset size: {len(dataset)}")
    print(f"Sample: {dataset[0]}")
```

- **Aspect Misalignment**: Check `[Warning] Aspect未对齐` in logs. If frequent, adjust `data_utils.py` thresholds (`fuzzy_threshold=80`, `semantic_threshold=0.7`).

## Notes

- **Performance**: Use a GPU for faster training. Reduce `--batch-size` (e.g., 8) or `--max-seq-len` (e.g., 64) if memory is limited.
- **Hyperparameters**: Best hyperparameters are logged in `optuna.log` and TensorBoard (`HPARAMS` tab).
- **Customization**:
  - Edit `train.py` to modify Optuna search ranges or add parameters.
  - Update `train.sh` to adjust `--n-trials` or `--num-epochs`.

## Support

For issues or questions, refer to the debugging section or check the project's issue tracker on your repository hosting platform.