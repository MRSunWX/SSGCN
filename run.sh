#!/bin/bash

mkdir -p logs

# =========================
# SSGCN on Restaurants
# =========================
echo "==== Running SSGCN on Restaurants ===="
CUDA_VISIBLE_DEVICES=0 python ./train.py \
  --data-dir './dataset/processed' \
  --bert-model 'bert-base-uncased' \
  --num-classes 3 \
  --dropout 0.5 \
  --hidden-dim 300 \
  --num-layers 2 \
  --max-hop 3 \
  --alpha 0.8 \
  --batch-size 16 \
  --lr-bert 2e-5 \
  --lr-other 1e-3 \
  --l2reg 1e-5 \
  --num-epochs 20 \
  --patience 5 \
  --warmup-ratio 0.1 \
  --seed 1000 \
  --output-dir './outputs/ssgcn_restaurant' \
  > logs/ssgcn_restaurant.log 2>&1

# =========================
# SSGCN on Laptops
# =========================
echo "==== Running SSGCN on Laptops ===="
CUDA_VISIBLE_DEVICES=0 python ./train.py \
  --data-dir './dataset/processed' \
  --bert-model 'bert-base-uncased' \
  --num-classes 3 \
  --dropout 0.5 \
  --hidden-dim 300 \
  --num-layers 2 \
  --max-hop 3 \
  --alpha 0.8 \
  --batch-size 16 \
  --lr-bert 2e-5 \
  --lr-other 1e-3 \
  --l2reg 1e-5 \
  --num-epochs 20 \
  --patience 5 \
  --warmup-ratio 0.1 \
  --seed 1000 \
  --output-dir './outputs/ssgcn_laptop' \
  > logs/ssgcn_laptop.log 2>&1
