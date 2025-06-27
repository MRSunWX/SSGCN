#!/bin/bash

# 设置 Python 可执行文件路径
PYTHON_EXEC=python

# 数据目录
DATA_DIR="./dataset/processed"

# 输出目录和日志目录
# 用于 Span-based Syntax Graph Convolutional Network (SSGCN)
OUTPUT_DIR_RESTAURANTS="./outputs/ssgcn_restaurants"
OUTPUT_DIR_LAPTOPS="./outputs/ssgcn_laptops"
LOG_DIR_RESTAURANTS="./logs/tensorboard_restaurants"
LOG_DIR_LAPTOPS="./logs/tensorboard_laptops"

# 创建输出和日志目录
mkdir -p "${OUTPUT_DIR_RESTAURANTS}"
mkdir -p "${OUTPUT_DIR_LAPTOPS}"
mkdir -p "${LOG_DIR_RESTAURANTS}"
mkdir -p "${LOG_DIR_LAPTOPS}"

# 检查 train.py 是否存在
if [ ! -f "train.py" ]; then
    echo "错误：train.py 未找到！请确保 train.py 位于当前目录。"
    exit 1
fi

# 检查数据集文件是否存在
for DATASET in Restaurants_Train.pkl Restaurants_Test.pkl Laptops_Train.pkl Laptops_Test.pkl; do
    if [ ! -f "${DATA_DIR}/${DATASET}" ]; then
        echo "错误：数据集文件 ${DATA_DIR}/${DATASET} 未找到！"
        exit 1
    fi
done

# 检查 data_utils.py 是否存在
if [ ! -f "data_utils.py" ]; then
    echo "错误：data_utils.py 未找到！请确保 data_utils.py 位于当前目录。"
    exit 1
fi

# 检查 ssgcn.py 是否存在
if [ ! -f "ssgcn.py" ]; then
    echo "错误：ssgcn.py 未找到！请确保 ssgcn.py 位于当前目录。"
    exit 1
fi

echo "开始调参 Restaurants 数据集..."
${PYTHON_EXEC} train.py \
    --dataset restaurants \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_DIR_RESTAURANTS}" \
    --log-dir "${LOG_DIR_RESTAURANTS}" \
    --n-trials 10 \
    --num-epochs 20 \
    --seed 1000 \
    --max-hop 3 \
    --alpha 1.0 \
    > "${LOG_DIR_RESTAURANTS}/optuna.log" 2>&1
echo "Restaurants 数据集调参完成！日志保存至 ${LOG_DIR_RESTAURANTS}/optuna.log"
echo "最佳模型保存至 ${OUTPUT_DIR_RESTAURANTS}/best_model.pt"

echo "开始调参 Laptops 数据集..."
${PYTHON_EXEC} train.py \
    --dataset laptops \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_DIR_LAPTOPS}" \
    --log-dir "${LOG_DIR_LAPTOPS}" \
    --n-trials 10 \
    --num-epochs 20 \
    --seed 1000 \
    --max-hop 3 \
    --alpha 1.0 \
    > "${LOG_DIR_LAPTOPS}/optuna.log" 2>&1
echo "Laptops 数据集调参完成！日志保存至 ${LOG_DIR_LAPTOPS}/optuna.log"
echo "最佳模型保存至 ${OUTPUT_DIR_LAPTOPS}/best_model.pt"

echo "所有调参任务完成！"
echo "启动 TensorBoard 查看结果："
echo "tensorboard --logdir ./logs/tensorboard_restaurants --port 6006"
echo "tensorboard --logdir ./logs/tensorboard_laptops --port 6007"