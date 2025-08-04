#!/bin/bash

# 设置环境变量
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9  # 显存
export HF_HOME="/mnt/ssd0/data/pi0_agent/zwc/cache/huggingface"
export OPENPI_DATA_HOME="/mnt/ssd0/data/pi0_agent/zwc/cache/"
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5  # 使用GPU

# 运行训练脚本

source /home/zwc/openpi/.venv/bin/activate

# pi0 fast
python scripts/train.py pi0_fast_airbot \
    --exp-name=airbot_finetune_pi0_fast_experiment_whole_task_1 \
    --overwrite


# pi0 base
# python scripts/train.py pi0_airbot \
#     --exp-name=airbot_finetune_pi0_experiment_multi_task_2 \
#     --model.action-horizon=20 \
#     --batch-size=64 \
#     --overwrite