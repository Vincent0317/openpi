#!/bin/bash

# ���û�������
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9  # �Դ�
export HF_HOME="/mnt/ssd0/data/pi0_agent/zwc/cache/huggingface"
export OPENPI_DATA_HOME="/mnt/ssd0/data/pi0_agent/zwc/cache/"
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # ʹ��GPU

# ����ѵ���ű�

source /home/zwc/openpi/.venv/bin/activate


# pi0 base
python scripts/train.py pi0_airbot_low_mem_finetune \
    --exp-name=airbot_finetune_pi0_low_mem_experiment_whole_task_2
    # --overwrite