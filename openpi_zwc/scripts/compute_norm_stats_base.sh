# ��������ģʽ��huggingface����Ŀ¼
export TRANSFORMERS_OFFLINE=1
export HF_HOME="/mnt/ssd0/data/pi0_agent/zwc/cache/huggingface"
export OPENPI_DATA_HOME="/mnt/ssd0/data/pi0_agent/zwc/cache/"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 

# Ȼ�����м���ű�
# python scripts/compute_norm_stats.py --config-name pi0_airbot
python scripts/compute_norm_stats.py --config-name pi0_airbot_low_mem_finetune