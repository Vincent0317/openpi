# Զ��copy

scp -r "D:\A Code\openpi" zwc@122.115.57.205:/home/zwc
scp -r /data2/zwc/cache zwc@122.115.57.205:/mnt/ssd0/data/pi0_agent/zwc
scp -r /mnt/ssd0/data/pi0_agent/zwc/pi0_airbot/airbot_finetune_pi0_experiment/29999 dodo@1.tcp.cpolar.cn -p 22217:/data2/zwc/checkpoints
scp -r zwc@122.115.57.205:/mnt/ssd0/data/pi0_agent/zwc/pi0_airbot/airbot_finetune_pi0_experiment/29999 /data2/zwc/checkpoints

# �����
source /home/robot/.zwc/openpi/.venv/bin/activate

# uv pip ��װ���廪Դ��
uv pip install tensorrt==10.9.0.34 -i https://pypi.tuna.tsinghua.edu.cn/simple

# �ն˲�ѯ�ļ���״̬
stat �ļ��� 

# �ļ��дӴ�С����
du -h --max-depth=1 | sort -hr

python scripts/compute_norm_stats.py --config-name pi0_fast_airbot


## Airbot Fintune ##
# ����ת��
python examples/airbot/convert_airbot_data_to_lerobot.py

# ��stat

python scripts/compute_norm_stats.py --config-name pi0_airbot

python scripts/compute_norm_stats.py --config-name pi0_fast_airbot
python scripts/compute_norm_stats.py --config-name pi0_airbot_low_mem_finetune


# ѵ��
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train.py pi0_airbot --exp-name=airbot_finetune_pi0_experiment --overwrite

# ���Կ�ռ��
nvitop


