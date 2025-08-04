# -*- coding: utf-8 -*-
import os
import openpi.shared.download as download

# 设置缓存目录为您指定的路径
os.environ["OPENPI_DATA_HOME"] = "/mnt/ssd0/data/pi0_agent/zwc/cache"

# 下载权重
local_path = download.maybe_download("s3://openpi-assets/checkpoints/pi0_fast_base/params", force_download=True)
print(f"weights download to: {local_path}")