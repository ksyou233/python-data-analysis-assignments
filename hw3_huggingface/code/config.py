"""
hw3_huggingface 的本地配置（不提交到 git / 不上传）。
读取优先级：
  1. 环境变量 HF_TOKEN
  2. 本模块内 DEFAULT_HF_TOKEN
"""

# ============================================================
# HF_TOKEN：请通过环境变量提供，不要在仓库里写死
# ============================================================
DEFAULT_HF_TOKEN: str = ""

# ============================================================
# 其他可调参数
# ============================================================
TOP_N_PER_DIRECTION: int = 100
REQUEST_TIMEOUT: int = 20
OUTPUT_DIR_NAME: str = "data"


def get_hf_token() -> str:
    """
    返回最终使用的 HF_TOKEN：
      优先级 = 环境变量 > DEFAULT
    """
    import os
    return os.environ.get("HF_TOKEN") or DEFAULT_HF_TOKEN
