"""
hw4_recruit 的本地配置
"""
import os

# 复用 HF token，但不要在仓库里明文保存
DEFAULT_HF_TOKEN = ""


def get_hf_token() -> str:
    return os.environ.get("HF_TOKEN") or DEFAULT_HF_TOKEN


# 数据源 URL（开源招聘数据集，可下载全部）
DATASET_URLS = {
    "data_jobs_parquet":
        "https://huggingface.co/datasets/lukebarousse/data_jobs/resolve/"
        "refs%2Fconvert%2Fparquet/default/train/0000.parquet",
}