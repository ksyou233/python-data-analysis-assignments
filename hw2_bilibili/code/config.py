"""
hw2_bilibili 的本地配置（不提交到 git / 不上传）。
读取优先级：
  1. 环境变量 BILI_SESSDATA
  2. 本模块内 DEFAULT_SESSDATA
若都没有，则视为匿名爬取。
"""

# ============================================================
# SESSDATA：请妥善保管，不要提交到公开仓库
# 用法：
#   方式 A：export BILI_SESSDATA="..." + 改下面 DEFAULT 为空
#   方式 B：直接改下面的 DEFAULT，然后运行即可
# ============================================================
DEFAULT_SESSDATA: str = "1008b6b0%2C1786193693%2C2f000%2A22CjA8bGOSY8nZUYdRQR7Vwdc-Tpi3nbWAhk4CwbW0aDHDzImtJVgwvsGZaB0mZKpvFb0SVjl1QWYxM251UGV1Sy0tNTJUNzVybmlET255WDF3YzZrY1ZwTlE4MFRKNktVeklkUk1jZGpTOTFDalBoMTdaUWFBTTVuR3VoRGNwR3NQQUZaSjBfLUhnIIEC"

DEFAULT_UCONTENT: str = ""

# ============================================================
# 其他可调参数
# ============================================================
TOP_N_VIDEOS: int = 50
COMMENTS_PER_VIDEO: int = 10
REQUEST_TIMEOUT: int = 12
OUTPUT_DIR_NAME: str = "data"


def get_sessdata() -> str:
    """
    返回最终使用的 SESSDATA：
      优先级 = 环境变量 > DEFAULT
    """
    import os
    return os.environ.get("BILI_SESSDATA") or DEFAULT_SESSDATA


def get_ucontent() -> str:
    """可选：浏览器中由部分 JS 设置，请打开 devtools 在 Network 里搜 `bili_jct`/`csrf` 一并粘贴"""
    import os
    return os.environ.get("BILI_UCONTENT") or DEFAULT_UCONTENT
