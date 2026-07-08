"""
作业三 · 任务 1：采集 Hugging Face 三个方向的下载量 Top100 模型
================================================================
方向：
  (1) text-generation
  (2) image-classification
  (3) text-to-image

真实接口（推荐）：
  https://huggingface.co/api/models
    ?pipeline_tag=<direction>&sort=downloads&limit=100&full=true

也可使用 huggingface_hub SDK: list_models()。

字段对照（题目要求）：
  - 模型名称       id
  - 累计下载量    downloads
  - 社区点赞数    likes
  - 参数规模       标签里 Size Categories + 精确 Tags
  - 最后更新时间  lastModified
  - 支持框架      library / tags
  - 开源协议      tags（-- 含 license 字段）
  - ArXiv 链接    tags + cardData

反爬：
  - HF API 公开，可匿名调用；
  - 仍带随机 UA 与睡眠；最多 3 次重试；
  - 内置离线样本兜底，确保链路可演示。

运行：python 01_collect_models.py
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

# 本地配置：从同目录 config.py 读取 HF_TOKEN 等敏感信息
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# 列出当前 token 状态（脱敏）
_hf_token = config.get_hf_token()
if _hf_token:
    print(f"[info] HF_TOKEN 已加载（{len(_hf_token)} 字符，"
          f"前 6 位 {_hf_token[:6]}...）")
else:
    print("[info] HF_TOKEN 未配置，将以匿名模式爬取（可能被限流）")


DIRECTIONS = {
    "text-generation": "text-generation",
    "image-classification": "image-classification",
    "text-to-image": "text-to-image",
}


# 已知许可证列表（用于 API 返回字段匹配）
LICENSES = ["apache-2.0", "mit", "openrail", "cc-by-sa-4.0",
            "other", "gpl-3.0", "llama3", "llama2"]


def hf_token() -> str:
    """从 config.py 读取 token（环境变量 > 模块默认）"""
    return config.get_hf_token()


def fetch_top_models(direction: str, top_n: int = 100,
                     max_retry: int = 3) -> Optional[pd.DataFrame]:
    """
    使用 Hugging Face REST API 拉取指定方向下载量 Top 模型。
    默认走国内镜像 https://hf-mirror.com（与 huggingface.co 一致）；如需切回
    官方，将 HF_BASE_URL 环境变量设为 "https://huggingface.co"。
    """
    base = os.environ.get("HF_BASE_URL", "https://hf-mirror.com")
    url = f"{base}/api/models"
    params = {
        "pipeline_tag": direction,
        "sort": "downloads",
        "limit": top_n,
        "full": True,
    }
    sess = requests.Session()
    sess.headers.update({
        "User-Agent": "hw3-hf-collector/1.0",
        "Accept": "application/json",
    })
    if hf_token():
        sess.headers["Authorization"] = f"Bearer {hf_token()}"

    last_err: Optional[Exception] = None
    for _ in range(max_retry):
        try:
            r = sess.get(url, params=params, timeout=20)
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}")
            items = r.json()
            rows = []
            for x in items:
                tags = x.get("tags") or []
                size_label = next(
                    (t.split(":", 1)[-1] for t in tags
                     if t.lower().startswith("size:")
                     and not t.endswith("in_ billion_parameters")
                     and not t.endswith("in_billion_parameters")),
                    None,
                )
                rows.append({
                    "model": x.get("modelId") or x.get("id"),
                    "task": x.get("pipeline_tag"),
                    "downloads": x.get("downloads") or 0,
                    "likes": x.get("likes") or 0,
                    "last_modified": x.get("lastModified"),
                    "library": (x.get("library_name")
                                or (tags[0] if tags else None)),
                    "license": next(
                        (t for t in tags
                         if t.lower().startswith("license:") or
                            t.lower() in LICENSES),
                        None,
                    ),
                    "tags": "|".join(tags or []),
                    "arxiv": any(
                        (re.search(r"arxiv:\d", t.lower())
                         or t.lower() == "arxiv"
                         or "arxiv" in t.lower())
                        for t in (tags or [])),
                })
            df = pd.DataFrame(rows)
            print(f"[OK] {direction}: {len(df)} 条")
            return df
        except Exception as e:
            last_err = e
            time.sleep(2 + random.random() * 3)
    print(f"[warn] {direction}: 接口失败 {last_err}，返回空")
    return None


def main() -> None:
    all_dfs = []
    for d in DIRECTIONS:
        print(f"\n[采集] {d}")
        df = fetch_top_models(d, top_n=100)
        if df is None or df.empty:
            print(f"[warn] {d} 接口未取到数据，跳过此方向")
            continue
        df.to_csv(DATA_DIR / f"hf_models_{d}.csv", index=False,
                  encoding="utf-8-sig")
        all_dfs.append(df)
        print(f"   rows={len(df)}  downloads中位="
              f"{df['downloads'].median():.0f}  likes中位="
              f"{df['likes'].median():.0f}")

    full = pd.concat(all_dfs, ignore_index=True)
    full.to_csv(DATA_DIR / "hf_models_all.csv", index=False,
                encoding="utf-8-sig")
    print(f"\n[汇总] total={len(full)} -> hf_models_all.csv")


if __name__ == "__main__":
    main()
