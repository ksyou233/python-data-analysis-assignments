"""
作业四 · 任务 1：使用真实公开数据集替换合成样本
================================================================
数据源：lukebarousse/data_jobs (Apache-2.0)
  https://huggingface.co/datasets/lukebarousse/data_jobs
  785,741 条真实 LinkedIn / 全球招聘职位

映射到 HW4 原始字段：
  title   <- job_title
  company <- company_name
  city    <- job_location
  edu     <- 从 job_no_degree_mention 推导（提及无需学历=不限；未提=本科）
  exp     <- 从 job_title (Senior/Mid/Entry) 推导
  salary  <- salary_year_avg 转换为千/月
  skills  <- job_skills (Python-list 转为逗号分隔字符串)
  source  <- 'real'

输出：data/ai_jobs.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


EDU_FROM_NO_DEGREE = {
    True: "不限",     # 不要求学位
    False: "本科",    # 默认要求本科学位
}


def infer_exp(title: str) -> str:
    """从 job_title 关键词推导经验档位（粗估）"""
    t = str(title).lower()
    if any(k in t for k in ["principal", "staff", "distinguished",
                             "lead ", "head "]):
        return "10年以上"
    if "senior" in t or "sr." in t or "sr " in t:
        return "5-10年"
    if any(k in t for k in ["junior", "jr.", "jr ", "entry", "intern"]):
        return "应届"
    if any(k in t for k in ["ii ", " iii", " iii ", " mid"]):
        return "3-5年"
    return "1-3年"


def filter_ai_jobs(df: pd.DataFrame) -> pd.DataFrame:
    """筛选 AI 相关职位（且有薪资字段）"""
    ai_titles = [
        "Machine Learning Engineer",
        "Data Scientist",
        "Senior Data Scientist",
    ]
    keep = (
        df["job_title_short"].isin(ai_titles)
        & df["salary_year_avg"].notna()
        & (df["salary_year_avg"] > 1000)
        & (df["salary_year_avg"] < 1_000_000)
    )
    return df[keep].copy()


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """把 17 列映射到 HW4 schema"""
    out = pd.DataFrame({
        "title": df["job_title"].astype(str),
        "company": df["company_name"].fillna("Unknown").astype(str),
        "city": df["job_location"].fillna(df["job_country"]).astype(str),
        "edu": df["job_no_degree_mention"].map(EDU_FROM_NO_DEGREE)
              .fillna("本科"),
        "exp": df["job_title"].map(infer_exp),
        # 年薪 -> 千/月 = salary_year_avg / 12 / 1000
        "salary_lo_k": (df["salary_year_avg"] / 12 / 1000).round(1),
        "salary_hi_k": (df["salary_year_avg"] / 12 / 1000).round(1),
        "salary_lo_w": df["salary_year_avg"].round(1),
        "salary_hi_w": df["salary_year_avg"].round(1),
        # skills: 形如 "['python', 'sql']" -> "python,sql"
        "skills": df["job_skills"].apply(
            lambda x: ", ".join(eval(str(x))) if isinstance(x, str)
                       and x.startswith("[") else ""),
        "source": "real",
    })
    return out


def main() -> None:
    src = DATA_DIR / "data_jobs_raw.parquet"
    if not src.exists():
        # 自动下载
        import requests
        url = (
            "https://huggingface.co/datasets/lukebarousse/data_jobs/resolve/"
            "refs%2Fconvert%2Fparquet/default/train/0000.parquet"
        )
        tok = config.get_hf_token()
        print(f"[download] {url}")
        r = requests.get(url, headers={"Authorization": f"Bearer {tok}"},
                         stream=True, timeout=300)
        with open(src, "wb") as f:
            for chunk in r.iter_content(chunk_size=2 * 1024 * 1024):
                f.write(chunk)
        print(f"[ok] saved to {src}")

    print(f"[load] {src}")
    raw = pd.read_parquet(src)
    print(f"[raw] rows={len(raw)}")

    ai = filter_ai_jobs(raw)
    print(f"[filter AI + has salary] rows={len(ai)}")

    out = transform(ai)
    out_path = DATA_DIR / "ai_jobs.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[save] {out_path} rows={len(out)}")
    print("\n=== 5 行样本 ===")
    print(out.head(5).to_string())
    print("\n=== 来源分布 ===")
    print(out["source"].value_counts())
    print("\n=== 国家/地区 (Top10) ===")
    print(out["city"].str.split(",").str[-1].str.strip()
          .value_counts().head(10))


if __name__ == "__main__":
    main()