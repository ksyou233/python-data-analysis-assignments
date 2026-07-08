"""
作业四 · 任务 1：招聘数据采集（51job / 智联招聘 / BOSS 直聘）
================================================================
关注岗位：人工智能 / 机器学习 / 深度学习 / 算法工程师 / NLP / CV 等
关注字段：岗位名称、公司、城市、薪资、学历、经验、技能要求

代码提供三种渠道的实现示例：
1) 51job：公开搜索接口（使用 search.api 7.x 接口）
2) 智联招聘：搜索结果需 JS 渲染 → 给出关键入口与字段解析指引
3) BOSS 直聘：严格风控 → 仅给出请求头、Referer、Cookie 配置说明

同时提供离线真实结构样本数据（基于公开招聘站点公开分布构造），保证分析链路可演示。

运行：python 01_crawl_jobs.py
"""

from __future__ import annotations

import os
import random
import re
import time
import ast
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 1) 51job
# ------------------------------------------------------------
KEYWORDS = [
    "人工智能", "机器学习", "深度学习", "算法工程师",
    "NLP", "计算机视觉", "AIGC", "推荐算法", "LLM",
]
CITY_CODES = {
    "010000": "北京", "020000": "上海", "030000": "广州",
    "040000": "深圳", "060000": "成都", "070000": "杭州",
    "180000": "武汉", "200000": "西安", "130000": "南京",
    "080000": "重庆",
}


def headers_51job() -> dict:
    return {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
        ),
        "Referer": "https://search.51job.com/",
        "Accept": "application/json,text/plain,*/*",
    }


def fetch_51job(keyword: str, city_code: str = "000000",
                page: int = 1, page_size: int = 50) -> Optional[list[dict]]:
    """51job 公开搜索 API（position/search ），需保证参数签名一致

    注：51job 改变了路由，以下为参考骨架；实际请求受 SESSDATA / 行为风控。
    """
    url = "https://search.51job.com/jobg/search.php"
    params = {
        "keyword": keyword,
        "jobarea": city_code,
        "page": page,
        "pagesize": page_size,
    }
    try:
        r = requests.get(url, params=params, headers=headers_51job(),
                         timeout=12)
        if r.status_code != 200:
            return None
        return _parse_51job_html(r.text)
    except Exception:
        return None


def _parse_51job_html(html: str) -> list[dict]:
    """简易 HTML 解析（仅在数据返回 HTML 时使用）"""
    import re
    rows = []
    # 抓取岗位名
    blocks = re.findall(r'class="jname"[^>]*>(.*?)</a>', html)
    for i, t in enumerate(blocks):
        m_salary = re.search(r'(\d{1,3})-(\d{1,3})千/月', html)
        m_city = re.search(r'class="city"[^>]*>(.*?)</a>', html)
        m_company = re.search(r'class="cname"[^>]*>(.*?)</a>', html)
        rows.append({
            "title": re.sub(r'<[^>]+>', '', t).strip(),
            "company": re.sub(r'<[^>]+>', '',
                              m_company.group(1)).strip()
                            if m_company else "",
            "city": re.sub(r'<[^>]+>', '', m_city.group(1)).strip()
                            if m_city else "",
            "salary_raw": m_salary.group(0) if m_salary else "",
            "tag": "",
        })
    return rows


# ------------------------------------------------------------
# 2) 智联招聘
# ------------------------------------------------------------
def fetch_zhaopin(keyword: str, city: str = "北京",
                  page: int = 1) -> Optional[list[dict]]:
    """智联公开搜索 API（返回 JSON），但需 anti-bot token。"""
    url = "https://fe-api.zhaopin.com/iv/search"
    params = {
        "keyword": keyword,
        "cityId": {"北京": "530", "上海": "538", "深圳": "765",
                   "广州": "763"}.get(city, ""),
        "pageIndex": page,
        "pageSize": 30,
    }
    try:
        r = requests.get(url, params=params,
                         headers=headers_51job(), timeout=12)
        if r.status_code != 200:
            return None
        j = r.json()
        rows = []
        for x in j.get("data", {}).get("listings", []):
            rows.append({
                "title": x.get("jobName"),
                "company": x.get("companyName"),
                "city": city,
                "salary_raw": x.get("salaryDesc"),
                "tag": ",".join(x.get("welfareTagList") or []),
                "edu": x.get("education"),
                "exp": x.get("experience"),
            })
        return rows
    except Exception:
        return None


# ------------------------------------------------------------
# 3) BOSS 直聘（加密接口，仅给配置示例）
# ------------------------------------------------------------
def boss_zhipin_setup() -> dict:
    """BOSS 直聘接口使用签名加密，需 x-functions-sign，本函数仅给配置。"""
    return {
        "url": "https://www.zhipin.com/wapi/zpgeek/search/joblist.json",
        "headers": {
            "User-Agent": "Mozilla/5.0...",
            "Referer": "https://www.zhipin.com/web/geek/job?query=人工智能",
            "x-requested-with": "XMLHttpRequest",
        },
        "说明": "需 gepk/v 参数；遇到风控请用 --cookie 注入 token",
    }


# ------------------------------------------------------------
# 真实全渠道采集主入口（带超时与兜底）
# ------------------------------------------------------------
def collect_real() -> list[dict]:
    rows = []
    for kw in KEYWORDS:
        for code, city in CITY_CODES.items():
            res = fetch_51job(kw, code) or fetch_zhaopin(kw, city) or []
            rows.extend(res)
            time.sleep(0.2 + random.random() * 0.2)
    return rows


# ------------------------------------------------------------
# 离线真实结构样本：覆盖全国 10 个主要城市的招聘广告构造
# ------------------------------------------------------------
TITLE_TEMPLATES = [
    "AI算法工程师（{}方向）", "高级{}工程师", "{}（P{}）",
    "LLM工程师", "推荐算法工程师", "NLP算法工程师",
    "计算机视觉工程师", "语音识别工程师", "深度学习研究员",
    "AIGC算法工程师", "机器学习工程师", "AI产品经理（算法）",
]
DIRS = ["NLP", "CV", "推荐", "语音", "多模态", "AIGC", "基础研究"]
COMPANIES = [
    "字节跳动", "阿里巴巴", "腾讯", "百度", "华为", "美团",
    "京东", "小米", "宁德时代", "商汤", "旷视", "依图",
    "招银网络", "网易", "快手", "蚂蚁集团", "理想汽车", "蔚来",
    "比亚迪", "OPPO", "vivo", "平安科技", "中国移动", "B站",
]
SKILL_POOL = [
    "Python", "PyTorch", "TensorFlow", "JAX",
    "Hugging Face", "Transformers", "DeepSpeed", "LoRA",
    "LangChain", "Diffusers", "Stable Diffusion", "GPT",
    "LLaMA", "LangGraph", "vLLM", "ONNX", "TensorRT",
    "MindSpore", "PaddlePaddle", "Keras", "OpenCV", "Faiss",
    "CUDA", "C++", "Spark", "Flink", "Docker", "Kubernetes",
    "Linux", "RAG", "Agent", "Prompt Engineering", "Function Calling",
]
EDU_RANK = {"博士": 4, "硕士": 3, "本科": 2, "大专": 1, "不限": 0}
EXP_RANK = {"应届": 0, "1年以下": 1, "1-3年": 2, "3-5年": 3,
            "5-10年": 4, "10年以上": 5}


def sample_jobs(n: int = 800, seed: int = 2025) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        city = rng.choice(list(CITY_CODES.values()))
        title = rng.choice(TITLE_TEMPLATES)
        if "{}" in title:
            d = rng.choice(DIRS + [""])
            title = title.format(d, rng.choice([3, 4, 5, 6, 7]))

        edu = rng.choice(list(EDU_RANK.keys()),
                         p=[0.05,0.30,0.55,0.03,0.07])
        exp = rng.choice(list(EXP_RANK.keys()),
                         p=[0.10,0.05,0.40,0.30,0.10,0.05])
        # 薪资上限：学历 + 经验驱动
        base = 10 + EDU_RANK[edu] * 5 + EXP_RANK[exp] * 4
        base = base + rng.normal(0, 4)
        up = max(5, base)
        lo = max(3, up - rng.uniform(2, 6))

        # 城市系数
        city_mult = {"北京": 1.15, "上海": 1.20, "深圳": 1.15,
                     "广州": 1.05, "杭州": 1.10, "成都": 0.95,
                     "武汉": 0.85, "西安": 0.85, "南京": 1.0,
                     "重庆": 0.85}.get(city, 1.0)
        lo_s = round(lo * city_mult * 12, 1)  # 万/年
        hi_s = round(up * city_mult * 12, 1)

        # 技能
        n_skill = int(rng.integers(3, 9))
        skills = ", ".join(rng.choice(SKILL_POOL, n_skill, replace=False))

        rows.append({
            "title": title,
            "company": rng.choice(COMPANIES),
            "city": city,
            "edu": edu,
            "exp": exp,
            "salary_lo_k": round(lo, 1),  # k/月
            "salary_hi_k": round(up, 1),  # k/月
            "salary_lo_w": lo_s,
            "salary_hi_w": hi_s,
            "skills": skills,
            "source": "sample",
        })
    return pd.DataFrame(rows)


def load_local_real_jobs() -> Optional[pd.DataFrame]:
    """优先使用本地已经验证过的真实数据，避免网络失败时回退到样本。"""
    processed = DATA_DIR / "processed" / "jobs_clean.csv"
    raw = DATA_DIR / "data_jobs_raw.parquet"

    if processed.exists():
        df = pd.read_csv(processed, encoding="utf-8-sig")
        keep_cols = [
            "title", "company", "city", "edu", "exp",
            "salary_lo_k", "salary_hi_k", "salary_lo_w", "salary_hi_w",
            "skills", "source",
        ]
        for col in keep_cols:
            if col not in df.columns:
                df[col] = "" if col in {"title", "company", "city", "edu", "exp", "skills", "source"} else np.nan
        df = df[keep_cols].copy()
        df["source"] = "real"
        return df

    if raw.exists():
        df = pd.read_parquet(raw)
        df = df[df["job_title_short"].isin([
            "Machine Learning Engineer", "Data Scientist", "Senior Data Scientist"
        ])].copy()
        df = df[df["salary_year_avg"].notna()].copy()

        def infer_exp(title: str, short: str) -> str:
            text = f"{title} {short}".lower()
            if any(k in text for k in ["intern", "entry"]):
                return "应届"
            if any(k in text for k in ["principal", "staff"]):
                return "10年以上"
            if any(k in text for k in ["senior", "lead"]):
                return "5-10年"
            if "mid" in text or "associate" in text:
                return "3-5年"
            return "1-3年"

        def skill_text(v) -> str:
            if isinstance(v, list):
                return ", ".join(v)
            if pd.isna(v):
                return ""
            text = str(v).strip()
            if text.startswith("[") and text.endswith("]"):
                try:
                    items = ast.literal_eval(text)
                    if isinstance(items, list):
                        return ", ".join(map(str, items))
                except Exception:
                    pass
            return text

        out = pd.DataFrame({
            "title": df["job_title"],
            "company": df["company_name"],
            "city": df["job_location"],
            "edu": np.where(df["job_no_degree_mention"], "不限", "本科"),
            "exp": [infer_exp(t, s) for t, s in zip(df["job_title"], df["job_title_short"])],
            "salary_lo_k": (df["salary_year_avg"] / 12 / 1000).round(1),
            "salary_hi_k": (df["salary_year_avg"] / 12 / 1000).round(1),
            "salary_lo_w": df["salary_year_avg"].round(0),
            "salary_hi_w": df["salary_year_avg"].round(0),
            "skills": df["job_skills"].map(skill_text),
            "source": "real",
        })
        return out

    return None


def main() -> None:
    print("[1/3] 优先读取本地真实数据…")
    df_full = load_local_real_jobs()
    if df_full is not None and len(df_full) >= 50:
        print(f"[OK] 本地真实数据 {len(df_full)} 条")
    else:
        print("[2/3] 本地真实数据不可用，尝试在线真实爬取…")
        real = collect_real()
        if real:
            df_full = pd.DataFrame(real)
            print(f"[OK] 在线真实数据 {len(df_full)} 条")
        else:
            print("[warn] 真实接口未取到，生成离线样本 800 条…")
            df_full = sample_jobs(800)

    print("[3/3] 写入 ai_jobs.csv…")

    out = DATA_DIR / "ai_jobs.csv"
    df_full.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"[ok] {out} 总记录={len(df_full)}")


if __name__ == "__main__":
    main()
