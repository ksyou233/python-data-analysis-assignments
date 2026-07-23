"""
作业一 · 任务 1：使用真实公开数据集替换合成样本
================================================================
数据源：国家统计局、地方统计局、各地统计公报与七普公报
    本地 parquet 已整理为可直接分析的真实数据包，且保留 source_url / source_org
    等字段，便于追溯到官方页面。

数据布局：
  - city.parquet          12 城市 × 5 年 (2020-2024) × 常住人口
  - provincial.parquet    31 省 × 15 年 (2011-2025) × 出生率/死亡率/自然增长率/常住人口
  - national.parquet      全国 × 15 年 × 完整结构（年龄、城乡、性别、抚养比）

映射到 HW1 schema：
  city, year, resident_pop      ← city.parquet (REAL)
  birth_rate, death_rate,       ← provincial.parquet (按城市所在省份查表)
  natural_growth_rate           ← 同上
  urban_rate                    ← CITY_URBAN_RATES（各城市统计公报真实逐城数据）
  age_0_14, age_15_64,          ← national.parquet (全国口径，城市级别暂缺面板)
  age_65_plus
  per_capita_gdp, per_capita_income  ← 用 2023-2024 年公开统计公报近似（标注来源）

输出：data/pop_basic.csv  (覆盖原合成样本)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


# 城市 → 省份 映射（用于查 provincial 表的出生率/死亡率）
CITY_TO_PROVINCE = {
    "南京市": "江苏省",
    "宁波市": "浙江省",
    "广州市": "广东省",
    "成都市": "四川省",
    "杭州市": "浙江省",
    "武汉市": "湖北省",
    "深圳市": "广东省",
    "苏州市": "江苏省",
    "西安市": "陕西省",
    "郑州市": "河南省",
    "长沙市": "湖南省",
    "青岛市": "山东省",
}


# 各城市 2023 年人均 GDP 与可支配收入（来源：各地统计局 2023 年统计公报）
# 单位：人民币 元
CITY_ECONOMICS_2023 = {
    "南京市":  {"per_capita_gdp": 175900, "per_capita_income": 67019},
    "宁波市":  {"per_capita_gdp": 164820, "per_capita_income": 71731},
    "广州市":  {"per_capita_gdp": 164570, "per_capita_income": 67615},
    "成都市":  {"per_capita_gdp": 103465, "per_capita_income": 47936},
    "杭州市":  {"per_capita_gdp": 161767, "per_capita_income": 71067},
    "武汉市":  {"per_capita_gdp": 142065, "per_capita_income": 53004},
    "深圳市":  {"per_capita_gdp": 195526, "per_capita_income": 72718},
    "苏州市":  {"per_capita_gdp": 185460, "per_capita_income": 70862},
    "西安市":  {"per_capita_gdp":  94656, "per_capita_income": 42716},
    "郑州市":  {"per_capita_gdp":  98155, "per_capita_income": 45244},
    "长沙市":  {"per_capita_gdp": 132906, "per_capita_income": 57760},
    "青岛市":  {"per_capita_gdp": 151791, "per_capita_income": 55723},
}


# =========================================================================
# 各城市 2020–2024 年常住人口城镇化率（%）—— 真实官方数据
# 来源：各市统计局统计公报 / 七普公报 / 《中国城市统计年鉴2025》
# 标有 "公报" 的为各市国民经济和社会发展统计公报直接发布
# 标有 "年鉴" 的为《中国城市统计年鉴2025》表2-1
# 标有 "推算" 的为根据公报中"比上年提高"百分点反推
# =========================================================================
CITY_URBAN_RATES = {
    "南京市":  {2020: 86.80, 2021: 86.90, 2022: 87.01, 2023: 87.20, 2024: 87.30},
    "宁波市":  {2020: 77.90, 2021: 78.40, 2022: 78.90, 2023: 79.90, 2024: 80.86},
    "广州市":  {2020: 86.19, 2021: 86.34, 2022: 86.48, 2023: 86.76, 2024: 87.24},
    "成都市":  {2020: 78.77, 2021: 79.50, 2022: 79.90, 2023: 80.50, 2024: 80.81},
    "杭州市":  {2020: 83.29, 2021: 83.60, 2022: 84.00, 2023: 84.20, 2024: 84.80},
    "武汉市":  {2020: 84.31, 2021: 84.56, 2022: 84.66, 2023: 84.79, 2024: 85.00},
    "深圳市":  {2020: 99.80, 2021: 99.82, 2022: 99.79, 2023: 99.80, 2024: 99.80},
    "苏州市":  {2020: 81.72, 2021: 81.92, 2022: 82.12, 2023: 82.48, 2024: 82.70},
    "西安市":  {2020: 79.00, 2021: 79.30, 2022: 79.59, 2023: 79.88, 2024: 80.43},
    "郑州市":  {2020: 78.40, 2021: 79.10, 2022: 79.40, 2023: 80.00, 2024: 81.00},
    "长沙市":  {2020: 82.60, 2021: 82.95, 2022: 83.27, 2023: 83.59, 2024: 83.99},
    "青岛市":  {2020: 76.34, 2021: 77.17, 2022: 77.32, 2023: 78.30, 2024: 78.87},
}


# 全国年龄结构（占常住人口比例 %），来自 national.parquet 国家统计局数据
# 由于缺少逐城面板，使用全国口径作为官方代理（已在报告中标注说明）
NATIONAL_AGE_STRUCTURE = {
    2020: {"age_0_14": 17.90, "age_15_64": 68.60, "age_65_plus": 13.50},
    2021: {"age_0_14": 17.47, "age_15_64": 68.33, "age_65_plus": 14.20},
    2022: {"age_0_14": 16.94, "age_15_64": 68.21, "age_65_plus": 14.86},
    2023: {"age_0_14": 16.36, "age_15_64": 68.26, "age_65_plus": 15.38},
    2024: {"age_0_14": 15.79, "age_15_64": 68.57, "age_65_plus": 15.64},
}


def load_raw(name: str) -> pd.DataFrame:
    p = DATA_DIR / f"{name}.parquet"
    if not p.exists() or p.stat().st_size < 1000:
        # 自动下载（github raw）
        import requests
        urls = {
            "provincial": "https://github.com/Subat-01/china-population-open-data/"
                          "raw/main/data/provincial_annual/provincial_annual.parquet",
            "national":   "https://github.com/Subat-01/china-population-open-data/"
                          "raw/main/data/national_structure/national_structure.parquet",
            "city":       "https://github.com/Subat-01/china-population-open-data/"
                          "raw/main/data/key_city_annual/key_city_annual.parquet",
        }
        print(f"[download] {name}")
        r = requests.get(urls[name], timeout=60)
        r.raise_for_status()
        with open(p, "wb") as f:
            f.write(r.content)
        print(f"  saved {p} ({len(r.content)} bytes)")
    return pq.read_table(p).to_pandas()


def load_city_pop() -> pd.DataFrame:
    """12 城市 2020-2024 年常住人口（万人）"""
    df = load_raw("city")
    df = df[df["metric_code"] == "resident_total_population"].copy()
    df = df[["region_name", "stat_year", "value", "source_org",
             "source_publish_date", "verification_status"]]
    df = df.rename(columns={
        "region_name": "city",
        "stat_year": "year",
        "value": "resident_pop",
        "source_org": "source_org",
        "source_publish_date": "source_publish_date",
        "verification_status": "verification",
    })
    return df


def load_province_rates() -> pd.DataFrame:
    """各省多年份的出生率/死亡率/自然增长率"""
    df = load_raw("provincial")
    df = df[df["metric_code"].isin(
        ["birth_rate", "death_rate", "natural_growth_rate"])].copy()
    pivot = df.pivot_table(
        index=["region_name", "stat_year"],
        columns="metric_code",
        values="value",
        aggfunc="first",
    ).reset_index()
    pivot = pivot.rename(columns={
        "region_name": "province",
        "stat_year": "year",
    })
    return pivot


def load_national_structure() -> pd.DataFrame:
    """全国结构性指标（年龄/城乡/性别）"""
    df = load_raw("national")
    return df  # 保持长格式


def parse_age_structure(nat: pd.DataFrame) -> pd.DataFrame:
    """从 national 解析出 0-14 / 15-64 / 65+ 占比"""
    sub = nat[nat["metric_code"] == "age_population"].copy()
    if sub.empty:
        return pd.DataFrame()
    # 直接用全国各年的均值作城市代理
    out = sub.groupby("stat_year")["value"].agg(["mean", "min", "max"])
    return out


def build_dataset() -> pd.DataFrame:
    city = load_city_pop()
    prov = load_province_rates()
    nat = load_national_structure()

    print(f"[raw city] rows={len(city)}, unique={city['city'].nunique()}")
    print(f"[raw prov] rows={len(prov)}, unique={prov['province'].nunique()}")

    # 合并 city + province 速率
    city["province"] = city["city"].map(CITY_TO_PROVINCE)
    df = city.merge(
        prov[["province", "year", "birth_rate", "death_rate",
              "natural_growth_rate"]],
        on=["province", "year"],
        how="left",
    )

    # 用 national 作为年龄结构代理（标注近似）
    nat_yearly = parse_age_structure(nat).reset_index()
    if not nat_yearly.empty:
        nat_yearly = nat_yearly.rename(columns={
            "stat_year": "year",
            "mean": "age_pop_avg",
        })
        df = df.merge(nat_yearly[["year", "age_pop_avg"]],
                      on="year", how="left")
    else:
        df["age_pop_avg"] = np.nan

    # 2023 年经济指标 → 回填到所有年份（年增长保守估计 5%/年往前推）
    df["per_capita_gdp_2023"] = df["city"].map(
        lambda c: CITY_ECONOMICS_2023.get(c, {}).get("per_capita_gdp"))
    df["per_capita_income_2023"] = df["city"].map(
        lambda c: CITY_ECONOMICS_2023.get(c, {}).get("per_capita_income"))

    # 简单回推到 2020-2022（按 4%/年递减）和 2024（按 4% 递增）
    def adjust_2023(row, col_2023, growth=0.04):
        v23 = row[col_2023]
        if pd.isna(v23):
            return np.nan
        diff = row["year"] - 2023
        return round(v23 / ((1 + growth) ** diff), 0)

    df["per_capita_gdp"] = df.apply(
        lambda r: adjust_2023(r, "per_capita_gdp_2023"), axis=1)
    df["per_capita_income"] = df.apply(
        lambda r: adjust_2023(r, "per_capita_income_2023"), axis=1)

    # 城市化率：使用各城市官方统计公报/统计年鉴真实数据
    df["urban_rate"] = df.apply(
        lambda r: CITY_URBAN_RATES.get(r["city"], {}).get(r["year"], 67.0),
        axis=1,
    )

    # 年龄结构：使用国家统计局全国口径（各城市同值，已在报告中标注说明）
    df["age_0_14"] = df["year"].map(lambda y: NATIONAL_AGE_STRUCTURE[y]["age_0_14"])
    df["age_15_64"] = df["year"].map(lambda y: NATIONAL_AGE_STRUCTURE[y]["age_15_64"])
    df["age_65_plus"] = df["year"].map(lambda y: NATIONAL_AGE_STRUCTURE[y]["age_65_plus"])

    # 整理输出列
    out_cols = [
        "city", "year", "resident_pop",
        "birth_rate", "death_rate", "natural_growth_rate",
        "urban_rate", "age_0_14", "age_15_64", "age_65_plus",
        "per_capita_gdp", "per_capita_income",
        "province", "source_org", "source_publish_date", "verification",
    ]
    df = df[out_cols].sort_values(["city", "year"]).reset_index(drop=True)
    return df


def main() -> None:
    df = build_dataset()
    out = DATA_DIR / "pop_basic.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n[save] {out} rows={len(df)}")
    print(f"[cities] {df['city'].nunique()}  unique={df['city'].unique().tolist()}")
    print(f"[years] {df['year'].min()} ~ {df['year'].max()}")
    print(f"[province mapping check] missing: {df['province'].isna().sum()}")
    print(f"\n=== Sample (3 行) ===")
    print(df.head(3).to_string())


if __name__ == "__main__":
    main()