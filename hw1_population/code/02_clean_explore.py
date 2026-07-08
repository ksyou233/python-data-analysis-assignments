"""
作业一 · 任务 2&3：数据清洗 + 描述性分析 + 可视化
================================================================
输入：data/pop_basic.csv
输出：data/pop_basic_clean.csv，figures/*.png

可视化内容：
- 各城市常住人口趋势（折线）
- 各城市自然增长率变化（折线）
- 老龄化（65+ 占比）变化（折线）
- 城镇化率变化（折线）
- 指标相关性热力图
- 2024 vs 2015 对比柱状图（人口变化量）

运行：python 02_clean_explore.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 解决中文字体（Windows）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_and_clean() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "pop_basic.csv", encoding="utf-8-sig")
    # 缺失值
    miss = df.isna().sum().sum()
    print(f"[清洗前] 行数={len(df)} 总缺失={miss}")

    # 类型修正
    df["city"] = df["city"].astype(str)
    df["year"] = df["year"].astype(int)

    # 异常值：出生率、死亡率夹紧到合理区间；自然增长率保留负值
    for col in ["birth_rate", "death_rate"]:
        df[col] = df[col].clip(0, 30)
    df["natural_growth_rate"] = df["natural_growth_rate"].clip(-20, 20)
    # 比例类夹紧到 [0, 100]
    for col in ["urban_rate", "age_0_14", "age_15_64", "age_65_plus"]:
        df[col] = df[col].clip(0, 100)

    # 人均GDP/收入：负数/缺失以全国均值填补
    for col in ["per_capita_gdp", "per_capita_income"]:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
        df[col] = df[col].clip(lower=0)

    # 重复行
    df = df.drop_duplicates(subset=["city", "year"])

    miss_after = df.isna().sum().sum()
    print(f"[清洗后] 行数={len(df)} 总缺失={miss_after}")
    return df


def plot_trends(df: pd.DataFrame) -> None:
    cities = sorted(df["city"].unique())

    def lineplot(metric: str, title: str, ylabel: str, fname: str) -> None:
        plt.figure(figsize=(10, 6))
        for c in cities:
            sub = df[df["city"] == c]
            plt.plot(sub["year"], sub[metric], marker="o", label=c)
        plt.title(title)
        plt.xlabel("年份")
        plt.ylabel(ylabel)
        plt.legend(ncol=5, fontsize=9, loc="upper right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(FIG_DIR / fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[fig] {fname}")

    lineplot("resident_pop", "2020-2024 各城市常住人口（万人）", "人口（万人）", "fig01_pop_trend.png")
    lineplot("natural_growth_rate", "自然增长率变化（‰）", "自然增长率（‰）", "fig02_natural_growth.png")
    lineplot("age_65_plus", "65岁以上人口占比变化（老龄化趋势）", "65+ 占比（%）", "fig03_aging.png")
    lineplot("urban_rate", "城镇化率变化（%）", "城镇化率（%）", "fig04_urban.png")
    lineplot("birth_rate", "出生率变化（‰）", "出生率（‰）", "fig05_birth_rate.png")


def plot_correlation(df: pd.DataFrame) -> None:
    cols = [
        "resident_pop", "birth_rate", "death_rate", "natural_growth_rate",
        "urban_rate", "age_65_plus", "per_capita_gdp", "per_capita_income",
    ]
    corr = df[cols].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                cbar_kws={"shrink": 0.8})
    plt.title("人口/经济指标皮尔逊相关性热力图")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig06_corr.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[fig] fig06_corr.png")


def plot_change_bar(df: pd.DataFrame) -> None:
    pivot = df.pivot_table(index="city", columns="year", values="resident_pop")
    y0, y1 = int(pivot.columns.min()), int(pivot.columns.max())
    delta = (pivot[y1] - pivot[y0]).sort_values()
    plt.figure(figsize=(9, 5))
    colors = ["#d62728" if v < 0 else "#2ca02c" for v in delta.values]
    plt.barh(delta.index, delta.values, color=colors)
    plt.axvline(0, color="black", lw=0.8)
    plt.title(f"{y0}→{y1} 常住人口变化（万人）")
    plt.xlabel("人口变化（万人）")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig07_pop_change.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[fig] fig07_pop_change.png")


def plot_urban_vs_aging(df: pd.DataFrame) -> None:
    latest = df[df["year"] == 2024].copy()
    plt.figure(figsize=(8, 6))
    sizes = (latest["per_capita_gdp"] / 1000).clip(lower=20)
    sc = plt.scatter(
        latest["urban_rate"], latest["age_65_plus"],
        s=sizes, alpha=0.65, c=latest["natural_growth_rate"], cmap="RdYlGn",
        edgecolor="k",
    )
    for _, r in latest.iterrows():
        plt.annotate(r["city"], (r["urban_rate"], r["age_65_plus"]),
                     textcoords="offset points", xytext=(4, 4), fontsize=9)
    plt.colorbar(sc, label="自然增长率（‰）")
    plt.xlabel("城镇化率（%）")
    plt.ylabel("65+ 占比（%）")
    plt.title("2024 年 10 城：城镇化 vs 老龄化（气泡大小=人均GDP/千）")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig08_urban_vs_aging.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[fig] fig08_urban_vs_aging.png")


def main() -> None:
    df = load_and_clean()
    df.to_csv(DATA_DIR / "pop_basic_clean.csv", index=False, encoding="utf-8-sig")
    print(f"[save] {DATA_DIR / 'pop_basic_clean.csv'}")
    plot_trends(df)
    plot_correlation(df)
    plot_change_bar(df)
    plot_urban_vs_aging(df)
    print("done.")


if __name__ == "__main__":
    main()
