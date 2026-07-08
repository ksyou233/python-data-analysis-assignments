"""
作业二 · 任务 2~4：分区表现分析 + 评论多维度分析 + 扩展挖掘
==============================================================
任务（对应题目要求）：
(2) 排行榜数据：各分区视频数量占比与平均互动率 + 可视化
(3) 评论数据：情感倾向、评论意图、评论质量三维分析
(4) 扩展挖掘：作者分布热力、分区×时段、关键词云等

输入：data/top_videos.csv, data/hot_comments.csv
输出：figures/fig01~fig10.png，data/topic_stats.csv 等

运行：python 02_analyze.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import jieba
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from snownlp import SnowNLP
from wordcloud import WordCloud

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


INTENT_KEYWORDS = {
    "提问": ["?", "？", "怎么", "为什么", "求", "有没有", "哪里",
            "哪个", "一下", "如何"],
    "夸奖": ["太强", "大佬", "牛", "爱了", "宝藏", "棒", "硬核", "厉害"],
    "吐槽": ["不行", "差评", "烂", "失望", "标题党", "太过",
            "不好", "难看"],
    "建议": ["建议", "如果", "其实可以", "希望", "不妨", "可以改成",
            "应该", "加个"],
}


def add_engagement(df: pd.DataFrame) -> pd.DataFrame:
    """互动率 = (点赞 + 弹幕 + 评论) / 播放量（参考 B 站常见口径）"""
    df = df.copy()
    df["interaction_rate"] = (
        (df["like"] + df["danmaku"] + df["reply"]) / df["view"]
    ) * 100
    return df


def plot_district_distribution(df: pd.DataFrame) -> None:
    count = df["tname"].value_counts()
    pct = (count / len(df) * 100).round(1)

    plt.figure(figsize=(11, 5))
    plt.bar(count.index.astype(str), count.values, color="#3182bd")
    for i, v in enumerate(count.values):
        plt.text(i, v + 0.1, f"{pct.iloc[i]}%", ha="center", fontsize=10)
    plt.title("图1 · 全站排行榜 TOP50 各分区视频数量分布与占比")
    plt.ylabel("视频数")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig01_district_count.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[fig] fig01_district_count.png")

    # 数量 + 平均互动率
    grp = df.groupby("tname").agg(
        count=("bvid", "count"),
        avg_view=("view", "mean"),
        avg_like=("like", "mean"),
        avg_rate=("interaction_rate", "mean"),
    ).round(2).sort_values("count", ascending=False)
    grp["count"].to_csv(DATA_DIR / "topic_stats.csv", encoding="utf-8-sig",
                        header=["count"])
    return grp


def plot_district_engagement(grp: pd.DataFrame) -> None:
    df = grp.reset_index().sort_values("avg_rate", ascending=False)
    plt.figure(figsize=(11, 5))
    bars = plt.bar(df["tname"], df["avg_rate"], color="#7bccc4")
    for b, v in zip(bars, df["avg_rate"].values):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.05,
                 f"{v:.2f}%", ha="center", fontsize=10)
    plt.title("图2 · 各分区平均互动率（%（点赞+弹幕+评论）/播放量）")
    plt.ylabel("互动率 (%)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig02_district_engagement.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("[fig] fig02_district_engagement.png")


def plot_view_vs_like(df: pd.DataFrame) -> None:
    plt.figure(figsize=(9, 6))
    plt.scatter(np.log10(df["view"] + 1), np.log10(df["like"] + 1),
                c="#756bb1", alpha=0.8, s=40)
    for _, r in df.iterrows():
        if r["view"] > 5_000_000:
            plt.annotate(r["title"][:14], (np.log10(r["view"] + 1),
                                            np.log10(r["like"] + 1)),
                         textcoords="offset points", xytext=(3, 3),
                         fontsize=8, color="gray")
    plt.xlabel("log10(播放量)")
    plt.ylabel("log10(点赞)")
    plt.title("图3 · TOP50 播放量 vs 点赞（对数尺度）")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig03_view_vs_like.png", dpi=150,
                bbox_inches="tight")
    plt.close()
    print("[fig] fig03_view_vs_like.png")


# ------------------------------------------------------------
# 评论多维度分析
# ------------------------------------------------------------
def sentiment_score(text: str) -> float:
    """使用 SnowNLP 给 0–1 的正向概率；返回 0.4–0.6 视为中性。"""
    try:
        s = SnowNLP(text)
        return s.sentiments
    except Exception:
        return 0.5


def classify_sentiment(score: float) -> str:
    if score >= 0.6: return "正"
    if score <= 0.4: return "负"
    return "中"


def classify_intent(text: str) -> str:
    for label, kws in INTENT_KEYWORDS.items():
        for kw in kws:
            if kw in text:
                return label
    return "陈述"


def comment_quality(text: str, likes: int) -> float:
    """简单质量评分：长度分 + 点赞分，加权"""
    length = len(text)
    length_score = min(length / 30.0, 1.0) * 0.5
    like_score = min(np.log1p(likes) / np.log1p(1000), 1.0) * 0.5
    return round(length_score + like_score, 3)


def main() -> None:
    df_v = pd.read_csv(DATA_DIR / "top_videos.csv", encoding="utf-8-sig")
    df_c = pd.read_csv(DATA_DIR / "hot_comments.csv", encoding="utf-8-sig")
    print(f"[data] videos={len(df_v)}, comments={len(df_c)}")

    df_v = add_engagement(df_v)
    grp = plot_district_distribution(df_v)
    plot_district_engagement(grp)
    plot_view_vs_like(df_v)

    # 评论多维分析
    df_c["sent_score"] = df_c["message"].astype(str).map(sentiment_score)
    df_c["sentiment"] = df_c["sent_score"].map(classify_sentiment)
    df_c["intent"] = df_c["message"].astype(str).map(classify_intent)
    df_c["quality"] = [
        comment_quality(str(m), int(l))
        for m, l in zip(df_c["message"], df_c["like"])
    ]
    df_c.to_csv(DATA_DIR / "comments_analyzed.csv", index=False,
                encoding="utf-8-sig")
    print("[ok] comments_analyzed.csv 已分析 + 保存")

    # 情感分布
    plt.figure(figsize=(7, 5))
    counts = df_c["sentiment"].value_counts()
    plt.pie(counts.values, labels=counts.index, autopct="%.1f%%",
            colors=["#74c476", "#fb6a4a", "#fdae6b"])
    plt.title("图4 · 热门评论情感分布")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig04_sentiment.png", dpi=150,
                bbox_inches="tight")
    plt.close()
    print("[fig] fig04_sentiment.png")

    # 意图分布
    plt.figure(figsize=(8, 5))
    intent = df_c["intent"].value_counts().sort_values(ascending=False)
    plt.bar(intent.index, intent.values, color="#9e9ac8")
    for i, v in enumerate(intent.values):
        plt.text(i, v + 1, str(v), ha="center")
    plt.title("图5 · 热门评论意图分类")
    plt.ylabel("评论数")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig05_intent.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[fig] fig05_intent.png")

    # 质量分布
    plt.figure(figsize=(8, 5))
    plt.hist(df_c["quality"], bins=20, color="#9ecae1", edgecolor="white")
    plt.title("图6 · 评论质量分数分布")
    plt.xlabel("质量分数 (0–1)")
    plt.ylabel("评论数")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig06_quality.png", dpi=150,
                bbox_inches="tight")
    plt.close()
    print("[fig] fig06_quality.png")

    # ============================================================
    # 任务(4)：扩展挖掘
    # ============================================================
    # 4.1 各分区的情感分布热力
    cross = pd.crosstab(df_c["bvid"], df_c["sentiment"])
    # 取视频所属分区拼回去
    bvid2tname = df_v.set_index("bvid")["tname"]
    cross["tname"] = cross.index.map(bvid2tname)
    melted = cross.melt(id_vars="tname", var_name="sentiment",
                        value_name="count").fillna(0)
    grouped = melted.groupby(["tname", "sentiment"]).sum().reset_index()
    pivot = grouped.pivot(index="tname", columns="sentiment",
                          values="count").fillna(0)
    plt.figure(figsize=(9, 6))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlOrRd")
    plt.title("图7 · 各分区评论情感计数热力")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig07_district_sentiment.png", dpi=150,
                bbox_inches="tight")
    plt.close()
    print("[fig] fig07_district_sentiment.png")

    # 4.2 发布时段的视频分布（按时戳 -> 小时）
    df_v["hour"] = pd.to_datetime(df_v["pubdate"], unit="s").dt.hour
    pivot_hour = df_v.groupby(["tname", "hour"]).size().unstack(fill_value=0)
    plt.figure(figsize=(11, 6))
    sns.heatmap(pivot_hour, cmap="Blues", cbar_kws={"label": "视频数"})
    plt.title("图8 · 分区×发布小时 视频数热力")
    plt.xlabel("小时")
    plt.ylabel("分区")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig08_district_hour.png", dpi=150,
                bbox_inches="tight")
    plt.close()
    print("[fig] fig08_district_hour.png")

    # 4.3 平均情感 vs 播放量（视频级聚合）
    sent_per_video = df_c.groupby("bvid")["sent_score"].mean()
    df_v = df_v.merge(sent_per_video.rename("avg_sent"), on="bvid",
                      how="left")
    plt.figure(figsize=(9, 6))
    plt.scatter(np.log10(df_v["view"] + 1), df_v["avg_sent"].fillna(0.5),
                c=df_v["interaction_rate"], cmap="viridis", s=50,
                edgecolor="k", alpha=0.8)
    plt.colorbar(label="互动率(%)")
    plt.xlabel("log10(播放量)")
    plt.ylabel("平均评论情感正向性")
    plt.title("图9 · 播放量 vs 平均评论情感正向性（颜色=互动率）")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig09_view_vs_sentiment.png", dpi=150,
                bbox_inches="tight")
    plt.close()
    print("[fig] fig09_view_vs_sentiment.png")

    # 4.4 评论词云
    text = " ".join([
        re.sub(r"[^一-龥a-zA-Z0-9 ]", " ", str(m))
        for m in df_c["message"]
    ])
    segs = [w for w in jieba.cut(text) if len(w) > 1]
    body = " ".join(segs)
    wc = WordCloud(
        font_path="C:/Windows/Fonts/msyh.ttc",
        width=900, height=520,
        background_color="white",
        max_words=200,
        collocations=False,
    ).generate(body)
    plt.figure(figsize=(11, 6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title("图10 · 热门评论词云")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig10_wordcloud.png", dpi=150,
                bbox_inches="tight")
    plt.close()
    print("[fig] fig10_wordcloud.png")

    # 汇总
    print("\n========= 主要汇总 =========")
    print(grp)
    print("\n情感分布:\n", df_c["sentiment"].value_counts())
    print("\n意图分布:\n", df_c["intent"].value_counts())


if __name__ == "__main__":
    main()
