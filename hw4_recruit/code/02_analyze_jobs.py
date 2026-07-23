"""
作业四 · 任务 2~4：特征工程 + 可视化 + 薪资预测
====================================================
任务（对应题目要求）：
  (2) 岗位类型、城市、薪资、学历、经验、技能需求 → 清洗 + 特征工程
  (3) 描述性可视化 + 关键洞见
  (4) 薪资预测建模（线性回归 / 随机森林 / XGBoost）
  + 不同岗位方向的门槛与薪资差异

输入：data/ai_jobs.csv
输出：figures/fig01-fig09.png，data/processed/*.csv，data/insights.md

运行：python 02_analyze_jobs.py
"""

from __future__ import annotations

from pathlib import Path

import jieba
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud

import xgboost as xgb

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PROC_DIR = DATA_DIR / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


EDU_RANK = {"博士": 4, "硕士": 3, "本科": 2, "大专": 1, "不限": 0}
EXP_RANK = {"应届": 0, "1年以下": 1, "1-3年": 2, "3-5年": 3,
            "5-10年": 4, "10年以上": 5}


def classify_job(title: str) -> str:
    """根据标题分类为大方向"""
    title = str(title)
    if "NLP" in title or "大模型" in title or "LLM" in title or "AIGC" in title:
        return "NLP/大模型"
    if "视觉" in title or "CV" in title or "图像" in title:
        return "CV/视觉"
    if "推荐" in title:
        return "推荐"
    if "语音" in title:
        return "语音"
    if "多模态" in title:
        return "多模态"
    if "研究员" in title or "研究" in title:
        return "基础研究"
    if "产品" in title:
        return "AI产品"
    return "通用算法/机器学习"


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["edu_rank"] = df["edu"].map(EDU_RANK).fillna(0)
    df["exp_rank"] = df["exp"].map(EXP_RANK).fillna(0)
    df["salary_mid_w"] = (df["salary_lo_w"] + df["salary_hi_w"]) / 2  # 万元/年
    df["salary_mid_k"] = (df["salary_lo_k"] + df["salary_hi_k"]) / 2  # 千/月

    df["direction"] = df["title"].map(classify_job)
    df["n_skills"] = df["skills"].fillna("").apply(
        lambda x: len([s for s in x.split(",") if s.strip()]))
    # 技能命中特征
    keyskills = ["PyTorch", "TensorFlow", "Hugging Face", "Transformers",
                 "LoRA", "Diffusers", "LangChain", "vLLM",
                 "CUDA", "C++", "RAG", "Agent", "MindSpore"]
    for sk in keyskills:
        df[f"has_{sk}"] = df["skills"].fillna("").str.contains(sk).astype(int)

    # 城市分层
    tier = {"北京": "T1", "上海": "T1", "深圳": "T1", "杭州": "T1.5",
            "广州": "T1.5", "成都": "T2", "南京": "T1.5",
            "武汉": "T2", "西安": "T2", "重庆": "T2"}
    df["city_tier"] = df["city"].map(tier).fillna("Other")
    return df


def plot_salary_distribution(df: pd.DataFrame) -> None:
    plt.figure(figsize=(9, 5))
    sns.histplot(df["salary_mid_w"], bins=40, kde=True, color="#3182bd")
    plt.title("图1 · AI 岗位中位薪资分布（万元/年）")
    plt.xlabel("中位薪资（万元/年）")
    plt.ylabel("岗位数")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig01_salary_dist.png", dpi=150,
                bbox_inches="tight")
    plt.close()
    print("[fig] fig01_salary_dist.png")


def plot_city_distribution(df: pd.DataFrame) -> None:
    city_count = df["city"].value_counts().head(10)
    plt.figure(figsize=(10, 5))
    bars = plt.bar(city_count.index, city_count.values, color="#74c476")
    for b, v in zip(bars, city_count.values):
        plt.text(b.get_x() + b.get_width() / 2, v + 1, str(v),
                 ha="center", fontsize=9)
    plt.title("图2 · AI 岗位 城市分布 Top10")
    plt.ylabel("岗位数")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig02_city_count.png", dpi=150,
                bbox_inches="tight")
    plt.close()
    print("[fig] fig02_city_count.png")


def plot_direction_salary(df: pd.DataFrame) -> None:
    order = (df.groupby("direction")["salary_mid_w"].median()
             .sort_values(ascending=False).index)
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df, x="direction", y="salary_mid_w", order=order,
                palette="Set2")
    plt.title("图3 · 不同方向薪资箱线图（万元/年）")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig03_direction_salary.png", dpi=150,
                bbox_inches="tight")
    plt.close()
    print("[fig] fig03_direction_salary.png")

    # 平均薪资
    avg = df.groupby("direction")["salary_mid_w"].agg(
        ["mean", "median", "count"]).round(2).sort_values("mean",
                                                         ascending=False)
    avg.to_csv(PROC_DIR / "direction_salary.csv", encoding="utf-8-sig")


def plot_edu_salary(df: pd.DataFrame) -> None:
    order = ["不限", "大专", "本科", "硕士", "博士"]
    plt.figure(figsize=(9, 5))
    sns.boxplot(data=df, x="edu", y="salary_mid_w", order=order,
                palette="Set3")
    plt.title("图4 · 不同学历的薪资分布（万元/年）")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig04_edu_salary.png", dpi=150,
                bbox_inches="tight")
    plt.close()
    print("[fig] fig04_edu_salary.png")


def plot_exp_salary(df: pd.DataFrame) -> None:
    order = list(EXP_RANK.keys())
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df, x="exp", y="salary_mid_w", order=order,
                palette="Set1")
    plt.title("图5 · 不同经验的薪资分布（万元/年）")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig05_exp_salary.png", dpi=150,
                bbox_inches="tight")
    plt.close()
    print("[fig] fig05_exp_salary.png")


def plot_skills_wordcloud(df: pd.DataFrame) -> None:
    text = " ".join(jieba.cut(" ".join(df["skills"].fillna("").tolist())))
    # 把所有 SK 用空格分开重新组合，去掉单字
    text = re.sub(r"[^\w\s]", " ", text) if False else text
    text = " ".join([w for w in text.split() if len(w) > 1])
    wc = WordCloud(
        font_path="C:/Windows/Fonts/msyh.ttc",
        width=900, height=520,
        background_color="white",
        max_words=200,
    ).generate(text)
    plt.figure(figsize=(11, 6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title("图6 · 技能关键词词云")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig06_skills_wordcloud.png", dpi=150,
                bbox_inches="tight")
    plt.close()
    print("[fig] fig06_skills_wordcloud.png")


def plot_skill_frequency(df: pd.DataFrame) -> None:
    skills = (df["skills"].fillna("")
                .str.split(",")
                .explode()
                .str.strip())
    skills = skills[skills != ""]
    top = skills.value_counts().head(20)
    plt.figure(figsize=(10, 6))
    plt.barh(top.index[::-1], top.values[::-1], color="#6baed6")
    plt.title("图7 · 高频技能 Top20")
    plt.xlabel("出现岗位数")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig07_skills_freq.png", dpi=150,
                bbox_inches="tight")
    plt.close()
    print("[fig] fig07_skills_freq.png")
    top.to_csv(PROC_DIR / "skill_freq.csv", encoding="utf-8-sig",
               header=["count"])


def model_salary(df: pd.DataFrame) -> dict:
    # 数值特征 + 城市 one-hot + 方向 one-hot + 技能命中
    cat_cols = ["city_tier", "direction"]
    num_cols = ["edu_rank", "exp_rank", "n_skills"] + \
        [c for c in df.columns if c.startswith("has_")]
    X = pd.concat([
        pd.get_dummies(df[cat_cols].astype(str)),
        df[num_cols].astype(float)
    ], axis=1)
    y = df["salary_mid_w"].values

    X_tr, X_te, y_tr, y_te = train_test_split(
        X.values, y, test_size=0.2, random_state=42)

    # Ridge 基线
    ridge = Ridge(alpha=1.0).fit(X_tr, y_tr)
    ytr_r = ridge.predict(X_tr); yte_r = ridge.predict(X_te)
    # RF
    rf = RandomForestRegressor(
        n_estimators=400, max_depth=8, min_samples_leaf=5,
        random_state=42, n_jobs=-1).fit(X_tr, y_tr)
    yte_rf = rf.predict(X_te)
    # XGB
    xgbr = xgb.XGBRegressor(
        n_estimators=500, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        random_state=42, n_jobs=-1, verbosity=0).fit(X_tr, y_tr)
    yte_xgb = xgbr.predict(X_te)

    metrics = pd.DataFrame({
        "model": ["Ridge", "RandomForest", "XGBoost"],
        "R2_test": [
            round(r2_score(y_te, yte_r), 4),
            round(r2_score(y_te, yte_rf), 4),
            round(r2_score(y_te, yte_xgb), 4),
        ],
        "RMSE": [
            round(np.sqrt(mean_squared_error(y_te, yte_r)), 4),
            round(np.sqrt(mean_squared_error(y_te, yte_rf)), 4),
            round(np.sqrt(mean_squared_error(y_te, yte_xgb)), 4),
        ],
        "MAE": [
            round(mean_absolute_error(y_te, yte_r), 4),
            round(mean_absolute_error(y_te, yte_rf), 4),
            round(mean_absolute_error(y_te, yte_xgb), 4),
        ],
    })
    metrics.to_csv(PROC_DIR / "model_metrics.csv", index=False,
                   encoding="utf-8-sig")
    print("\n==== 模型评估 ====")
    print(metrics)

    # 特征重要性（最优模型 RandomForest）
    imp = pd.Series(rf.feature_importances_, index=X.columns)
    imp = imp.sort_values(ascending=False).head(15)
    imp.to_csv(PROC_DIR / "rf_importance.csv", encoding="utf-8-sig",
               header=["importance"])

    plt.figure(figsize=(9, 6))
    plt.barh(imp.index[::-1], imp.values[::-1], color="#756bb1")
    plt.title("图8 · RandomForest 特征重要性（薪资预测）")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig08_rf_importance.png", dpi=150,
                bbox_inches="tight")
    plt.close()
    print("[fig] fig08_rf_importance.png")

    # 真实 vs 预测（XGBoost）
    plt.figure(figsize=(8, 6))
    plt.scatter(y_te, yte_xgb, alpha=0.7, color="#3182bd")
    plt.plot([y_te.min(), y_te.max()], [y_te.min(), y_te.max()],
             "k--", lw=1)
    plt.xlabel("真实中位薪资（万元/年）")
    plt.ylabel("XGBoost 预测薪资")
    plt.title("图9 · XGBoost 预测 vs 真实")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig09_pred.png", dpi=150,
                bbox_inches="tight")
    plt.close()
    print("[fig] fig09_pred.png")
    return {"metrics": metrics, "top_imp": imp}


import re  # noqa: E402


def insights_md(df: pd.DataFrame, importance: pd.Series) -> None:
    out = ["# AI 招聘市场关键洞见\n"]

    direction = (df.groupby("direction")["salary_mid_w"].median()
                 .sort_values(ascending=False))
    out.append("## 不同方向的薪资梯度（中位万元/年）\n")
    out.append(direction.round(1).to_string())
    out.append("\n")

    edu = (df.groupby("edu")["salary_mid_w"].median()
           .reindex(["不限", "大专", "本科", "硕士", "博士"]))
    out.append("\n## 不同学历的薪资中位\n")
    out.append(edu.round(1).to_string())
    out.append("\n")

    skills = (df["skills"].fillna("")
                .str.split(",").explode()
                .str.strip().value_counts().head(15))
    out.append("\n## Top15 技能\n")
    out.append(skills.to_string())
    out.append("\n")

    out.append("\n## XGBoost 模型最重要的 15 个特征\n")
    out.append(importance.to_string())
    out.append("\n")

    (DATA_DIR / "insights.md").write_text("\n".join(out), encoding="utf-8")
    print("[save] insights.md")


def main() -> None:
    df = pd.read_csv(DATA_DIR / "ai_jobs.csv", encoding="utf-8-sig")
    print(f"[data] rows={len(df)}")
    print(df["edu"].value_counts(), "\n", df["exp"].value_counts())

    df = feature_engineering(df)
    df.to_csv(PROC_DIR / "jobs_clean.csv", index=False,
              encoding="utf-8-sig")
    print(f"[clean] shape={df.shape}")

    plot_salary_distribution(df)
    plot_city_distribution(df)
    plot_direction_salary(df)
    plot_edu_salary(df)
    plot_exp_salary(df)
    plot_skills_wordcloud(df)
    plot_skill_frequency(df)

    res = model_salary(df)
    insights_md(df, res["top_imp"])

    print("[done]")


if __name__ == "__main__":
    main()
