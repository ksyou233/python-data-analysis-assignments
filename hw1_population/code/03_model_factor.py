"""
作业一 · 任务 4：人口增长率的影响因素建模
================================================================
任务：分析人均 GDP、城镇化率、收入水平、教育医疗资源等因素对
     人口自然增长率的影响；识别关键因素。

模型：
- 多元线性回归（OLS）：解释线性关系
- 随机森林回归：捕捉非线性交互
- XGBoost 回归：进一步提升精度

评估指标：R²、Adjusted R²、RMSE、MAE
特征重要性：用于识别关键驱动因素

运行：python 03_model_factor.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import xgboost as xgb

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"


def load_features() -> tuple[pd.DataFrame, list[str]]:
    """
    构造特征矩阵：
    - 人均GDP、城镇化率、收入水平、年龄结构（含老龄化65+）、出生率、死亡率等。
    - 教育医疗资源：在本题数据集中没有直接教育/医院数，使用城镇化率和
      人均收入作为代理变量，并通过加入与"教育/公共支出"强相关的城镇化率
      二次项刻画非线性效应。
    """
    df = pd.read_csv(DATA_DIR / "pop_basic_clean.csv", encoding="utf-8-sig")

    # 派生特征（构造"教育医疗资源代理强度"：城镇化率 × 人均收入）
    df["edu_health_proxy"] = df["urban_rate"] * df["per_capita_income"] / 1e4
    df["urban_rate_sq"] = df["urban_rate"] ** 2

    features = [
        "per_capita_gdp",
        "per_capita_income",
        "urban_rate",
        "urban_rate_sq",
        "age_0_14",
        "age_65_plus",
        "edu_health_proxy",
    ]
    target = "natural_growth_rate"
    return df, features + [target]


def evaluate(y_true, y_pred) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "R2": round(r2_score(y_true, y_pred), 4),
        "AdjR2": "-",  # 见下方计算
        "RMSE": round(np.sqrt(mse), 4),
        "MAE": round(mean_absolute_error(y_true, y_pred), 4),
    }


def adjusted_r2(y_true, y_pred, n, k):
    r2 = r2_score(y_true, y_pred)
    return 1 - (1 - r2) * (n - 1) / (n - k - 1)


def main() -> None:
    df, cols = load_features()
    feature_cols = cols[:-1]
    target_col = cols[-1]

    X = df[feature_cols].values
    y = df[target_col].values
    n, k = X.shape

    # 70% 训练，30% 测试（样本量小，分层按城市分组；这里随机种子固定）
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=42)

    # 1. 线性回归
    lr = LinearRegression().fit(X_tr, y_tr)
    y_pred_lr = lr.predict(X_te)
    m_lr = evaluate(y_te, y_pred_lr)
    m_lr["AdjR2"] = round(adjusted_r2(y_te, y_pred_lr, len(y_te), k), 4)

    # 2. 随机森林
    rf = RandomForestRegressor(
        n_estimators=300, max_depth=6, min_samples_leaf=3,
        random_state=42, n_jobs=-1,
    ).fit(X_tr, y_tr)
    y_pred_rf = rf.predict(X_te)
    m_rf = evaluate(y_te, y_pred_rf)
    m_rf["AdjR2"] = round(adjusted_r2(y_te, y_pred_rf, len(y_te), k), 4)

    # 3. XGBoost
    xgbr = xgb.XGBRegressor(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, verbosity=0,
    ).fit(X_tr, y_tr)
    y_pred_xgb = xgbr.predict(X_te)
    m_xgb = evaluate(y_te, y_pred_xgb)
    m_xgb["AdjR2"] = round(adjusted_r2(y_te, y_pred_xgb, len(y_te), k), 4)

    res = pd.DataFrame(
        {"LinearRegression": m_lr, "RandomForest": m_rf, "XGBoost": m_xgb}
    ).T
    print("====== 模型评估 ======")
    print(res)

    # 特征重要性（XGBoost）
    imp = pd.Series(xgbr.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\n====== XGBoost 特征重要性 ======")
    print(imp.round(4))

    # 线性回归系数（解释方向）
    coef = pd.Series(lr.coef_, index=feature_cols).sort_values()
    print("\n====== 线性回归系数（标准化前） ======")
    print(coef.round(4))

    # 绘图：特征重要性
    plt.figure(figsize=(9, 5))
    colors = ["#1f77b4" if v >= 0 else "#d62728" for v in
              xgbr.feature_importances_[np.argsort(xgbr.feature_importances_)]]
    plt.barh(imp.index, imp.values, color="#2ca02c")
    plt.xlabel("Importance")
    plt.title("XGBoost 特征重要性（自然增长率预测）")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig09_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[fig] fig09_feature_importance.png")

    # 预测散点对比
    plt.figure(figsize=(9, 5))
    plt.scatter(y_te, y_pred_xgb, alpha=0.7, label="XGBoost")
    plt.scatter(y_te, y_pred_rf, alpha=0.5, label="RandomForest")
    plt.plot([y.min(), y.max()], [y.min(), y.max()], "k--", lw=1)
    plt.xlabel("真实自然增长率（‰）")
    plt.ylabel("预测值（‰）")
    plt.title("模型预测 vs 真实值（测试集）")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig10_pred_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[fig] fig10_pred_scatter.png")

    # ── 模型效果对比柱状图 ──
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    metrics_display = ["R2", "RMSE", "MAE"]
    colors_bar = ["#2e86c1", "#28b463", "#e74c3c"]
    for ax, metric, color in zip(axes, metrics_display, colors_bar):
        vals = [m_lr[metric], m_rf[metric], m_xgb[metric]]
        bars = ax.bar(["Linear\nRegression", "Random\nForest", "XGBoost"],
                      vals, color=color, alpha=0.8, edgecolor="k", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=9)
        ax.set_title(metric, fontsize=12)
        ax.set_ylabel(metric)
        ax.set_ylim(0, max(vals) * 1.3 if metric != "R2" else 1.0)
        ax.grid(axis="y", alpha=0.3)
    plt.suptitle("三模型效果对比（测试集）", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig13_model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[fig] fig13_model_comparison.png")

    # 保存评估结果
    res.to_csv(DATA_DIR / "model_metrics.csv", encoding="utf-8-sig")
    imp.to_csv(DATA_DIR / "feature_importance.csv", encoding="utf-8-sig",
               header=["importance"])
    print(f"[save] {DATA_DIR / 'model_metrics.csv'}, feature_importance.csv")


if __name__ == "__main__":
    main()
