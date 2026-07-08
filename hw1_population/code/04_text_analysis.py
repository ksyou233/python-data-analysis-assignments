"""
作业一 · 任务 5：3-5 个代表性城市人口相关文本分析
================================================================
- 文本来源：5个城市（北京、上海、广州、深圳、成都）的人口普查/统计公报摘录
- 处理方法：
    1) 中文分词（jieba）
    2) 停用词过滤
    3) TF-IDF 关键词提取
    4) 词云展示
    5) 与数据建模结果交叉讨论

运行：python 04_text_analysis.py
"""

from __future__ import annotations

from pathlib import Path

import jieba
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
TEXT_DIR = DATA_DIR / "pop_text"
TEXT_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------
# 5 城市人口相关文本（与公开统计公报口径一致）
# ----------------------------------------
CITY_TEXTS = {
    "北京": (
        "北京市第七次全国人口普查公报显示，全市常住人口为2189.3万人，"
        "与第六次普查相比，增长率明显放缓，外来人口增速回落。"
        "人口老龄化进一步加深，60岁及以上人口占比升至19.6%；"
        "城镇化率持续提升，城镇人口比重已超87%。"
        "受教育程度持续提高，人才红利效应凸显，但住房、教育、医疗资源"
        "配置仍面临人口结构性挑战。"
    ),
    "上海": (
        "上海市统计公报指出，常住人口约2487万人，人口集聚效应明显但增速趋稳。"
        "出生率持续走低，人口自然增长率进入负值区间；老龄化程度位居全国前列，"
        "65岁及以上比重超过18%。"
        "通过完善公共服务配套、推进人才引进、"
        "完善养老医疗体系建设来积极应对结构性变化。"
    ),
    "广州": (
        "广州市统计公报显示，作为超大城市，常住人口稳步增长，"
        "吸纳了大量外来年轻劳动力；城镇化率超过86%。"
        "城市聚焦公共服务扩容、保障性住房和基础教育学位供给，"
        "以缓解外来人口落户和子女教育压力。"
    ),
    "深圳": (
        "深圳市年度统计公报显示，常住人口约1768万，人口结构年轻化突出，"
        "常住人口平均年龄较低。"
        "人才政策持续吸引高新技术和金融领域青年人才；"
        "在教育、医疗和住房保障方面加大投入，"
        "对人口红利转化为发展动能有显著影响。"
    ),
    "成都": (
        "成都市统计公报指出，常住人口超过2120万，连续多年保持增长，"
        "作为西部中心城市人口集聚力强。"
        "城镇化率快速提升至75%以上；"
        "通过户籍制度改革、人才补贴、产业新城建设扩大就业承载力；"
        "但中心城区老龄化也在加速，未来需关注养老服务普惠化。"
    ),
}

# 中文停用词
STOPWORDS = set("""
的 了 和 是 在 有 与 及 对 把 被 由 于 以及 各 共 这 那 也 等 之 则 并
或 一 二 三 四 五 六 七 八 九 十
我们 他们 你 我 他 她 它 您 自己
""".split())


def tokenize(text: str) -> list[str]:
    import re
    words = [w.strip() for w in jieba.cut(text) if w.strip()]
    # 去掉纯数字/百分号/年份/单字符
    keep = []
    for w in words:
        if len(w) < 2:
            continue
        if re.fullmatch(r"\d+(\.\d+)?%?", w):
            continue
        if w in STOPWORDS:
            continue
        keep.append(w)
    return keep


def main() -> None:
    # 1) 保存文本
    for city, text in CITY_TEXTS.items():
        (TEXT_DIR / f"{city}.txt").write_text(text, encoding="utf-8")

    corpus = [" ".join(tokenize(t)) for t in CITY_TEXTS.values()]
    cities = list(CITY_TEXTS.keys())

    # 2) TF-IDF
    vec = TfidfVectorizer(max_df=0.9, min_df=1, token_pattern=r"(?u)\S+")
    tfidf = vec.fit_transform(corpus)
    terms = vec.get_feature_names_out()
    df = pd.DataFrame(tfidf.toarray(), index=cities, columns=terms)
    print("TF-IDF 矩阵前几列：")
    print(df.iloc[:, :8].round(3))

    # 3) 每城市 Top-15 关键词
    top_dict = {}
    for i, city in enumerate(cities):
        s = df.iloc[i].sort_values(ascending=False).head(15)
        top_dict[city] = s
        print(f"\n== {city} Top15 ==")
        print(s.round(3))

    # 4) 词云（合并全部）
    all_text = " ".join(corpus)
    wc = WordCloud(
        font_path="C:/Windows/Fonts/msyh.ttc",
        width=800, height=500,
        background_color="white",
        max_words=150,
    ).generate(all_text)
    plt.figure(figsize=(10, 6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title("5 城人口公报词云")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig11_wordcloud.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[fig] fig11_wordcloud.png")

    # 5) 每城市 Top10 横向条形图（小倍数）
    fig, axes = plt.subplots(3, 2, figsize=(13, 10))
    axes = axes.ravel()
    for idx, city in enumerate(cities):
        s = top_dict[city].sort_values()
        axes[idx].barh(s.index, s.values, color="#3182bd")
        axes[idx].set_title(city)
        axes[idx].set_xlabel("TF-IDF")
        axes[idx].grid(True, axis="x", alpha=0.3)
    axes[-1].axis("off")
    plt.suptitle("5 城市 Top10 关键词（TF-IDF）", y=1.02, fontsize=14)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig12_top_keywords.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[fig] fig12_top_keywords.png")

if __name__ == "__main__":
    main()
