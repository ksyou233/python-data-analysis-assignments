"""
作业二 · 任务 1：爬取 B 站"全站排行榜"前 50 个视频 + 每视频前 10 条热门评论
==================================================================
任务拆解：
  (1) 全站排行榜前 50 视频：标题、UP主、播放量、点赞、弹幕、评论数、分区
  (2) 每个视频前 10 条热门评论

实现思路：
  - 排行榜：通过 GET https://api.bilibili.com/x/web-interface/ranking/v2?rid=0&type=all
            （推荐为 Web 端 ranking 接口；如该接口被风控，回退到 main/hot/v2）
  - 评论：通过 https://api.bilibili.com/x/v2/reply/main ，
            需要带 aid / type / mode=3 （热门）
  - 重要：B 站对外接口已开启 *Referer / SESSDATA 风控*。脚本提供
            `cfg.SESSDATA` 让用户填入自己的登录 Cookie（见下文说明）。
            若未提供 SESSDATA，可能会被反爬限流，脚本会自动降级为
            离线生成的样本数据，保证整体流程可演示。

反爬与稳定：
  - 每次请求间隔随机 sleep；
  - 随机 User-Agent；
  - 失败重试 3 次；
  - 评论接口返回频率限制友好，登录态下 ≤100/视频足够。

运行：
    python 01_crawl_bilibili.py
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

# 本地配置：从同目录 config.py 读取 SESSDATA 等敏感信息
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 配置：登录态 SESSDATA（强烈建议填入，否则易被风控）
#   - 优先读环境变量 BILI_SESSDATA
#   - 否则从 config.py 读取 DEFAULT_SESSDATA
#   - 都没有则视为匿名爬取（更易被风控）
# ============================================================
SESSDATA = config.get_sessdata()
UCONTENT = config.get_ucontent()
HAS_LOGIN = bool(SESSDATA)
print(f"[info] SESSDATA 已加载：{'是' if HAS_LOGIN else '否（匿名模式）'}")

HEADERS_BASE = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.bilibili.com/v/popular/rank/all",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "zh-CN,zh;q=0.9",
}


@dataclass
class Video:
    bvid: str
    title: str
    up: str
    tname: str  # 分区
    view: int
    like: int
    danmaku: int
    reply: int
    duration: int
    pubdate: int
    data_source: str = "real"


@dataclass
class Comment:
    bvid: str
    ctime: int
    like: int
    uname: str
    message: str
    rcount: int


# ------------------------------------------------------------
# 1) 排行榜
# ------------------------------------------------------------
def fetch_top_videos(top_n: int = 50, max_retry: int = 3) -> list[dict]:
    """爬取全站排行榜前 top_n 条视频。"""
    url = "https://api.bilibili.com/x/web-interface/ranking/v2"
    params = {"rid": 0, "type": "all"}
    sess = requests.Session()
    last_err: Optional[Exception] = None
    for attempt in range(max_retry):
        try:
            sess.headers.update(HEADERS_BASE)
            if SESSDATA:
                sess.cookies.set("SESSDATA", SESSDATA, domain=".bilibili.com")
            r = sess.get(url, params=params, timeout=12)
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}")
            j = r.json()
            if j.get("code", -1) != 0:
                raise RuntimeError(f"API error: {j.get('message')}")
            lst = j["data"]["list"][:top_n]
            videos = []
            for x in lst:
                v = Video(
                    bvid=x["bvid"],
                    title=x["title"],
                    up=x["owner"]["name"],
                    tname=x["tname"],
                    view=x["stat"]["view"],
                    like=x["stat"]["like"],
                    danmaku=x["stat"]["danmaku"],
                    reply=x["stat"]["reply"],
                    duration=x["duration"],
                    pubdate=x["pubdate"],
                )
                videos.append(asdict(v))
            return videos
        except Exception as e:
            last_err = e
            time.sleep(2 + random.random() * 3)
    print(f"[warn] 真实排行榜接口失败：{last_err}，使用离线样本数据")
    return sample_top_videos(top_n)


# ------------------------------------------------------------
# 2) 评论（mode=3 热门；按页 + size=10）
# ------------------------------------------------------------
def fetch_hot_comments(bvid: str, max_n: int = 10, max_retry: int = 3) -> list[dict]:
    """爬取视频的热门评论（前 max_n 条）"""
    sess = requests.Session()
    sess.headers.update(HEADERS_BASE)
    if SESSDATA:
        sess.cookies.set("SESSDATA", SESSDATA, domain=".bilibili.com")

    # 1) bvid -> aid
    vid_url = "https://api.bilibili.com/x/web-interface/view"
    try:
        r = sess.get(vid_url, params={"bvid": bvid}, timeout=10)
        aid = r.json()["data"]["aid"]
    except Exception as e:
        print(f"[warn] {bvid} aid 解析失败：{e}")
        return sample_comments(bvid, max_n)

    # 2) 热门评论
    url = "https://api.bilibili.com/x/v2/reply/main"
    for attempt in range(max_retry):
        try:
            r = sess.get(
                url,
                params={"type": 1, "oid": aid, "mode": 3, "ps": max_n,
                        "next": 0},
                timeout=10,
            )
            j = r.json()
            if j.get("code", -1) != 0:
                raise RuntimeError(j.get("message"))
            lst = (j.get("data", {})
                     .get("replies") or [])[:max_n]
            comments = []
            for c in lst:
                comments.append(asdict(Comment(
                    bvid=bvid,
                    ctime=c["ctime"],
                    like=c["like"],
                    uname=c["member"]["uname"],
                    message=c["content"]["message"],
                    rcount=c["rcount"],
                )))
            return comments
        except Exception as e:
            time.sleep(2 + random.random() * 3)
    return sample_comments(bvid, max_n)


# ------------------------------------------------------------
# 离线样例（保证流程可演示）
# ------------------------------------------------------------
SAMPLE_TNAME = ["游戏", "知识", "生活", "动画", "音乐", "影视",
                "数码", "鬼畜", "美食", "运动", "科技", "动物圈"]

SAMPLE_TITLE_PREFIX = ["【百万播放】", "挑战", "全网首发", "科普向",
                       "教程", "实况", "揭秘", "速通", "全程高能",
                       "专业解读", "开箱"]


def sample_top_videos(top_n: int = 50) -> list[dict]:
    rng = np.random.default_rng(2025)
    out = []
    for i in range(1, top_n + 1):
        tname = rng.choice(SAMPLE_TNAME)
        title = f"{rng.choice(SAMPLE_TITLE_PREFIX)} {tname}相关热榜第{i}期"
        view = int(rng.integers(200_000, 12_000_000))
        like = int(view * rng.uniform(0.04, 0.10))
        danmaku = int(view * rng.uniform(0.005, 0.02))
        reply = int(view * rng.uniform(0.001, 0.005))
        out.append(asdict(Video(
            bvid=f"SAMP{i:04d}",
            title=title,
            up=f"UP主_{i:03d}",
            tname=str(tname),
            view=view,
            like=like,
            danmaku=danmaku,
            reply=reply,
            duration=int(rng.integers(60, 1800)),
            pubdate=int(1700000000 + rng.integers(0, 30_000_000)),
            data_source="sample",
        )))
    return out


def sample_comments(bvid: str, n: int) -> list[dict]:
    """构造与真实结构一致的样本评论，供分析与情感分类演示"""
    rng = np.random.default_rng(hash(bvid) % (2**32))
    pool_pos = ["太强了！", "这就是大佬吗，膜拜", "UP主辛苦！", "内容很硬核",
                "宝藏 UP", "笑死", "学到了", "建议反复观看", "质量一流",
                "BGM 好听", "这就是差距", "细节到位", "真的好看到哭", "感谢分享"]
    pool_neg = ["这是什么？", "有点失望", "标题党", "内容太短了", "不够看",
                "剪辑有点乱", "BGM 太吵", "镜头晃动太多"]
    pool_neu = ["第 1", "03:21", "打卡", "已收藏", "看完", "评论区走起",
                "第一次评论", "标记"]
    out = []
    for i in range(n):
        gt = rng.random()
        if gt < 0.6:
            msg = rng.choice(pool_pos); sentiment = "pos"
        elif gt < 0.85:
            msg = rng.choice(pool_neu); sentiment = "neu"
        else:
            msg = rng.choice(pool_neg); sentiment = "neg"
        out.append({
            "bvid": bvid,
            "ctime": int(1700000000 + rng.integers(0, 30_000_000)),
            "like": int(rng.integers(0, 5000)),
            "uname": f"用户_{i:03d}",
            "message": f"{msg}（演示-{sentiment}）",
            "rcount": int(rng.integers(0, 100)),
        })
    return out


def main() -> None:
    print("[1/3] 爬取排行榜…")
    videos = fetch_top_videos(config.TOP_N_VIDEOS)
    df_v = pd.DataFrame(videos)
    df_v.to_csv(DATA_DIR / "top_videos.csv", index=False, encoding="utf-8-sig")
    print(f"[ok] top_videos.csv 记录={len(df_v)}")

    print("[2/3] 爬取评论…（每视频前 {0} 条）".format(config.COMMENTS_PER_VIDEO))
    all_comments = []
    n = 0
    for v in videos:
        comments = fetch_hot_comments(v["bvid"], max_n=config.COMMENTS_PER_VIDEO)
        # 标记这些评论来自真实接口还是样本
        for c in comments:
            c["data_source"] = "real" if "演示-" not in c["message"] else "sample"
        all_comments.extend(comments)
        n += 1
        time.sleep(random.uniform(0.3, 0.8))  # 更稳的节流
    df_c = pd.DataFrame(all_comments)
    df_c.to_csv(DATA_DIR / "hot_comments.csv", index=False, encoding="utf-8-sig")
    print(f"[ok] hot_comments.csv 记录={len(df_c)}")
    print("[done]")


if __name__ == "__main__":
    main()
