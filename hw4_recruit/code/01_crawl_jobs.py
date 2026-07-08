"""
作业四 · 任务 1：加载招聘数据
================================================================
数据源：lukebarousse/data_jobs（Hugging Face 公开数据集）
  从本地已验证的真实 parquet 文件构建 ai_jobs.csv，
  或通过 HF API 自动下载。

实际爬取逻辑见同目录 01b_load_real_data.py，
本脚本作为统一入口直接代理到真实数据源。
"""

from __future__ import annotations

from pathlib import Path
import sys
import subprocess

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


def main() -> None:
    ai_csv = DATA_DIR / "ai_jobs.csv"

    # 如果已有清理后的 CSV，直接确认已就绪
    if ai_csv.exists():
        import pandas as pd
        df = pd.read_csv(ai_csv, encoding="utf-8-sig")
        print(f"[ok] ai_jobs.csv 已存在，共 {len(df)} 条记录")
        print(f"     来源分布: {df['source'].value_counts().to_dict()}")
        return

    # 否则调用实际数据加载脚本
    script = ROOT / "code" / "01b_load_real_data.py"
    if script.exists():
        print("[info] 未找到 ai_jobs.csv，运行 01b_load_real_data.py 构建…")
        subprocess.run([sys.executable, str(script)], check=True)
    else:
        print("[warn] 无法找到 01b_load_real_data.py，请检查数据集文件")
        print("     预期的数据流: 01b_load_real_data.py → ai_jobs.csv → 02_analyze_jobs.py")


if __name__ == "__main__":
    main()
