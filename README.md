# Python与智能数据分析作业仓库

本仓库以 `assignments/` 作为根目录，汇总四个作业的代码、数据、图表与报告。

## 目录结构

- `hw1_population/`：中国主要城市人口变化与影响因素分析
- `hw2_bilibili/`：B 站全站排行榜视频与热门评论分析
- `hw3_huggingface/`：Hugging Face 三方向 Top100 模型分析
- `hw4_recruit/`：AI 岗位招聘数据分析与薪资预测

每个作业目录通常包含：

- `code/`：采集、清洗、分析脚本
- `data/`：原始数据、清洗结果和中间产物
- `figures/`：分析图表
- `report.md`：作业报告

## 运行说明

1. 进入对应作业目录。
2. 使用已配置的 Python 环境运行 `code/` 下的脚本。
3. 图表和中间结果会输出到 `data/` 或 `figures/`。

示例：

```powershell
cd "d:\Learning\Python与智能数据分析\assignments\hw4_recruit\code"
D:/anaconda/python.exe 02_analyze_jobs.py
```

## 数据说明

- `hw1_population/`、`hw2_bilibili/`、`hw3_huggingface/`、`hw4_recruit/` 都已经按真实数据链路整理。
- 根级 `.gitignore` 已忽略常见缓存、虚拟环境、编辑器文件和临时文件。
- 如需进一步压缩仓库体积，可按需要取消注释 `.gitignore` 中的大文件规则。

## 提交与推送

当前仓库已在 `assignments/` 下初始化为 git 仓库。若要上传到 GitHub，只需在本地添加远程地址并执行 push。
