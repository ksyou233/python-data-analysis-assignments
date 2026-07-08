# Python Data Analysis Assignments

This repository is the root for the four homework projects under `assignments/`. It collects code, data, figures, and reports for each task in one place.

## Projects

- `hw1_population/` - China major-city population change and factor analysis
- `hw2_bilibili/` - Bilibili ranking videos and hot comment analysis
- `hw3_huggingface/` - Hugging Face top-100 model analysis across three directions
- `hw4_recruit/` - AI job market analysis and salary prediction

## Repository Layout

Each homework folder follows the same structure:

- `code/` - data collection, cleaning, analysis, and modeling scripts
- `data/` - raw data, cleaned tables, and intermediate outputs
- `figures/` - generated charts used in the report
- `report.md` - final report for the assignment

Some projects also include an additional `processed/` subfolder under `data/` for cleaned tables, feature matrices, and model artifacts.

## Data Status

All four homework projects are currently aligned with real data sources and regenerated outputs:

- `hw1_population/` uses official statistical sources and published population bulletins
- `hw2_bilibili/` uses a live Bilibili crawl plus analysis outputs from the refreshed dataset
- `hw3_huggingface/` uses the verified Hugging Face model dataset and regenerated figures
- `hw4_recruit/` uses the verified job-market dataset and regenerated figures

## How To Run

Use the existing base Python environment at `D:/anaconda/python.exe`.

Example:

```powershell
cd "d:\Learning\Python与智能数据分析\assignments\hw4_recruit\code"
D:/anaconda/python.exe 02_analyze_jobs.py
```

Typical workflow:

1. Run the data collection or data-loading script in `code/`.
2. Run the analysis script in `code/`.
3. Review the regenerated charts under `figures/`.
4. Read the corresponding `report.md`.

## Git And Ignore Rules

The repository includes a root-level `.gitignore` that already ignores common Python caches, virtual environments, editor files, and temporary artifacts.

If you want to keep the repository smaller, you can optionally ignore large raw datasets or other derived artifacts in each project folder.

## Commit And Push

The repository has been initialized with `assignments/` as the git root and pushed to GitHub.

Remote:

- `git@github.com:ksyou233/Python-.git`

If you rename the GitHub repository later, update the local remote URL with `git remote set-url origin <new-url>`.

## Notes

- Reports are written in Markdown for easy review inside VS Code.
- Figures are regenerated from the current real datasets rather than older sample outputs.
- Secrets are kept out of the repository history; environment variables should be used for any token-based access.
