# Gold Price Prediction（黄金价格预测）

## 项目简介

该项目基于时间序列预测黄金价格，包含传统机器学习（Linear、Ridge、SVR）和深度学习（LSTM、GRU）模型。通过实验结果对比与可视化，找到最优预测策略。已集成 Streamlit 仪表盘供实时展示。

## 目录结构

- `app_live.py`: Streamlit 可视化仪表盘。
- `data/raw/gold_data_enriched.csv`: 原始扩展数据（含特征工程后）。
- `data/processed/`: 训练/验证/测试分割数据。
- `notebooks/`:
  - `fill_seq_baselines.py`: 线性/岭回归/SVR 基线实验。
  - `seq_len_sweep_c11.py`: LSTM/GRU 不同序列长度扫参。
  - `export_loss_history.py`: 导出训练损失历史。
- `outputs/`
  - `reports/seq_len_sweep_c11_results.csv`: 模型指标结果。
  - `figures/`: 实验图表。
  - `dashboard_*.csv`: 仪表盘数据快照。
- `requirements.txt`: Python 依赖列表。

## 环境与依赖

推荐使用 Python 3.10+。

安装依赖：

```bash
pip install -r requirements.txt
```

文件 `requirements.txt`（当前）：

- streamlit==1.55.0
- pandas==2.3.3
- numpy==2.4.1
- plotly==6.6.0

> 若执行完整训练代码还需：
> - torch
> - scikit-learn


## 数据与复现

1. 确保 `outputs/reports/seq_len_sweep_c11_results.csv` 或 `seq_len_sweep_c11_results_partial.csv` 存在，`app_live.py` 自动读取。
2. 若缺少，可先执行：
   - `notebooks/fill_seq_baselines.py`
   - `notebooks/seq_len_sweep_c11.py`
   - `notebooks/export_loss_history.py`
3. 结果默认输出在 `outputs/` 目录，路径可在脚本中 `ROOT` 变量调整。

## 功能概述

`app_live.py` 提供：

- 全局结果读取与模型选项（模型、序列长度）
- 指标对比：MAE、RMSE、MAPE、R²、gap、训练时间
- 可视化：实际值 vs 预测、损失曲线、特征相关性、seq_len 曲线

## 注意

- 该 `README` 已实际写入项目根目录：`README.md`。
- 若希望增加一键运行脚本（`run_all.sh` / `run_all.ps1`），可继续补充。
