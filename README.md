# Gold Price Prediction | 黄金价格预测

基于传统机器学习与循环神经网络（LSTM / GRU）的黄金价格预测项目，包含 Notebook 实验与 Streamlit 交互看板。

## 当前项目状态

- 模型范围：`Linear`、`Ridge`、`SVR`、`LSTM`、`GRU`
- 已接入 `SEQ_LEN` 切换：`20 / 30 / 50`
- Loss 交互区支持特征方案切换：`A-宏观(3维)`、`B-全部外部(10维)`、`C-全部(11维)`
- 主交互页面：工作区根目录的 `app_live.py`

## 目录说明

项目包含两层目录，请注意路径：

- 工作区根目录：`d:/Ma/HK/school/521 A.I/WC/gold_price_prediction/`
  - `app_live.py`（Streamlit 页面入口）
- 项目数据与 Notebook 目录：`d:/Ma/HK/school/521 A.I/WC/gold_price_prediction/gold_price_prediction/`
  - `notebooks/`
  - `data/`
  - `outputs/`
  - `README.md`（本文件）

## 快速开始

### 1) 安装依赖

```bash
pip install -r "d:/Ma/HK/school/521 A.I/WC/gold_price_prediction/gold_price_prediction/requirements.txt"
```

### 2) 运行主 Notebook（可选）

主实验文件：

- `d:/Ma/HK/school/521 A.I/WC/gold_price_prediction/gold_price_prediction/notebooks/gold_price_prediction_full.ipynb`

### 3) 启动 Streamlit 看板

```bash
cd "d:/Ma/HK/school/521 A.I/WC/gold_price_prediction/gold_price_prediction"
python -m streamlit run "app_live.py"
```

## 关键输出文件

- `outputs/reports/seq_len_sweep_c11_results.csv`
  - 指标表（`model, seq_len, mae, rmse, mape, test_r2, train_r2, gap, train_time_sec`）
- `outputs/reports/dashboard_model_predictions.csv`
  - Actual vs Predicted 曲线数据
- `outputs/reports/dashboard_loss_history.csv`
  - Loss 曲线数据（`seq_len, scheme, model, epoch, train_loss, val_loss`）

## 数据补齐脚本（常用）

### 补齐 20/30/50 的基线模型指标（Linear/Ridge/SVR）

```bash
python "d:/Ma/HK/school/521 A.I/WC/gold_price_prediction/gold_price_prediction/notebooks/fill_seq_baselines.py"
```

### 导出 A/B/C + LSTM/GRU 的 Loss 历史（供交互曲线使用）

```bash
python "d:/Ma/HK/school/521 A.I/WC/gold_price_prediction/gold_price_prediction/notebooks/export_loss_history.py"
```

## 看板功能说明（app_live.py）

- Sidebar 固定展示 5 模型选择，不再因缺数据而隐藏
- `SEQ_LEN` 切换后，指标卡、表格、预测曲线、loss 曲线、对比图同步切换
- Loss 区包含“特征方案”下拉；每次显示两张交互图：`LSTM` 与 `GRU`

## 常见问题

- 页面模型下拉不全：先运行 `fill_seq_baselines.py`，再刷新页面
- Loss 图为空：先运行 `export_loss_history.py`，确保 `dashboard_loss_history.csv` 有数据
- 页面显示旧结果：刷新 Streamlit（或重启 `streamlit run`）
