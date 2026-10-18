# Big Data Assignment 1

使用 PiSSA 微調 Llama-3.2-1B，評估 GSM8K／MATH，並分析原始模型的奇異值分布。

## 環境準備

需要 NVIDIA GPU 與支援 CUDA 的 PyTorch。本次使用 PyTorch 2.2.0+cu118、Transformers 4.57.6。

```bash
pip install 'numpy<2' 'transformers==4.57.6' 'datasets<4' pandas matplotlib tqdm safetensors nbclient nbformat ipykernel
python -m ipykernel install --user --name python3
hf auth login
```

Hugging Face 帳號須先取得 `meta-llama/Llama-3.2-1B` 存取權。設定已下載的模型路徑與 GPU（以下以 GPU 9 為例）：

```bash
export ASSIGNMENT_MODEL_PATH=/path/to/Llama-3.2-1B
export CUDA_VISIBLE_DEVICES=9
export ASSIGNMENT_MICRO_BATCH_SIZE=4
export ASSIGNMENT_EVAL_BATCH_SIZE=16
```

## 執行實驗

**Q1–Q2**：修改 `finetune_q1-q3.ipynb` 第二個 cell 的設定，再執行：

```python
MANUAL_RUNS = [{'rank': r, 'component': 'default'} for r in (16, 64, 256)]
RUN_FRESH_BASELINE = True
```

```bash
python run_notebook.py finetune_q1-q3.ipynb --timeout 18000
```

**Q3**：同一個 cell 改成以下設定，再執行相同指令（repository 目前儲存的是此設定）：

```python
MANUAL_RUNS = [{'rank': 16, 'component': c} for c in ('default', '25pct', '50pct', 'bottom')]
RUN_FRESH_BASELINE = False
```

**Q4**：分析原始模型 layer 0、7、15 的 `q_proj`：

```bash
python run_notebook.py finetune_q4.ipynb --timeout 18000
```

本機結果存於 `Assignment1_results/`，包含 accuracy、訓練時間、預測、adapter、圖表與執行後的 notebook。程式會跳過已成功完成的 run；若更改超參數，請先將舊結果目錄移走，以免沿用舊結果。
