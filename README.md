# Big Data Analytics Techniques and Applications — Assignment 1

這份作業要用 SVD-based PiSSA 微調 `meta-llama/Llama-3.2-1B`，並以 GSM8K 和 MATH 的 accuracy 評估數學問答表現。作業是個人作業；最後要交一份英文報告，以及公開的 GitHub 程式碼 repository。

## 你要完成什麼

報告最多 3 頁，A4，從 Word 匯出 PDF。依照 PDF 作業說明，逐題回答以下問題，提供結果、圖表和你自己的解釋：

### Question 1：不同 rank 的準確率（1%）

1. 評估未微調的 Llama-3.2-1B，在 GSM8K 和 MATH 上的 accuracy。
2. 用 rank 16、64、256 分別微調，再報告兩個資料集的 accuracy。
3. 比較 rank 改變時兩個資料集的表現趨勢，並解釋可能原因。

`finetune_q1-q3.ipynb` 目前儲存的是 Q3 設定。首次重現 Q1–Q2 時，請使用下方三個 `component: 'default'` 的 run，並設定 `RUN_FRESH_BASELINE = True`，以執行 baseline 和三個 rank。

### Question 2：訓練時間（1%）

沿用 Question 1 的 rank 16、64、256，記錄每個 run 的訓練時間（分鐘），比較並解釋趨勢。Notebook 會計算訓練迴圈時間並寫入 `benchmark_results.csv` 的 `train_minutes` 欄；計時不含 SVD 初始化、合併模型和評估。請在報告中註明這個計時範圍。

### Question 3：奇異值區間（1%）

固定 rank 16，分別取下列四個區間做微調：

| component | 選取的奇異值 |
| --- | --- |
| `default` | 最大的 16 個（起點 0） |
| `25pct` | 從奇異值序列約 25% 處開始，連續 16 個 |
| `50pct` | 從約 50% 處開始，連續 16 個 |
| `bottom` | 最小的 16 個 |

對 GSM8K 和 MATH 各自繪製四種設定的 accuracy 長條圖，並解釋兩個資料集的趨勢。Notebook 已支援四種 component；在 `MANUAL_RUNS` 設定 rank 16 的四個 component 後執行。兩張圖會存到 `q3_figures/`。

### Question 4：q_proj 權重的 SVD（2%）

這題只分析原始模型，不需要微調或評估。分析 Transformer layer 0、7、15 的 `q_proj` 權重：

1. 對每個 layer 做 SVD，計算累積能量百分比：`cumsum(S**2) / sum(S**2) * 100`。
2. 找出累積能量第一次達到至少 50% 的最小 rank，列在表格中。
3. 每個 layer 各畫兩張圖：前 200 個 singular values 的 magnitude 長條圖，以及 cumulative energy 折線圖，共六張圖。
4. 說明三個 layer 的奇異值分布與達到 50% 能量所需 rank 的差異。

`finetune_q4.ipynb` 會對三個 layer 完成分析，輸出最小 rank 表格和六張圖。

## 開始前：Kaggle 與 Hugging Face

範例程式原本以 Kaggle 為執行環境；也可在有 CUDA GPU 的本機執行。Kaggle 設定如下：

1. 準備可使用 GPU 的 Kaggle 帳號；課程投影片建議選 GPU T4 x2。Notebook 會在有兩張 GPU 時使用 DataParallel，也可在單張 CUDA GPU 執行。
2. 到 Hugging Face 申請 `meta-llama/Llama-3.2-1B` 的 gated model 存取權，建立 read token。
3. 在 Kaggle notebook 的 **Add-ons → Secrets** 新增 secret，名稱設為 `HF_TOKEN`，值貼上 Hugging Face token。不要把 token 寫進 notebook 或上傳到 GitHub。
4. 將 `finetune_q1-q3.ipynb`、`finetune_q4.ipynb` 上傳或匯入 Kaggle，依照下方各題設定執行。

Notebook 會下載 `meta-math/MetaMathQA` 的前 25,000 筆訓練資料，以及 `fxmeng/pissa-dataset` 中 GSM8K、MATH 各 1,000 筆測試資料。作業指定設定包含 1 epoch、learning rate `2e-5`、FP16、最多生成 256 tokens；除題目指定的 rank 和 component 外，不要任意更動基本設定。

### 在本機使用 GPU

本機已下載模型時，可設定 `ASSIGNMENT_MODEL_PATH` 指向模型資料夾。以下以 GPU 9 為例；`CUDA_VISIBLE_DEVICES=9` 會讓程式內的 `cuda:0` 對應到實體 GPU 9。先啟用包含 PyTorch、Transformers、Datasets、nbclient 的 Python 環境，再執行：

```bash
export ASSIGNMENT_MODEL_PATH=/home/ccchen/models/Llama-3.2-1B
export ASSIGNMENT_MICRO_BATCH_SIZE=4
export ASSIGNMENT_EVAL_BATCH_SIZE=16
CUDA_VISIBLE_DEVICES=9 python run_notebook.py finetune_q1-q3.ipynb --timeout 18000
CUDA_VISIBLE_DEVICES=9 python run_notebook.py finetune_q4.ipynb
```

`run_notebook.py` 會把執行後的 notebook 存在 `Assignment1_results/`；此目錄已加入 `.gitignore`，不會把模型輸出和預測檔推到公開 repository。Q3 執行前，將 `MANUAL_RUNS` 改成 rank 16 的四個 component，並將 `RUN_FRESH_BASELINE` 設為 `False`。

## 執行 Question 1–3

在 `finetune_q1-q3.ipynb` 設定 cell 編輯 `MANUAL_RUNS`。設定格式如下：

```python
MANUAL_RUNS = [
    {'rank': 16, 'component': 'default'},
    {'rank': 64, 'component': 'default'},
    {'rank': 256, 'component': 'default'},
]
RUN_FRESH_BASELINE = True
```

Question 1–2 使用上面的設定。Question 3 則改成：

```python
MANUAL_RUNS = [
    {'rank': 16, 'component': 'default'},
    {'rank': 16, 'component': '25pct'},
    {'rank': 16, 'component': '50pct'},
    {'rank': 16, 'component': 'bottom'},
]
RUN_FRESH_BASELINE = False
```

先執行上述 Question 1–2 設定。再將 `MANUAL_RUNS` 換成上述四種 component，並把 `RUN_FRESH_BASELINE` 設為 `False`，執行 Question 3。跑完先確認輸出中的每個 run `status` 都是 `ok`。程式會載入既有成功結果並按 run_tag 跳過已完成的設定。若改變超參數或資料後重新實驗，請改用新的結果目錄，避免誤用舊結果。

## 執行 Question 4

在 Kaggle 執行 `finetune_q4.ipynb`，使用相同 `HF_TOKEN` secret。程式會對 layer 0、7、15 各自計算 SVD、最小 50% 能量 rank，並產生六張圖。將 rank 表格、圖和觀察整理進報告。

## 輸出與繳交

Q1–Q3 notebook 的結果預設寫到 `/kaggle/working/Assignment1/`。主要檔案包括：

- `benchmark_results.csv`：每個 run 的 `gsm8k_accuracy`、`math_accuracy`、訓練時間等摘要欄位。
- `metamath_pissa_{component}_r{rank}_summary.json`：個別 run 的摘要。
- `metamath_pissa_{component}_r{rank}_full_model/`：合併後模型檔案（若 `SAVE_FULL_MODEL = True`）。

將需要的結果從 Kaggle 下載到本機，製作英文報告。請在報告中加入公開 GitHub repository 連結，檔名依課程要求命名為 `Asg1_{studentID}.pdf`，並在 E3 上傳 PDF。程式碼須在截止前推送到公開 GitHub repository。PDF 投影片列出的截止時間為 **2026 年 10 月 18 日 23:55**，並註明逾期不收。

## Repository 檔案

- `finetune_q1-q3.ipynb`：Question 1–3 的實驗 notebook。
- `finetune_q4.ipynb`：Question 4 的 SVD 分析 notebook。
- `BigData_Assignment 1.pptx (2).pdf`：作業說明、題目與繳交規定。

本作業範例程式改編自 [PiSSA 官方程式](https://github.com/MuLabPKU/PiSSA)。報告須用英文撰寫，請依實際實驗結果作答並自行解釋；不要直接把未驗證的推測當成結果。

## 本機已完成的實驗結果（2026-10-11）

所有 run 的 `status` 均為 `ok`，結果位於被 `.gitignore` 忽略的 `Assignment1_results/`：

| 設定 | GSM8K accuracy | MATH accuracy | 訓練時間（分鐘） |
| --- | ---: | ---: | ---: |
| 未微調 baseline | 3.5% | 0.1% | — |
| rank 16，default | 15.6% | 3.7% | 24.97 |
| rank 64，default | 20.3% | 4.5% | 23.69 |
| rank 256，default | 27.3% | 5.7% | 29.52 |
| rank 16，25pct | 12.6% | 3.4% | 25.02 |
| rank 16，50pct | 11.8% | 2.4% | 23.00 |
| rank 16，bottom | 11.7% | 2.4% | 22.52 |

Q4 的 layer 0、7、15 達到至少 50% 累積能量的最小 rank 分別是 **62、120、84**。總表、個別 summary、adapter、預測檔及圖表皆已輸出。`Assignment1_results/Asg1_DRAFT.pdf` 是三頁英文報告草稿；繳交前還要填入學號與公開 GitHub URL，並檢查內容符合個人報告要求。

## 報告修訂（2026-10-18）

`Assignment1_results/Asg1_DRAFT.pdf` 已補上 Q1 rank／適應能力分析與 baseline 長條圖、Q2 非單調時間趨勢分析，Q3 圖表按 Top 16 → 25% → 50% → Bottom 16 排列。Q4 的 layer 0 圖組移至第二頁，layer 7、15 位於第三頁，並補充能量集中程度及低秩近似的解釋。三頁報告中的六張 Q4 圖已改用較大的標籤。

沒有重新訓練或評估模型。為重新繪製清晰的 Q4 圖，僅在 CPU 上對原始 FP16 權重做 FP32 SVD，並確認 50% 能量 rank 仍為 62、120、84。重建方式：

```bash
# 已存在 q4_spectra.npz 時可略過第一行。
python export_q4_spectra.py /home/ccchen/models/Llama-3.2-1B/model.safetensors
python make_report.py
# 有實際資料後可直接填入報告：
python make_report.py --student-id YOUR_STUDENT_ID --github-url https://github.com/chester71206/big_data_HW1
```

`make_report.py` 使用已儲存的結果與預測檔，並輸出 `math_prediction_audit.csv`：微調設定的 MATH 未解析數為 452–602／1,000，baseline 為 958／1,000。因預測檔沒有 token 數與停止原因，無法可靠地統計截斷／格式未被辨識各占多少；不把未解析一律視為 parser 的錯誤。

本作業的公開程式碼 repository 為 https://github.com/chester71206/big_data_HW1 。報告繳交前仍須填入學號。此產生器以 Matplotlib 輸出 PDF，並非 Word 匯出；若依照作業指定的 Word 流程繳交，還需將內容與圖表整理至 Word 後匯出並檢查三頁限制。
