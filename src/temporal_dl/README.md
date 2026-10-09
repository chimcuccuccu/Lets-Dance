# `src/temporal_dl/` — Person 2

Học **nhịp / timing lệch** so với reference. Đây là nhánh Deep Learning của Person 2.

Input: chuỗi ghép geometry performer, geometry reference, và `dtw_distance` theo thời gian.

## File

| File | Vai trò | Tuần |
|---|---|---|
| `dataset.py` | Loader sequence + `build_temporal_sequence()` | **T4** — split theo person bổ sung ở T5 |
| `lstm.py` | BiLSTM | **T4** — API `temporal_model()` bổ sung ở T5 |
| `train.py` | Vòng train (`pretrain` / `train`) | **T4** — CLI cho ablation bổ sung ở T5 |
| `infer.py` | Load checkpoint → embedding | **T5** |

Geometry feature (T2) và DTW (T3) nằm ở `src/features/` — xem `geometry.py`, `dtw.py`, `run_dtw_all.py`.

## Deliverable theo tuần

| Tuần | Phải giao | Artifact |
|---|---|---|
| **Tuần 4** | Dataset loader cho sequence `(performer_geo, reference_geo, dtw_distance)`; BiLSTM 1 lớp đầu tiên, load warm-start nếu có; *model train chạy được, loss giảm* | `dataset.py`, `lstm.py`, `train.py`, `experiments/temporal_dl/train_history.csv`, `temporal_best.pth` |
| **Tuần 5** | So 3 kịch bản (a)/(b)/(c) qua một XGBoost tạm (MAE/RMSE); Pearson/Spearman với `khop_nhip`; API `temporal_model(seq) -> embedding, dtw_features` | `infer.py`, `scripts/{eval_temporal_xgb,corr_temporal_khop_nhip,plot_temporal_eval}.py`, `experiments/temporal_dl/{ablation_*,correlation_*,week5_results.txt,plots/}` |

Checkpoint pretrain AIST++ (`checkpoints/temporal_pretrained.pt`) là bước **Tuần 3–4 không bắt buộc**, dùng làm điểm khởi đầu cho cả hai tuần — xem `docs/temporal_pretrain.md`.

---

## Tuần 4 — loader + BiLSTM đầu tiên

Deliverable là *"model train chạy được, loss giảm"*: `experiments/temporal_dl/train_history.csv` và `pretrain_history.csv`. **Hai file này không được ghi đè** — luôn truyền `--run-name` khi train lại.

```powershell
# 0) Cần có trước: annotations/dtw_features.csv (Tuần 3)
python -m src.features.run_dtw_all

# 1) Pretrain AIST++ — Tuần 3-4, tuỳ chọn, bỏ qua được
python -m src.temporal_dl.train pretrain --epochs 20

# 2) Train BiLSTM chính (tự load temporal_pretrained.pt nếu có)
python -m src.temporal_dl.train train --split official --target-col khop_nhip `
  --run-name khopnhip_dtw_official --epochs 60 --patience 12
```

Checkpoint: `checkpoints/temporal_pretrained.pt` (lệnh 1) và `experiments/temporal_dl/temporal_khopnhip_dtw_official.pth` (lệnh 2).

> ⚠️ **Luôn truyền `--run-name`.** Bỏ trống thì lệnh ghi ra `temporal_best.pth` + `train_history.csv` — là artifact Tuần 4 đã commit.

### Hai quy tắc bắt buộc về dữ liệu

Đây là ràng buộc lên loader ở file này. Cả hai được **chặn bằng code** chứ không dựa vào quy ước, vì bảng đo của Tuần 5 chỉ có nghĩa khi cả hai được giữ.

Giải thích đầy đủ ở [`docs/temporal_eval.md`](../../docs/temporal_eval.md) §1.

1. **Chia train/val ở cấp video, nhóm theo `person_id`.** Dataset phát một sample cho mỗi cửa sổ, nên chia ở cấp sample sẽ để cửa sổ của cùng một video nằm cả hai phía và `val_mae` chỉ đo khả năng nhớ. Thêm nữa 20 người nhảy 217 video, không nhóm theo người thì model học "người này thường được chấm bao nhiêu".
2. **Target không được nằm trong input.** `dtw_distance_total` là kênh thứ 17 của input, nên không được dùng làm target. `train.py` chặn bằng `SystemExit`, trừ khi truyền `--allow-dtw-target`.

Số để trích dẫn là `val_mae_video_raw` (gộp theo video, thang điểm gốc) — `val_mae` đã normalize thì không so được với baseline.

Hai hàm thực thi việc chia (`split_video_indices()`, `get_temporal_loaders_grouped()`) được **bổ sung ở Tuần 5**, khi cần một con số generalization để so 3 kịch bản. `get_temporal_loaders()` cũ giữ nguyên hành vi để reproduce lại đường chạy của Tuần 4.

---

## Tuần 5 — ablation 3 kịch bản + API

### Ablation — 3 kịch bản

| Kịch bản | Ý nghĩa | input_dim |
|---|---|---|
| (a) DTW-only | Chỉ `dtw_distance`, không DL | — |
| (b) LSTM-only | Bỏ DTW, chỉ raw sequence | 16 |
| (c) DTW+LSTM | Kết hợp | 17 |

#### Kết quả (GroupKFold 5 fold — bảng để trích dẫn)

| setup | MAE | ±std |
|---|---|---|
| **(c) DTW+LSTM** | **2.306** | 0.218 |
| (c−) DTW+LSTM, encoder không kênh dtw | 2.517 | 0.219 |
| Baseline dance-mean | 2.530 | 0.333 |
| (b) LSTM-only | 2.618 | 0.114 |
| Baseline global-mean | 3.071 | 0.419 |
| (a) DTW-only | 3.198 | 0.503 |

**Verdict hai vế** (kiểm định t ghép cặp trên 5 fold):

1. **CÓ** — BiLSTM học được thứ `dtw_distance_total` không mã hoá: (a)→(c) ΔMAE = **+0.892**, p=**0.006**. Ngay cả (b) LSTM-only, không thấy kênh DTW, cũng hơn (a) 0.580 điểm (p=0.046).
2. **NHƯNG** — chưa thắng baseline "trung bình theo bài": dance-mean→(c) ΔMAE = +0.224, **p=0.17**. Model chưa deployable; phần lớn phương sai của `khop_nhip` vẫn là độ khó của bài.

Kênh dtw trong input chỉ giúp thêm 0.211 điểm (p=0.074) — phần lớn lợi ích đến từ bản thân BiLSTM.

Correlation với `khop_nhip` trên tập val (n=34): `temporal_pred_score` Spearman **0.8155** — cao hơn cả `tong_diem` (0.785). 29/64 chiều embedding qua BH-FDR.

> ⚠️ **Đừng lấy bảng split official làm headline.** Ở đó `(c)+dance` đạt 1.741 < dance-mean 1.902, nhưng CV cho thấy đó là nhiễu của n=34 — one-hot dance thực ra làm *tệ* đi (p=0.24). Tương tự `(a+)` đạt 1.917 trên official nhưng chỉ 2.891 trên CV, vì feature `T` gián tiếp mã hoá `dance_id`.

Bảng đầy đủ → `experiments/temporal_dl/ablation_xgb_khop_nhip{,_cv,_folds}.csv`, kiểm định → `ablation_paired_tests_khop_nhip.csv`, correlation → `correlation_khop_nhip.csv`.
Phương pháp → **`docs/temporal_eval.md`**. Kết quả + nhận định đầy đủ → **`experiments/temporal_dl/week5_results.txt`**.

#### Đọc bảng cho đúng

`khop_nhip` có std chỉ **4.11** điểm, nên MAE tuyệt đối gần như vô nghĩa nếu không so với 2 baseline tầm thường (split official 183/34):

| Bộ dự đoán | MAE | RMSE |
|---|---|---|
| Hằng số = mean train | 2.850 | 3.787 |
| Mean theo `dance_id` | **1.902** | 2.650 |

`n_val = 34` → SE(MAE) ≈ **±0.45 điểm**: chênh lệch dưới ~1 điểm giữa các kịch bản **không phân định được** trên một split. Dùng bảng CV.

#### Lệnh — bước 1: train 2 encoder

Kịch bản (b) phải có encoder **riêng** chưa bao giờ thấy kênh DTW, nên là 2 lệnh train chứ không phải 1 (lý do ở `docs/temporal_eval.md` §0.6).

```powershell
# (c) encoder 17 kênh   -> temporal_khopnhip_dtw_official.pth
python -m src.temporal_dl.train train --split official --target-col khop_nhip `
  --run-name khopnhip_dtw_official --epochs 60 --patience 12

# (b) encoder 16 kênh   -> temporal_khopnhip_nodtw_official.pth
python -m src.temporal_dl.train train --split official --target-col khop_nhip --no-dtw `
  --run-name khopnhip_nodtw_official --epochs 60 --patience 12
```

#### Lệnh — bước 2: GroupKFold 5 fold

10 lần train, mỗi lần vài phút CPU. Đây là bảng **để trích dẫn**, không phải split official.

```powershell
foreach ($k in 0..4) {
  python -m src.temporal_dl.train train --split groupkfold --fold $k --n-folds 5 `
    --target-col khop_nhip --run-name khopnhip_dtw_fold$k --epochs 60 --patience 12
  python -m src.temporal_dl.train train --split groupkfold --fold $k --n-folds 5 --no-dtw `
    --target-col khop_nhip --run-name khopnhip_nodtw_fold$k --epochs 60 --patience 12
}
```

→ `temporal_khopnhip_{dtw,nodtw}_fold{0..4}.pth` (10 file).

#### Lệnh — bước 3: eval, correlation, biểu đồ

Mọi lệnh dưới đây **tiêu thụ** `.pth`; trong ngoặc là lệnh đã sinh ra file đó.

```powershell
# 1) Bảng ablation, split official   (.pth ← bước 1)
python scripts/eval_temporal_xgb.py --split official --target khop_nhip --with-dance-onehot `
  --ckpt-dtw   experiments/temporal_dl/temporal_khopnhip_dtw_official.pth `
  --ckpt-nodtw experiments/temporal_dl/temporal_khopnhip_nodtw_official.pth

# 2) Bảng CV 5 fold + kiểm định t ghép cặp   (.pth ← bước 2, script tự quét --ckpt-dir)
python scripts/eval_temporal_xgb.py --cv-folds 5 --split groupkfold `
  --target khop_nhip --with-dance-onehot --ckpt-dir experiments/temporal_dl

# 3) Pearson / Spearman với khop_nhip   (.pth ← bước 1)
python scripts/corr_temporal_khop_nhip.py --target khop_nhip --fdr `
  --ckpt       experiments/temporal_dl/temporal_khopnhip_dtw_official.pth `
  --ckpt-nodtw experiments/temporal_dl/temporal_khopnhip_nodtw_official.pth

# 4) Biểu đồ — chỉ đọc CSV, KHÔNG cần .pth
python scripts/plot_temporal_eval.py

# 5) Control S5: xáo nhãn, mọi MAE phải tụt về ≈ baseline 2.85   (.pth ← bước 1)
python scripts/eval_temporal_xgb.py --split official --target khop_nhip --shuffle-labels `
  --ckpt-dtw   experiments/temporal_dl/temporal_khopnhip_dtw_official.pth `
  --ckpt-nodtw experiments/temporal_dl/temporal_khopnhip_nodtw_official.pth
```

Output (đều trong `experiments/temporal_dl/`): `ablation_xgb_khop_nhip{,_cv,_folds,_shuffled}.csv`, `ablation_paired_tests_khop_nhip.csv`, `ablation_preds_khop_nhip{,_shuffled}.csv`, `correlation_khop_nhip.csv`, `temporal_embeddings_{meta,2d}.csv`, `embeddings_sanity_khop_nhip{,_shuffled}.txt`, `plots/0{1..5}_*.png`.

`*.pth` bị `.gitignore` loại nên checkpoint không vào git — sinh lại bằng bước 1 + bước 2. Deliverable commit là các CSV + `week5_results.txt` + `plots/*.png`; `ablation_preds_khop_nhip.csv` cho phép kiểm tra lại mọi MAE mà không cần file weight.

### Deliverable: API

Hai hàm nằm ở hai file khác nhau, đều được export ra ở [`__init__.py`](__init__.py) nên import bằng một dòng:

| Hàm | Định nghĩa tại |
|---|---|
| `temporal_model(seq, ...) -> (embedding, dtw_features)` | [`lstm.py`](lstm.py) — `# Public inference API` |
| `load_temporal_checkpoint(path) -> (model, ckpt)` | [`infer.py`](infer.py) |

`infer.py` còn có `encode_video`, `aggregate_windows`, `predict_video_score`, `export_temporal_embeddings` — dùng khi cần xử lý cả bộ thay vì một video.

```python
from src.temporal_dl import load_temporal_checkpoint, temporal_model

model, ckpt = load_temporal_checkpoint(
    # .pth ← lệnh train (c) ở bước 1 bên trên
    "experiments/temporal_dl/temporal_khopnhip_dtw_official.pth")
embedding, dtw_features = temporal_model(seq, model=model, dtw_row=row)

# embedding             : (64,) = mean ⊕ std trên embedding từng cửa sổ
# dtw_features["vector"]: (4,) [total, seg_mean, seg_std, seg_max]
#   ← cố ý trùng thứ tự _dtw_row_features của Person 1 để fusion Tuần 6
#     concat được mà không cần adapter
```

### Caveat (chi tiết ở `docs/temporal_eval.md` §7)

- **`khop_nhip` không phải điểm chuyên gia** — trùng byte-for-byte `score_rhythm` trong `annotations/demo_source/labels_auto.csv` (`notes` = `auto`), không có script sinh nào trong repo.
- **`D01_P003_T01` chính là video reference** (pose giống hệt `dance_001_ref.npy`), được chấm 300/300. Một điểm này chiếm 1/4 giá trị Pearson: r = −0.4419 với nó, −0.3349 khi bỏ. Nó nằm trong split train nên không ảnh hưởng metric val.
- **`flag_ratio` là đại lượng *trong* một video**, không phải thang giữa các video: ngưỡng `mean + 1.5·σ` tính trên chính các đoạn của video đó, nên tỉ lệ gắn cờ gần như hằng số (~7.3%) theo thiết kế. Tương quan thấp với `khop_nhip` (r = −0.078) là **đúng như dự đoán, không phải lỗi**. Phép đánh giá đúng là precision/recall so với `error_labels.csv` — chưa làm được vì file đó mới có header.
