# `src/temporal_dl/` — Person 2, tuần 4–5

Học **nhịp / timing lệch** so với reference. Đây là nhánh Deep Learning của Person 2.

Input: chuỗi ghép geometry performer, geometry reference, và `dtw_distance` theo thời gian.

## File

| File | Vai trò |
|---|---|
| `dataset.py` | Loader sequence + `build_temporal_sequence()` + split theo person |
| `lstm.py` | BiLSTM + API `temporal_model()` |
| `train.py` | Vòng train (`pretrain` / `train`) |
| `infer.py` | Load checkpoint → embedding (Tuần 5) |

## Tuần 4 — xong

Dataset `(performer_geometry_seq, reference_geometry_seq, dtw_distance_seq)`, model 1-layer BiLSTM, loss giảm (`experiments/temporal_dl/train_history.csv`).

> **Erratum (phát hiện ở Tuần 5).** Deliverable Tuần 4 là *"model train chạy được, loss giảm"* — điều đó đúng và `train_history.csv` chứng minh. Nhưng con số `val_mae=0.0617` **không phải số generalization**, vì (i) split là `random_split` cấp **cửa sổ** nên cửa sổ của cùng một video nằm cả train lẫn val, và (ii) target là `dtw_distance_total` đã normalize, mà giá trị này cũng nằm trong kênh 17 của input → model tự tham chiếu. Tuần 5 cung cấp số generalization hợp lệ đầu tiên. File Tuần 4 giữ nguyên không sửa; run Tuần 5 ghi ra tên `temporal_khopnhip_*` riêng.

## Tuần 5 — 3 kịch bản bắt buộc

| Kịch bản | Ý nghĩa | input_dim |
|---|---|---|
| (a) DTW-only | Chỉ `dtw_distance`, không DL | — |
| (b) LSTM-only | Bỏ DTW, chỉ raw sequence | 16 |
| (c) DTW+LSTM | Kết hợp | 17 |

### Kết quả (GroupKFold 5 fold — bảng để trích dẫn)

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
Phương pháp → **`docs/temporal_week5.md`**. Kết quả + nhận định đầy đủ → **`experiments/temporal_dl/week5_results.txt`**.

### Đọc bảng cho đúng

`khop_nhip` có std chỉ **4.11** điểm, nên MAE tuyệt đối gần như vô nghĩa nếu không so với 2 baseline tầm thường (split official 183/34):

| Bộ dự đoán | MAE | RMSE |
|---|---|---|
| Hằng số = mean train | 2.850 | 3.787 |
| Mean theo `dance_id` | **1.902** | 2.650 |

`n_val = 34` → SE(MAE) ≈ **±0.45 điểm**: chênh lệch dưới ~1 điểm giữa các kịch bản **không phân định được** trên một split. Dùng bảng CV.

### Lệnh

```powershell
# encoder (c) 17-ch và (b) 16-ch
python -m src.temporal_dl.train train --split official --target-col khop_nhip `
  --run-name khopnhip_dtw_official --epochs 60 --patience 12
python -m src.temporal_dl.train train --split official --target-col khop_nhip --no-dtw `
  --run-name khopnhip_nodtw_official --epochs 60 --patience 12

# bảng ablation + correlation + biểu đồ
python scripts/eval_temporal_xgb.py --split official --target khop_nhip --with-dance-onehot `
  --ckpt-dtw   experiments/temporal_dl/temporal_khopnhip_dtw_official.pth `
  --ckpt-nodtw experiments/temporal_dl/temporal_khopnhip_nodtw_official.pth
python scripts/corr_temporal_khop_nhip.py --target khop_nhip --fdr `
  --ckpt experiments/temporal_dl/temporal_khopnhip_dtw_official.pth
python scripts/plot_temporal_week5.py
```

`*.pth` bị `.gitignore` loại nên checkpoint không vào git — sinh lại bằng 2 lệnh train ở trên. Deliverable commit là các CSV + `week5_results.txt` + `plots/*.png`; `ablation_preds_khop_nhip.csv` cho phép kiểm tra lại mọi MAE mà không cần file weight.

## Deliverable: API

```python
from src.temporal_dl import load_temporal_checkpoint, temporal_model

model, ckpt = load_temporal_checkpoint(
    "experiments/temporal_dl/temporal_khopnhip_dtw_official.pth")
embedding, dtw_features = temporal_model(seq, model=model, dtw_row=row)

# embedding             : (64,) = mean ⊕ std trên embedding từng cửa sổ
# dtw_features["vector"]: (4,) [total, seg_mean, seg_std, seg_max]
#   ← cố ý trùng thứ tự _dtw_row_features của Person 1 để fusion Tuần 6
#     concat được mà không cần adapter
```

## Caveat (chi tiết ở `docs/temporal_week5.md` §7)

- **`khop_nhip` không phải điểm chuyên gia** — trùng byte-for-byte `score_rhythm` trong `annotations/demo_source/labels_auto.csv` (`notes` = `auto`), không có script sinh nào trong repo.
- **`D01_P003_T01` chính là video reference** (pose giống hệt `dance_001_ref.npy`), được chấm 300/300. Một điểm này chiếm 1/4 giá trị Pearson: r = −0.4419 với nó, −0.3349 khi bỏ. Nó nằm trong split train nên không ảnh hưởng metric val.
- **`flag_ratio` là đại lượng *trong* một video**, không phải thang giữa các video: ngưỡng `mean + 1.5·σ` tính trên chính các đoạn của video đó, nên tỉ lệ gắn cờ gần như hằng số (~7.3%) theo thiết kế. Tương quan thấp với `khop_nhip` (r = −0.078) là **đúng như dự đoán, không phải lỗi**. Phép đánh giá đúng là precision/recall so với `error_labels.csv` — chưa làm được vì file đó mới có header.
