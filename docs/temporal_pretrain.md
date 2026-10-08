# Pretrain Temporal BiLSTM trên AIST++ (Person 2)

Temporal BiLSTM cần trích xuất các đặc trưng nhịp điệu (rhythm features) từ chuỗi chuyển động.
Vì bộ dữ liệu tự quay khá nhỏ — **241 video, trong đó 217 có nhãn** (chia 183 train / 34 test) — việc pretrain trên AIST++ sẽ giúp model học được các biểu diễn (representations) tốt về chuyển động và nhịp điệu trước khi fine-tune.

## Proxy Task: Phân loại thể loại nhảy (Genre Classification)

Trong pipeline của Person 2, khác với Person 1 (dùng proxy task dự đoán khác biệt frame), chúng ta tận dụng nhãn thể loại nhảy (genre) có sẵn trong AIST++ để làm proxy task.

AIST++ có 10 thể loại nhảy, được mã hoá ngay trong tên file (VD: `gBR_sBM_c01_d04_mBR0_ch01.npy` -> `gBR` là Breaking).
Các thể loại:
- `gBR`: Breaking
- `gPO`: Pop
- `gLO`: Locking
- `gMH`: Middle Hip-hop
- `gLH`: LA style Hip-hop
- `gHO`: House
- `gWA`: Waack
- `gKR`: Krump
- `gJS`: Jazz
- `gJB`: Jive-Ballet

**Kiến trúc Pretrain (AISTPPPretrainModel):**
1. **Input:** Geometry features `(B, T, 8)` của một người nhảy duy nhất (AIST++).
2. **Backbone:** `TemporalBiLSTM` (cùng kiến trúc với model chính).
3. **Head:** Linear classifier `(embed_dim -> 10 genres)`.

## Cách chạy Pretrain

Toàn bộ thao tác đều dùng module `src.temporal_dl.train`.

Mặc định, lệnh này sẽ load các file `*.npy` trong `data/aistpp/shared_basic_cache` (do Person 1 đã tạo).
Nó sẽ chunk các sequence thành các cửa sổ 60 frames (~1s), tính toán geometry features on-the-fly, và train mô hình BiLSTM phân loại.

```bash
python -m src.temporal_dl.train pretrain
```

Cấu hình mặc định:
- `--epochs 20`
- `--batch-size 32`
- `--window 60`

*Có thể xem thêm các cờ bằng lệnh `python -m src.temporal_dl.train pretrain --help`*

## Kết quả Pretrain

Trong quá trình train, history sẽ được ghi vào:
`experiments/temporal_dl/pretrain_history.csv`

Model đạt `val_acc` cao nhất sẽ được tự động lưu thành:
`checkpoints/temporal_pretrained.pt`

**Đánh giá:**
Do số lượng class là 10, random guess là 10%. Model TemporalBiLSTM chỉ với 64 hidden units có thể đạt được độ chính xác val ~55% sau 20 epoch. Điều này chứng tỏ backbone đã có khả năng trích xuất được những khác biệt về nhịp điệu và hình thái giữa các điệu nhảy, tạo đà tốt cho quá trình fine-tune (Main train).

## Load vào Main Training

Sau khi có file checkpoint ở bước Pretrain, quá trình training chính trên dataset tự quay (TemporalRegressionModel) có thể tái sử dụng các trọng số này (warm-start).

Lưu ý: main model có **hai biến thể input dimension**, đều khác pretrain = 8:

| Biến thể | input_dim | Ghép | Dùng cho |
|---|---|---|---|
| có kênh DTW | **17** | performer + reference + dtw | kịch bản (c) DTW+LSTM |
| bỏ kênh DTW | **16** | performer + reference | kịch bản (b) LSTM-only |

`load_pretrained_backbone` lọc theo shape nên warm-start được **cả hai** mà không cần sửa gì: nó **chỉ tái tạo recurrent weights (h→h)** và FC layers, để random cho weights chiếu input (`w_ih`). Dòng log `Bỏ qua 2 keys do shape mismatch: ['lstm.weight_ih_l0', 'lstm.weight_ih_l0_reverse']` là đúng, không phải lỗi.

Để tự động dùng checkpoint đã train:
```bash
python -m src.temporal_dl.train train --run-name khopnhip_dtw_official
```
*(Hệ thống sẽ tự quét tìm `checkpoints/temporal_pretrained.pt`)*

Từ Tuần 5, lệnh `train` mặc định là `--target-col khop_nhip --split official`. Đường legacy của Tuần 4 (target `dtw_distance_total`, `random_split` cấp cửa sổ) cần nói rõ ý định — dùng cờ `--no-scores` chứ không phải `--scores-csv ""`, vì PowerShell nuốt chuỗi rỗng:

```bash
python -m src.temporal_dl.train train --no-scores \
  --target-col dtw_distance_total --allow-dtw-target --split random \
  --epochs 30 --run-name week4_legacy_repro
```

## Ghi chú: cùng `window_frames` nhưng khác khoảng thời gian

Cả pretrain lẫn main train đều dùng `--window 60`, nhưng **60 frame ở hai bộ dữ liệu không đại diện cùng một khoảng thời gian**:

- Video tự quay: ~30 fps (đã kiểm tra: 30.00 / 30.11, một file 25.02) → 60 frame ≈ **2 giây**.
- Cache AIST++ (`data/aistpp/shared_basic_cache/`): mọi file đều đúng **360 frame** vì bước convert của Person 1 đã chuẩn hoá độ dài, nên một frame ở đây **không ứng với một khoảng thời gian cố định**.

Hệ quả: backbone được pretrain trên các cửa sổ có nhịp độ thời gian không khớp với lúc fine-tune. Điều này không làm hỏng warm-start (weights `w_ih` vốn đã bị bỏ qua), nhưng là một hạn chế cần nêu trong báo cáo khi bàn về lợi ích của pretrain.
