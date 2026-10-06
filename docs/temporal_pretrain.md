# Pretrain Temporal BiLSTM trên AIST++ (Person 2)

Temporal BiLSTM cần trích xuất các đặc trưng nhịp điệu (rhythm features) từ chuỗi chuyển động.
Vì bộ dữ liệu thật khá nhỏ (~1400 video), việc pretrain trên AIST++ sẽ giúp model học được các biểu diễn (representations) tốt về chuyển động và nhịp điệu.

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

## Load vào Main Training (Tuần 4)

Sau khi có file checkpoint ở bước Pretrain, quá trình training chính trên dataset tự quay (TemporalRegressionModel) có thể tái sử dụng các trọng số này (warm-start).

Lưu ý: Main model ghép (performer, reference, dtw) nên input dimension = 17, khác với pretrain = 8. Hệ thống `load_pretrained_backbone` đã được code để **chỉ tái tạo recurrent weights (h->h)** và FC layers, thiết lập random cho weights chiếu input (w_ih).

Để tự động dùng checkpoint đã train:
```bash
python -m src.temporal_dl.train train
```
*(Hệ thống sẽ tự quét tìm `checkpoints/temporal_pretrained.pt`)*
