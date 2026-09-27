# `notebooks/` — khảo sát và vẽ, không phải code production

Dùng Jupyter để hiểu data **trước / song song** với `src/`. Code chạy thật (train, API) viết trong `src/`.

Gợi ý notebook:

| Notebook | Người | Tuần | Việc |
|---|---|---|---|
| Khảo sát geometry / delay nhịp | P2 | 2 | Plot góc khớp, so performer vs reference |
| Plot vài chiều `diff_sequence` | P1 | 3 | Diff nhỏ = khớp, diff lớn = lệch |
| Scatter lỗi vs điểm | P3 | 3 | Sanity check trước khi train Error DL |
| PCA/t-SNE embedding | P1 | 5 | Điểm cao / thấp có tách cụm |
| Ablation charts | Chung | 6 | MAE/RMSE/R², held-out dance |
