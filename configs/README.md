# `configs/` — cấu hình chạy thí nghiệm

Để hyperparameter, đường dẫn data, seed, `dance_id` held-out **ra file** (YAML/JSON), không hard-code trong `train.py`.

Gợi ý sau này:

- `configs/data.yaml` — root `data/`, `poses/`, danh sách `dance_id`
- `configs/spatial.yaml` / `temporal.yaml` / `error.yaml` — lr, batch, hidden size
- `configs/fusion.yaml` — ablation flags, XGBoost params

Cả 3 người dùng chung seed và quy tắc split `person_id` để số liệu ablation so được với nhau.
