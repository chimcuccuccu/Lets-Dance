# `poses/` — pose đã extract (Person 1, tuần 2)

Output của `src/pose/extract_pose`. Shape mỗi file: **`T × 33 × 4`**.

## Tổ chức gợi ý

```
poses/
├── dance_001/
│   ├── dance_001_ref.npy
│   └── dance_001_p01_v01.npy
├── dance_002/
└── dance_003/
```

`.npy` không commit Git. Person 2 đọc folder này để tính geometry; Person 1 đọc tiếp để preprocess + diff.

**Deliverable tuần 2:** đủ file cho mọi video đã quay. Thiếu file = P2/P3 không làm tiếp được.
