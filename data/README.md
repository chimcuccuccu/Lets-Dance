# `data/` — dữ liệu thô (video)

Chứa video gốc theo từng bài nhảy. **Không** chứa pose đã extract (pose để ở `poses/`) và **không** chứa nhãn (nhãn để ở `annotations/`).

## Quy tắc

- Mỗi bài một thư mục `dances/dance_XXX/`.
- So sánh performer **chỉ** với reference của **đúng** `dance_id`. Không so nhầm bài.
- File video không commit Git (dung lượng lớn). Folder này tồn tại trên Git nhờ README / `.gitkeep` để cả nhóm thống nhất chỗ đặt file.

## Việc tuần 1 (cả nhóm)

1. Chọn 2–3 bài (30–60s/bài), nhạc royalty-free. Nên có 1 bài dễ/chậm và 1 bài khó/nhanh.
2. Quay **1 video reference** / bài (người nhảy chuẩn nhất).
3. Lên lịch quay 8–12 performer / bài, đa dạng trình độ (mới học / sơ / thành thạo).

Chi tiết đặt file: [`dances/README.md`](dances/README.md).
