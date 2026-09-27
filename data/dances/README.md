# `data/dances/` — từng bài nhảy

Mỗi `dance_id` là một bài độc lập, dùng chung pipeline (không train model riêng từng bài).

```
dances/
├── dance_001/     # bài 1
├── dance_002/     # bài 2
└── dance_003/     # bài 3 (có thể để trống nếu chỉ chọn 2 bài)
```

## Đặt tên file (thống nhất)

| Vai trò | Đường dẫn gợi ý |
|---|---|
| Reference | `dance_XXX/reference/{dance_id}_ref.mp4` |
| Performer | `dance_XXX/performers/{dance_id}_{person_id}_{video_id}.mp4` |

`person_id` bắt buộc vì **split train/val theo người**, không theo `dance_id` ngẫu nhiên — một người có thể nhảy nhiều bài.

Khi extract pose, Person 1 giữ nguyên `dance_id` + `video_id` để các nhánh còn lại ghép được với nhãn.
