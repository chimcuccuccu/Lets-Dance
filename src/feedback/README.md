# `src/feedback/` — Person 3 chủ trì, tuần 7

Prompt LLM (GPT-OSS-20B) để viết nhận xét tự nhiên.

**Input:** `score`, `errors` (kèm timestamp), `similar_cases`, `dance_id`.  
**Cấm:** LLM **không được tự đổi điểm**. Điểm chỉ đến từ XGBoost.

Ghép API cuối: `assess(video, dance_id) -> {score, errors, feedback, similar_cases}`.
