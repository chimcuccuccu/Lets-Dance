# Person 2 — Phương pháp (Temporal DL vs DTW)

File này giải thích **cách đo** và **tại sao đo như vậy**.
Kết quả số nằm ở `experiments/temporal_dl/week5_results.txt`.

Tuần 5 của Person 2 phải giao đúng 3 thứ:

1. So 3 kịch bản qua một XGBoost tạm, đo MAE/RMSE — **(a) DTW-only**, **(b) LSTM-only**, **(c) DTW+LSTM** — để trả lời: *DL có học thêm được gì ngoài thông tin DTW đã có không?*
2. Pearson/Spearman giữa `temporal_embedding` (hoặc dtw score) và cột `khop_nhip`.
3. API `temporal_model(seq) -> embedding, dtw_features`.

---

## 0. Vì sao thiết kế như vậy

Mục này ghi lại lý do đứng sau từng quyết định kỹ thuật của nhánh Temporal, để người review không phải đoán. Mỗi tiểu mục theo cùng một khuôn: **vấn đề → chọn gì → đã loại phương án nào và vì sao**.

### 0.1 Vì sao bài toán cần *cả* DTW lẫn Deep Learning

Đánh giá "khớp nhịp" là so một chuỗi chuyển động với một chuỗi tham chiếu mà hai chuỗi **không thẳng hàng theo thời gian**: người nhảy có thể làm đúng động tác nhưng sớm/muộn nửa nhịp, hoặc giữ đúng nhịp ở đoạn đầu rồi trôi dần.

- **Chỉ DTW** cho một khoảng cách tổng sau khi đã căn chỉnh tối ưu. Nó trả lời *"lệch bao nhiêu"* nhưng không trả lời *"lệch ở đâu, lệch kiểu gì"*, và nén cả bài thành một số.
- **Chỉ DL** thì phải tự học khái niệm "cùng động tác nhưng trễ" từ 183 video — quá ít. Không có mỏ neo thời gian, model dễ học nhầm thành "giống nhau từng frame".

Nên kiến trúc dùng DTW làm mỏ neo căn chỉnh và cấp feature, còn BiLSTM học phần cấu trúc mà một scalar không mã hoá được. Mục đích của Tuần 5 chính là **kiểm chứng xem phần "thêm" đó có thật hay không** — và câu trả lời là có (xem §5).

> **Nguyên tắc chung rút ra:** không đánh giá một công cụ đo *trong một video* bằng một phép tương quan *giữa các video*. Xem §7.3 cho trường hợp cụ thể.

### 0.2 Vì sao geometry 8 góc khớp, không phải toạ độ pose thô

Toạ độ MediaPipe phụ thuộc vị trí người trong khung hình, kích thước người, và khoảng cách tới camera. Góc khớp (gối, khuỷu, vai, hông) **bất biến với cả ba** — hai người cao thấp khác nhau làm đúng cùng động tác sẽ cho cùng dãy góc.

Đã loại: chuẩn hoá toạ độ theo hip + chiều cao. Nó chạy được, nhưng Person 1 đã dùng đúng cách đó cho nhánh Spatial; Person 2 lặp lại thì hai nhánh nhìn dữ liệu y hệt nhau và mất tính bổ sung khi fusion ở Tuần 6.

### 0.3 Vì sao cửa sổ 60 frame (~2 giây)

Phải đủ dài để chứa trọn một câu nhịp, đủ ngắn để còn định vị được lỗi theo thời gian. Ở 30 fps, video dài 219–690 frame cho 3–11 cửa sổ — đủ để tính `mean ⊕ std` mà không quá ít mẫu.

Đã loại: cửa sổ cả bài (mất hoàn toàn định vị thời gian, và không gộp được vì độ dài khác nhau); cửa sổ 1 giây (ngắn hơn chu kỳ của phần lớn động tác, cửa sổ chủ yếu bắt nhiễu).

### 0.4 Vì sao BiLSTM 1 lớp, hidden 64, embed 32

183 video train → 1109 cửa sổ. Model hiện tại đã 47k tham số; mọi thứ to hơn là mời overfit.

Hai chiều (bidirectional) vì lệch nhịp nhận ra được từ **cả hai phía**: một động tác bị trễ thì vừa thấy "đáng lẽ đã xảy ra rồi" khi nhìn tới, vừa thấy "vẫn còn kéo dài" khi nhìn lui.

Đã loại: Transformer (cần dữ liệu lớn hơn hẳn mới hơn được RNN ở quy mô này); LSTM nhiều lớp (overfit ngay với 1109 cửa sổ); GRU (gần tương đương BiLSTM ở quy mô này, đổi sang cũng không làm thay đổi kết luận nên không đáng tốn một nhánh thí nghiệm).

### 0.5 Vì sao pretrain AIST++ bằng phân loại thể loại nhảy

Dataset tự quay quá nhỏ để học biểu diễn chuyển động từ đầu. AIST++ lớn nhưng **không có nhãn điểm**, nên phải dùng proxy task.

Phân loại genre buộc backbone phải phân biệt Breaking với Waacking — tức phải mã hoá được nhịp độ và phong cách chuyển động, đúng thứ cần cho bài toán nhịp.

Đã loại: next-frame prediction. Model sẽ học nội suy mượt giữa hai frame liền kề — một kỹ năng gần như vô dụng cho việc đánh giá nhịp, và dễ đạt loss thấp mà không học được gì hữu ích.

Chi tiết và hạn chế (cùng `window_frames=60` nhưng AIST++ đã chuẩn hoá về 360 frame nên không ứng với cùng khoảng thời gian) ở `docs/temporal_pretrain.md`.

### 0.6 Vì sao phải HAI encoder riêng, không phải một encoder rồi bỏ feature ở tầng XGBoost

Đây là điểm thiết kế tinh tế nhất của tuần, và bỏ qua nó thì toàn bộ bảng ablation mất giá trị.

Kịch bản (b) theo định nghĩa là *"bỏ DTW, chỉ raw sequence"*. Cách làm dễ hơn — train **một** encoder 17 kênh rồi khi dựng feature cho (b) thì chỉ bỏ 4 cột DTW ở tầng XGBoost — **sai**, vì encoder đó đã thấy kênh DTW trong suốt quá trình train. Embedding của nó **đã hấp thụ** thông tin DTW. Bỏ cột DTW ở tầng sau không biến nó thành "LSTM-only" mà thành "LSTM-đã-biết-DTW", và so sánh (a) vs (b) trở nên vô nghĩa.

Nên (b) có encoder **16 kênh riêng**, train từ đầu, chưa bao giờ nhìn thấy `dtw_distance`.

Đó cũng là lý do tồn tại dòng **(c−)**: nó dùng embedding 16 kênh nhưng vẫn nối 4 cột DTW ở tầng XGBoost. Nhờ vậy tách bạch được hai thứ vốn bị trộn lẫn:

| | kênh dtw trong input LSTM | cột DTW ở tầng XGBoost |
|---|---|---|
| (b) LSTM-only | không | không |
| (c−) | không | có |
| (c) DTW+LSTM | có | có |

Thiếu (c−), (b) và (c) khác nhau **hai thứ cùng lúc** và không thể quy sự khác biệt cho nguyên nhân nào.

### 0.7 Vì sao so qua một XGBoost "tạm", không so trực tiếp prediction của LSTM

Kịch bản (a) DTW-only **không có model nào** — nó chỉ là 4 con số. Muốn so (a) với (b) và (c) một cách công bằng thì cả ba phải đi qua **cùng một bộ hồi quy**; nếu không thì ta đang so "model này với feature kia" chứ không so lượng thông tin trong feature.

XGBoost đóng vai bộ hồi quy chung đó. Hyperparameter copy nguyên văn từ `scripts/eval_spatial_xgb.py` của Person 1, nên bảng của hai người so được trực tiếp với nhau.

Đã loại: hồi quy tuyến tính (không bắt được quan hệ phi tuyến, sẽ hạ thấp mọi kịch bản không đồng đều); dùng thẳng prediction của LSTM (không áp dụng được cho (a), nên không so được).

Chữ "tạm" là có chủ ý: XGBoost chính thức của đồ án là ở Tuần 6 trên feature fusion của cả 3 người. Ở đây nó chỉ là dụng cụ đo.

### 0.8 Vì sao GroupKFold theo `person_id`, không phải KFold thường

20 người nhảy 217 video — tức **một người nhảy nhiều bài**.

KFold ngẫu nhiên sẽ để cùng một người xuất hiện ở cả train lẫn val. Khi đó model có thể học "người P007 nhảy kiểu này, thường được chấm khoảng 74 điểm" và ăn điểm trên val mà không hề học cách **đánh giá** chuyển động. MAE sẽ đẹp lên một cách giả tạo.

Nhóm theo `person_id` đảm bảo mọi video của một người nằm trọn một phía, nên con số đo được là khả năng tổng quát hoá sang **người chưa từng thấy** — đúng tình huống khi triển khai thật.

Đã loại: KFold thường (leakage như trên); nhóm theo `dance_id` (chỉ 12 bài nên mỗi fold quá to, và nó trả lời câu hỏi khác — "tổng quát sang *bài* mới", vốn là thí nghiệm held-out-dance của Tuần 6).

Đây là cùng một cạm bẫy với việc chia ở cấp cửa sổ (§1.1), chỉ ở một tầng khác: ở đó là cửa sổ cùng video nằm hai phía, ở đây là video cùng người nằm hai phía.

### 0.9 Vì sao kiểm định t ghép cặp, không nhìn hai khoảng mean ± std

Các fold dùng **chung** tập val cho mọi setup. Fold nào khó thì **mọi** setup đều tệ, nên phương sai giữa các fold lớn và làm hầu hết các khoảng `mean ± std` phủ lên nhau — che mất khác biệt có thật.

Ghép cặp khử đúng yếu tố đó: nó chỉ nhìn **hiệu** giữa hai setup trong từng fold.

Ví dụ cụ thể trong bảng này: (a) đạt 3.198 ± 0.503 còn (c) đạt 2.306 ± 0.218 — hai khoảng phủ nhau, nhìn qua tưởng không kết luận được. Nhưng hiệu theo từng fold là +1.265, +0.970, +0.826, +0.283, +1.117 — **dương ở cả 5 fold**, cho p = 0.006.

Đã loại: t-test độc lập (bỏ qua cấu trúc ghép cặp, mất gần hết power ở n=5).

### 0.10 Vì sao báo cả Pearson lẫn Spearman

Pearson nhạy với outlier. Trường hợp cụ thể: video `D01_P003_T01` (xem §7.2) kéo Pearson từ −0.33 lên −0.44, trong khi Spearman gần như không nhúc nhích (−0.368 vs −0.376).

Spearman đo **thứ hạng**, hợp với mục tiêu cuối cùng là xếp hạng người nhảy chứ không phải dự đoán đúng con số tuyệt đối.

Nên quy tắc là: báo cả hai, và khi hai con số lệch nhau nhiều thì coi đó là **tín hiệu có outlier cần đi tìm**, không phải cơ hội chọn con số đẹp hơn.

---

## 1. Hai quy tắc bắt buộc về dữ liệu

Hai quy tắc dưới đây quyết định việc mọi con số đo được có ý nghĩa hay không. Cả hai đều **được chặn bằng code**, không dựa vào quy ước — vì cả hai đều rất dễ vi phạm mà không nhận ra.

### 1.1 Chia train/val ở cấp video, nhóm theo `person_id`

`TemporalSeqDataset` phát **một sample cho mỗi cửa sổ 60 frame**, nên một video cho 3–11 sample. Nếu chia ngẫu nhiên ở cấp sample thì cửa sổ của cùng một video rơi về cả hai phía — mà các cửa sổ liền kề của cùng một người, cùng một bài, cùng một lần quay thì gần như giống hệt nhau. Khi đó `val_mae` đo khả năng **nhớ**, không phải khả năng tổng quát.

Thêm một tầng nữa: 20 người nhảy 217 video, nên một người xuất hiện ở nhiều bài. Không nhóm theo `person_id` thì model học được "người P007 thường được chấm khoảng 74 điểm" và ăn điểm mà chẳng học cách đánh giá chuyển động.

Nên quy tắc là: **mọi cửa sổ của một video, và mọi video của một người, phải nằm trọn một phía.**

Thực thi: `split_video_indices()` chia ở cấp video; `get_temporal_loaders_grouped()` mở ra cấp cửa sổ rồi assert không cửa sổ nào nằm hai phía. Mỗi lần chạy in ra `key_overlap=0 person_overlap=0`.

### 1.2 Target không được nằm trong input

`dtw_distance_total` vừa là một feature hữu ích, vừa là thứ được tile ra thành **kênh thứ 17 của input** (qua `_expand_dtw_to_frames`). Nếu lấy chính nó làm target thì model chỉ cần học đọc lại kênh 17 — loss giảm rất đẹp và con số thu được không nói gì về bài toán thật.

Nên: target mặc định là `khop_nhip` từ `annotations/scores.csv`, và `train.py` **`SystemExit`** nếu ai đó đặt `target_col="dtw_distance_total"` trong khi `use_dtw=True`, trừ khi truyền `--allow-dtw-target` để nói rõ là cố ý.

Đi kèm là một quy tắc nhỏ nhưng quan trọng: **không có fallback âm thầm.** Thiếu nhãn thì video bị **bỏ** và ghi log số lượng, chứ dataset không tự đổi sang một target khác.

### 1.3 Số đo phải đúng đơn vị

`val_mae` tính trên cửa sổ và ở thang đã normalize — **không so được** với baseline. History vì vậy ghi thêm:

- `val_mae_raw` = `val_mae × (tmax − tmin)`, quy về điểm thật;
- `val_mae_video_raw` = gộp prediction các cửa sổ theo video rồi mới tính MAE ở thang gốc.

`val_mae_video_raw` là số duy nhất đặt cạnh được baseline ở §4, và là số nên trích dẫn.

> Cùng cạm bẫy ở §1.1 còn tái xuất ở **tầng embedding**, nơi nó khó thấy hơn nhiều — xem §3.

---

## 2. Protocol

| Hạng mục | Giá trị |
|---|---|
| Target | `khop_nhip` (0–100) |
| Split chính | official `scores_train.csv` / `scores_test.csv` = **183 / 34** video, **16 / 4** person, rời nhau hoàn toàn |
| Val persons | P001, P002, P016, P018 |
| Split phụ | GroupKFold 5 fold theo `person_id` |
| Cửa sổ | 60 frame (~2 s ở 30 fps); hop 60 lúc train, **hop 30 lúc infer** |
| Gộp cửa sổ → video | `mean ⊕ std` → 64 chiều |
| Giảm chiều | StandardScaler → PCA(32), **fit chỉ trên hàng train** |
| XGBoost | hyperparameter copy nguyên văn từ `scripts/eval_spatial_xgb.py` (Person 1) để hai bảng so sánh được |
| Normalize target | min/max của **tập train** (63.0 → 100.0), không phải của cả bộ |

### Vì sao `mean ⊕ std`

- **Bất biến theo độ dài là bắt buộc.** T ∈ [219, 690] → 3–11 cửa sổ ở hop 60, 6–22 ở hop 30. Không thể ghép cố định.
- **`std` là cơ chế duy nhất để BiLSTM có thể hơn DTW.** `dtw_distance_total` là một scalar tổng. Độ lệch chuẩn của embedding giữa các cửa sổ mã hoá *mức đồng đều* của timing trong cả bài: người đúng nhịp nửa đầu rồi trôi khác hẳn người lệch đều nhẹ, nhưng hai người đó có thể có cùng `dtw_distance_total`. Chỉ pooling `mean` là ném thông tin đó đi, và khi ấy (b)/(c) gần như không còn headroom so với (a).
- **`max` không làm mặc định.** "Cửa sổ tệ nhất" nghe hợp lý cho điểm nhịp, nhưng với 3–11 cửa sổ thì đó là thống kê thứ tự một mẫu — variance rất cao và đổi nghĩa theo độ dài video. Có sẵn ở `--aggregate mean_std_minmax` để kiểm tra độ nhạy, báo một dòng, không làm số chính.
- **hop 30 lúc infer dù train hop 60.** Encoder chỉ thấy từng cửa sổ 60 frame nên bất biến với hop; giảm nửa hop làm đôi số mẫu đứng sau mỗi mean/std → bớt nhiễu gộp. Các cửa sổ chồng nhau đều thuộc cùng một video, mà video thì nằm trọn một phía của split, nên **không tạo leakage**.

---

## 3. Quy tắc chống leakage encoder

Đây là cạm bẫy tinh vi nhất của phần đánh giá: chính quy tắc §1.1 tái xuất ở một tầng trừu tượng cao hơn, và ở đó khó phát hiện hơn nhiều.

Embedding **là một hàm đã fit vào nhãn `khop_nhip`**. Nếu một video từng nằm trong tập train của encoder thì vector 64 chiều của nó đã hấp thụ nhãn của chính nó. Đưa vào XGBoost rồi eval trên đó là đo memorization, không phải generalization.

> **Quy tắc:** với mọi kịch bản, tập video mà encoder được train phải là **tập con của hàng train XGBoost**. Không video eval nào được xuất hiện trong tập train của bất kỳ encoder nào.

Thực thi:

1. Cả hai encoder train với `--split official`, tức chỉ trên 183 video của `scores_train.csv`. 34 video test không bị chạm tới — kể cả khi tính min/max để normalize target.
2. `eval_temporal_xgb.py` chạy trên đúng split đó, nên hàng val của XGBoost **chính là** tập video encoder chưa từng thấy.
3. **Assert, không giả định.** Checkpoint mang theo `config.train_videos`; script giao nó với `val_keys` và dừng nếu có phần tử chung. Checkpoint không mang trường đó thì không chứng minh được là không leakage, nên cũng bị từ chối. Đây là cái chặn quan trọng nhất cho fusion Tuần 6.
4. Trong `corr_temporal_khop_nhip.py`, cột `in_sample` đánh dấu mọi đại lượng dẫn từ embedding trên split `train`/`all`. **Chỉ dòng `in_sample=False` được dùng làm bằng chứng.** Để nó là một cột thay vì một footnote là cách ngăn con số bị trích dẫn sai ở Tuần 6.

---

## 4. Baseline tầm thường — đọc TRƯỚC mọi số của model

`khop_nhip` có spread rất hẹp: mean **76.82**, std **4.11**, range [63, 100]. Mean theo từng bài chỉ trải 72.4 → 80.0.

Hệ quả, đo trên split official 183/34:

| Bộ dự đoán | MAE | RMSE |
|---|---|---|
| Hằng số = mean của train | **2.850** | 3.787 |
| Mean theo `dance_id` của train | **1.902** | 2.650 |

**Mọi con số MAE trong bảng Tuần 5 là vô nghĩa nếu không đặt cạnh hai dòng này.** Một mô hình đạt MAE 2.6 nghe có vẻ tốt trên thang 0–100, nhưng nó còn thua dự đoán "trung bình của bài này".

Đó cũng là lý do bảng có khối `+dance` (one-hot `dance_id`): nếu không có, mọi kịch bản chủ yếu đang suy ra *"đây là bài nào"* và câu hỏi cần trả lời bị lấp. Có one-hot thì bảng đo **xếp hạng người nhảy trong cùng một bài** — câu hỏi deployable, và là chỗ DTW-vs-LSTM mới phân định được.

---

## 5. Power thống kê

`n_val = 34`. Với residual std ~2.6, sai số chuẩn của MAE ≈ **±0.45 điểm**.

Nghĩa là **chênh lệch dưới ~1 điểm giữa các kịch bản không phân định được trên một split**. Đây là lý do có bảng GroupKFold 5 fold: `MAE_mean ± MAE_std` qua các fold mới là con số đáng trích dẫn, và biểu đồ `04_cv_fold.png` cho thấy thanh lỗi có phủ nhau hay không.

Nếu các kịch bản không tách nhau quá nhiễu, kết luận đúng là *"không kết luận được ở quy mô dữ liệu này"* — **không** được chọn con số đẹp hơn để kể một câu chuyện.

**Kết quả thực tế.** Kiểm định t ghép cặp trên 5 fold (các fold dùng chung tập val cho mọi setup, nên ghép cặp khử được yếu tố "fold khó thì mọi setup đều tệ"):

- (a) DTW-only → **(c) DTW+LSTM: ΔMAE = +0.892, t(4)=5.28, p=0.006** → tách được.
- Baseline dance-mean → (c): ΔMAE = +0.224, t(4)=1.65, **p=0.17** → *không* tách được.

Tức là hai vế: BiLSTM học được thứ DTW không mã hoá, nhưng chưa thắng được baseline "trung bình theo bài". Số đầy đủ ở `experiments/temporal_dl/ablation_paired_tests_khop_nhip.csv`.

Lưu ý bảng CV chỉ có 5 fold nên df=4, power thấp: mọi kết luận "không có ý nghĩa" phải đọc là *"chưa đủ bằng chứng ở quy mô này"*, không phải *"đã chứng minh là không khác biệt"*.

---

## 6. Các kiểm tra tính hợp lệ

Chạy xong phải đúng, lệch là pipeline sai — dừng và sửa **trước khi** đọc bất kỳ số nào của model:

| Kiểm tra | Phải bằng |
|---|---|
| `dtw_distance_total × khop_nhip` @ split=all | n=217, Pearson **−0.4419**, Spearman **−0.3764** |
| `Baseline global-mean` @ official | MAE **2.850**, RMSE **3.787**, n_train=183, n_val=34 |
| `Baseline dance-mean` @ official | MAE **1.902**, RMSE **2.650** |
| `(a) DTW-only` val Pearson | trong khoảng **[0.25, 0.55]** — cao hơn hẳn thì nghi leakage |

Các sanity check được cài sẵn trong code:

| Mã | Kiểm tra | Cơ chế |
|---|---|---|
| S1 | Guard leakage nổ thật | truyền một checkpoint không có `config.train_videos` → exit 1 |
| S2 | Split rời nhau | in và assert mỗi lần chạy: `key_overlap=0 person_overlap=0` |
| S3 | Không cửa sổ nào ở cả hai phía | assert trong `get_temporal_loaders_grouped` |
| S4 | Guard target tự tham chiếu | `--target-col dtw_distance_total` không kèm `--allow-dtw-target` → exit 1 |
| S5 | Control nhãn xáo | `--shuffle-labels` → mọi kịch bản phải tụt về ≈2.85 và \|r\|≈0 |
| S6 | Encoder suy biến | `embeddings_sanity_*.txt`: >50% chiều có std < 1e-6 ⇒ ReLU cuối đã chết |
| S7 | Không lệch train/infer | `build_temporal_sequence()` phải `allclose` với cửa sổ đầu dataset lưu |

S5 là kiểm tra mạnh nhất: nếu một kịch bản vẫn "chạy tốt" trên nhãn đã xáo thì có đường thông tin ngoài ý muốn (feature dẫn từ nhãn, hoặc PCA fit trên cả bảng) — phải tìm ra trước khi báo cáo bất cứ điều gì.

---

## 7. Các caveat bắt buộc nêu trong báo cáo

### 7.1 `khop_nhip` không phải điểm chuyên gia

Cột `khop_nhip` trong `annotations/scores.csv` trùng **byte-for-byte** với `score_rhythm` trong `annotations/demo_source/labels_auto.csv`, mà cột `notes` của file đó ghi `auto`. `grep -r score_rhythm` không ra script sinh nào trong `src/` hay `scripts/`.

Quy trình chấm điểm của nhóm yêu cầu điểm do **người biết nhảy / dạy nhảy** chấm. Vậy **mọi correlation trong báo cáo này là correlation với một nhãn auto chưa rõ nguồn gốc, không phải với đánh giá của con người.**

Thêm bối cảnh: `corr(khop_nhip, khop_dong_tac) = 0.683` và `corr(khop_nhip, tong_diem) = 0.763`. Tức là ~47% phương sai của nhãn "khớp nhịp" đã nằm trong điểm "khớp động tác" — nhãn này mang rất ít thông tin nhịp độc lập.

Tuần 5 vẫn giao trên dữ liệu đang có. Nhưng thay cột này bằng điểm người chấm thật là **tiền đề cho mọi claim về mức đồng thuận với con người**, và là việc dữ liệu đáng làm nhất của Tuần 6. Lưu ý `annotations/demo_source/` bị `.gitignore` loại — đó là lý do điều này dễ bị bỏ sót.

### 7.2 Dataset chứa chính video reference như một performer

`poses/dance_001/D01_P003_T01.npy` **giống hệt từng byte** `poses/dance_001/dance_001_ref.npy` (cùng 312 frame, geometry `allclose` tuyệt đối). Nó được chấm 100 / 100 / 100 / 300 và có `dtw_distance_total = 0`.

Một điểm duy nhất này chiếm **1/4 giá trị Pearson**:

| Tập | Pearson | Spearman |
|---|---|---|
| n=217 (đầy đủ) | **−0.4419** | −0.3764 |
| n=216 (bỏ self-compare) | **−0.3349** | −0.3677 |

Spearman gần như không đổi vì dựa trên hạng — đó là con số bền vững hơn ở đây. Báo cáo phải ghi **cả hai**, không chọn con số đẹp hơn.

Video này nằm trong split **train**, nên các metric trên tập val không bị ảnh hưởng. Việc có nên loại nó khỏi `scores.csv` hay không là quyết định của cả nhóm (file nhãn thuộc Person 3), Person 2 chỉ nêu.

### 7.3 `flag_ratio` là đại lượng *trong* một video — đừng đọc tương quan thấp của nó như lỗi

`flag_ratio` (tỉ lệ đoạn bị `flag_suspicious_segments` gắn cờ) có Pearson với `khop_nhip` chỉ **−0.078** (n=217).

**Con số đó là đúng như dự đoán, không phải dấu hiệu rule engine hỏng.** `run_dtw_all.py` gọi `flag_suspicious_segments(win_dists)` cho **từng video**, với ngưỡng `mean + 1.5·std` tính trên **chính các đoạn của video đó**. Vậy nó là một thang **tương đối trong nội bộ một video**, không phải thang tuyệt đối giữa các video.

Hệ quả toán học: với gần như mọi dạng phân phối, tỉ lệ điểm vượt `mean + 1.5σ` xấp xỉ 6–7%. Quan sát thực tế khớp — 224 cờ trên ~3085 đoạn ≈ **7.3%** (lý thuyết phân phối chuẩn ≈ 6.7%). Tức `flag_ratio` **gần như là hằng số theo thiết kế**, nên về nguyên tắc nó *không thể* tương quan mạnh với một điểm tổng hợp cho cả video.

Cụ thể hơn, nó đo **độ không đồng đều**, không đo **độ tệ**:

- người nhảy tệ đều → mean cao, std thấp → **ít** cờ;
- người nhảy tốt nhưng trượt một đoạn → đoạn đó vượt ngưỡng → **có** cờ.

Và đó đúng là mục đích đã ghi trong docstring của hàm: *"Person 3 sẽ dựa vào cờ này để tập trung gán nhãn loại lỗi chi tiết"* — nhiệm vụ của nó là **phân loại ưu tiên trong nội bộ một video**, để người gán nhãn chỉ phải xem kỹ vài đoạn đáng ngờ thay vì cả bài.

**Phép đánh giá đúng** cho rule engine là precision/recall của cờ so với các đoạn trong `annotations/error_labels.csv`. **Chưa làm được**, vì file đó hiện mới chỉ có header. Khi Person 3 gán nhãn xong thì đó mới là lúc biết ngưỡng 1.5·σ có hợp lý hay không.

> **Cái bẫy cần tránh khi đọc bảng correlation.** Đem một công cụ đo *trong một video* đi tương quan *giữa các video* thì gần như luôn ra r ≈ 0, và rất dễ kết luận nhầm thành "công cụ bị hỏng, cần hiệu chỉnh ngưỡng". Dòng `flag_ratio` trong `correlation_khop_nhip.csv` phải đọc theo đúng nghĩa đó.

Một quan sát phụ có giá trị thật: `dtw_seg_min` tương quan mạnh hơn cả `dtw_distance_total` (−0.485 so với −0.442) — đáng chú ý khi thiết kế feature cho fusion Tuần 6.

---

### 7.4 Phần lớn tín hiệu nằm ở độ lớn embedding, không phải cấu trúc

`temporal_emb_l2norm` — chỉ là chuẩn L2 của vector 64 chiều — một mình đạt Spearman **−0.822** với `khop_nhip` trên tập val, gần bằng cả PC1 (−0.834).

Nghĩa là model học một thang "chất lượng" khá thô chứ không phải một biểu diễn nhịp phong phú. Đừng mô tả embedding như thể nó mã hoá cấu trúc timing tinh vi.

Liên quan: correlation trên val (0.71–0.80) **cao hơn** trên train (0.65–0.68). Model không thể tổng quát tốt hơn chỗ nó được fit, nên đây gần như chắc chắn là nhiễu của n=34 với đúng 4 person — không phải bằng chứng model tổng quát tốt.

### 7.5 Bẫy: dòng `(a+) DTW-only extended`

Trên split official nó đạt MAE 1.917, gần bằng dance-mean 1.902 — rất dễ bị đọc thành *"chỉ cần thêm vài feature DTW là đủ, không cần DL"*.

Trên CV nó chỉ đạt **2.891**, tệ hơn cả (b) 2.618 lẫn (c) 2.306, và so với (a) thì p=0.13.

Nguyên nhân gần như chắc chắn: `(a+)` thêm feature `T` (số frame), mà độ dài video gắn chặt với `dance_id`. Nó đang gián tiếp mã hoá *"đây là bài nào"* chứ không đo nhịp — đúng kiểu feature chỉ hoạt động trên một split cụ thể.

Cùng cơ chế giải thích vì sao `(c)+dance` đạt 1.741 trên official nhưng lại *tệ hơn* `(c)` trên CV (2.401 vs 2.306, p=0.24): one-hot `dance_id` không thêm thông tin thật, chỉ thêm chiều cho XGBoost overfit trên 34 dòng.

## 8. API deliverable

```python
from src.temporal_dl import load_temporal_checkpoint, temporal_model

model, ckpt = load_temporal_checkpoint(
    "experiments/temporal_dl/temporal_khopnhip_dtw_official.pth")
embedding, dtw_features = temporal_model(seq, model=model, dtw_row=row)

# embedding            : (64,) = mean ⊕ std trên embedding từng cửa sổ
# dtw_features["vector"]: (4,) [total, seg_mean, seg_std, seg_max]
```

`dtw_features["vector"]` **cố ý trùng khít thứ tự** `_dtw_row_features` trong `scripts/eval_spatial_xgb.py` của Person 1, để fusion Tuần 6 concat được mà không cần adapter.

Các khoá khác của `dtw_features`: `dtw_distance_total`, `dtw_seg_{mean,std,max,min}`, `n_segments`, `n_flagged`, `flag_ratio`, `predicted_score` (thang [0,1]), `predicted_target` (điểm `khop_nhip` sau de-normalize), `n_windows`, `window_frames`, `hop_frames`, `aggregate`, `input_dim`, `source` ∈ {`dtw_row`, `input_channel`, `unavailable`}.
