# docs/error_taxonomy.md — DRAFT v0.1 (for team review)

**Status:** DRAFT — for team review, not final specification.
**Purpose:** Định nghĩa 5 error_type dùng để annotator gán nhãn `annotations/error_labels.csv`, và sau đó dùng làm target cho Error DL multi-label model (Person 3).
**Scope:** Taxonomy này phải dùng được cho **mọi dance** trong project (không gắn với 1 động tác cụ thể của 1 bài).

---

## 0. Nguyên tắc nền tảng (đọc trước khi dùng taxonomy này)

1. DTW windowed distance **chỉ là tín hiệu gợi ý** để annotator biết đoạn nào nên xem kỹ hơn. DTW cao **không phải bằng chứng** rằng đoạn đó chắc chắn có lỗi, và **không xác định** đoạn đó thuộc error_type nào. Annotator luôn là người quyết định cuối cùng dựa trên việc xem lại video.
2. Một error_type không bao giờ được suy ra chỉ từ giá trị DTW. DTW cao có thể do: off_beat, wrong_move, low_amplitude, wrong_direction, missed_move, hoặc do lỗi kỹ thuật (MediaPipe mất landmark) — annotator phải xem video để phân biệt.
3. Taxonomy định nghĩa theo **loại sai lệch tổng quát**, không theo động tác cụ thể của 1 bài nhảy.

---

## A1. Mục đích sử dụng (nhắc lại luồng)

```
dtw_distance_windowed (Person 2)
        ↓
    [đoạn nghi ngờ lệch]
        ↓
annotator xem lại đoạn đó, áp dụng taxonomy này
        ↓
annotations/error_labels.csv
        ↓
Error DL multi-label model (Person 3)
```

Taxonomy phải đủ rõ để **2 annotator khác nhau** đọc cùng định nghĩa và cho ra nhãn tương đối nhất quán trên cùng 1 đoạn video (đây là tiêu chí chấp nhận, nên pilot-test bằng cách cho 2 người gán nhãn độc lập — xem mục A1.1).

### A1.1. Pilot test bắt buộc trước khi coi taxonomy là chính thức
- [ ] Ít nhất 1 video được 2 annotator gán nhãn độc lập.
- [ ] So sánh kết quả, ghi lại % đoạn đồng ý hoàn toàn / đồng ý một phần / bất đồng.
- [ ] Nếu bất đồng nhiều ở 1 cặp lỗi cụ thể → quay lại sửa mục "Boundary / distinction" tương ứng.
- **`[NEEDS REVIEW]`** — chưa có số liệu pilot test thật, cần nhóm chạy thử trước khi chốt bản v1.0.

---

## A2–A3. Chi tiết từng error_type

### 1. off_beat

**Tên tiếng Việt:** Trễ/sớm nhịp

**Definition:** Performer thực hiện một động tác về cơ bản đúng hình dạng/kỹ thuật so với reference, nhưng **thời điểm bắt đầu hoặc kết thúc động tác đó lệch đáng kể** so với thời điểm tương ứng trong reference (sau khi đã time-align ở mức tổng thể).

**Positive examples:**
1. Reference giơ tay lên tại giây 12.0; performer thực hiện đúng động tác giơ tay (cùng hình dạng, cùng biên độ) nhưng tại giây 12.6 — động tác đúng, chỉ lệch thời điểm → `off_beat`.
2. Reference có 1 cú xoay người kết thúc ở giây 20.0 để chuyển sang động tác tiếp theo; performer xoay đúng kiểu nhưng bắt đầu xoay sớm hơn 0.8 giây so với reference, khiến cả đoạn chuyển động "lệch pha" với nhạc/reference → `off_beat`.

**Negative examples (dễ nhầm nhưng KHÔNG phải off_beat):**
1. Performer trễ nhịp nghiêm trọng đến mức khi thực hiện đến đoạn mới của reference thì vẫn đang làm dở động tác cũ, dẫn đến **bỏ hẳn** động tác tiếp theo của reference trong khoảng thời gian đó → đây là `missed_move`, không phải `off_beat` (xem Boundary bên dưới).

**Boundary / distinction:**
- **off_beat vs missed_move:** Tua chậm lại đoạn nghi ngờ — nếu động tác **có xuất hiện** (dù lệch thời điểm) và có thể nhận ra rõ hình dạng tương ứng với động tác reference → `off_beat`. Nếu động tác **hoàn toàn không xuất hiện** trong cả đoạn (performer bỏ qua, chuyển thẳng sang động tác khác) → `missed_move`.

**Annotation rule:** Nếu động tác xuất hiện đầy đủ, đúng hình dạng, nhưng thời điểm xuất hiện lệch so với reference (ngưỡng cụ thể: `[NEEDS REVIEW — chưa có threshold định lượng]`) → gán `off_beat`.

---

### 2. wrong_move

**Tên tiếng Việt:** Sai động tác

**Definition:** Tại 1 đoạn thời gian nhất định, performer thực hiện một động tác **khác hẳn về bản chất** so với động tác mà reference quy định tại đoạn tương ứng (không phải cùng động tác bị lệch nhịp hay lệch hướng).

**Positive examples:**
1. Reference thực hiện động tác vỗ tay qua đầu; performer lại thực hiện động tác đưa 2 tay sang ngang — 2 động tác khác hẳn về bản chất, không phải biến thể của cùng 1 động tác → `wrong_move`.
2. Reference bước chân phải lên trước rồi khuỵu gối (lunge); performer lại nhảy tại chỗ (jump) ở đúng đoạn đó → `wrong_move`.

**Negative examples:**
1. Reference xoay người sang trái 90°, performer xoay đúng kiểu động tác nhưng sang phải 90° — đây là cùng loại động tác (xoay người), chỉ sai hướng → `wrong_direction`, không phải `wrong_move`.
2. Reference đưa tay lên cao hết cỡ, performer cũng đưa tay lên nhưng chỉ đến ngang vai — cùng loại động tác, chỉ khác biên độ → `low_amplitude`, không phải `wrong_move`.

**Boundary / distinction:**
- **wrong_move vs missed_move:** Nếu performer thực hiện **một động tác thay thế** (dù sai) tại đoạn đó → `wrong_move`. Nếu performer **không làm động tác nào cả** (đứng yên, hoặc chuyển tiếp mà bỏ qua hẳn đoạn đó) → `missed_move`.
- **wrong_move vs wrong_direction:** Nếu về bản chất vẫn là "cùng 1 loại động tác" (cùng nhóm cơ, cùng hình dạng chuyển động cơ bản) nhưng khác hướng không gian → `wrong_direction`. Nếu động tác thực hiện thuộc nhóm hoàn toàn khác (khác bộ phận cơ thể chính, khác kiểu chuyển động) → `wrong_move`.

**Annotation rule:** Nếu động tác thực hiện không thể coi là "biến thể lệch" của động tác reference (không cùng nhóm chuyển động cơ bản) → gán `wrong_move`.

---

### 3. low_amplitude

**Tên tiếng Việt:** Biên độ thấp

**Definition:** Performer thực hiện **đúng loại động tác** (đúng hình dạng, đúng hướng, đúng thời điểm tương đối) nhưng **biên độ chuyển động** (khoảng cách di chuyển của khớp/chi) nhỏ hơn đáng kể so với reference.

**Positive examples:**
1. Reference giơ tay thẳng lên quá đầu; performer cũng giơ tay lên nhưng chỉ đến ngang vai — đúng động tác, đúng hướng, chỉ thiếu biên độ → `low_amplitude`.
2. Reference nhảy lên cao (jump) rõ rệt; performer chỉ nhún nhẹ, bàn chân gần như không rời mặt đất → `low_amplitude`.

**Negative examples:**
1. Performer gần như không di chuyển chi nào trong cả đoạn, không có dấu hiệu "cố thực hiện động tác nhưng yếu" mà là **hoàn toàn bỏ qua** → đây là `missed_move`, không phải `low_amplitude` (xem Boundary bên dưới).

**Boundary / distinction:**
- **low_amplitude vs missed_move:** Nếu có thể nhận ra **ý định/hình dạng khởi đầu** của động tác (dù yếu) → `low_amplitude`. Nếu hoàn toàn không có dấu hiệu nào của động tác đó → `missed_move`. Ranh giới định lượng cụ thể (vd. biên độ dưới bao nhiêu % thì coi là "không có" thay vì "yếu") — `[NEEDS REVIEW — chưa có threshold định lượng]`.
- **low_amplitude vs wrong_move:** Nếu vẫn là đúng loại/hướng chuyển động, chỉ thiếu biên độ → `low_amplitude`. Nếu biên độ thấp đến mức động tác thực hiện trông giống một động tác khác hẳn (không còn nhận ra là "yếu" của động tác gốc) → cân nhắc `wrong_move` thay vì `low_amplitude` — annotator tự đánh giá theo ngữ cảnh, ghi chú lại nếu phân vân.

**Annotation rule:** Nếu nhận ra đúng loại/hướng động tác nhưng biên độ rõ ràng nhỏ hơn reference → gán `low_amplitude`.

---

### 4. wrong_direction

**Tên tiếng Việt:** Sai hướng

**Definition:** Performer thực hiện **đúng loại động tác** nhưng **hướng di chuyển trong không gian hoặc hướng cơ thể** (trái/phải, trước/sau, chiều xoay) bị sai so với reference.

**Positive examples:**
1. Reference bước chân phải sang phải; performer bước chân phải sang trái — cùng loại động tác (bước chân), sai hướng → `wrong_direction`.
2. Reference xoay người theo chiều kim đồng hồ; performer xoay đúng góc độ nhưng theo chiều ngược lại → `wrong_direction`.

**Negative examples:**
1. Performer xoay người đúng hướng nhưng góc xoay chỉ bằng một nửa reference (vd. 45° thay vì 90°) — đây là vấn đề biên độ/mức độ, không phải hướng → `low_amplitude`, không phải `wrong_direction`.

**Boundary / distinction:**
- **wrong_direction vs wrong_move:** Nếu động tác vẫn thuộc cùng nhóm chuyển động cơ bản (chỉ khác hướng không gian) → `wrong_direction`. Nếu động tác thực hiện khác hẳn về bản chất (không chỉ là đổi hướng của cùng 1 động tác) → `wrong_move`.

**Annotation rule:** Nếu động tác đúng loại nhưng hướng không gian/chiều xoay ngược hoặc lệch so với reference → gán `wrong_direction`.

---

### 5. missed_move

**Tên tiếng Việt:** Bỏ sót động tác

**Definition:** Tại 1 đoạn thời gian mà reference có 1 động tác cụ thể, performer **không thực hiện động tác nào tương ứng** (đứng yên, hoặc bỏ qua thẳng sang đoạn tiếp theo).

**Positive examples:**
1. Reference có 1 cú giậm chân tại giây 18; performer trong khoảng giây 17.5–19 gần như đứng yên, không có chuyển động tương ứng nào → `missed_move`.
2. Reference thực hiện 1 chuỗi 2 động tác liên tiếp (vỗ tay rồi xoay người); performer chỉ vỗ tay rồi đứng yên, bỏ hẳn phần xoay người → đoạn xoay người đó được gán `missed_move`.

**Negative examples:**
1. Performer có cố thực hiện động tác, dù rất yếu/nhỏ (biên độ gần như không đáng kể nhưng vẫn nhận ra ý định) → `low_amplitude`, không phải `missed_move`.

**Boundary / distinction:** Xem các mục Boundary ở `off_beat`, `wrong_move`, `low_amplitude` phía trên — nguyên tắc chung: `missed_move` chỉ dùng khi **hoàn toàn không có chuyển động nào nhận diện được** tương ứng với động tác reference tại đoạn đó.

**Annotation rule:** Nếu không có bất kỳ dấu hiệu chuyển động nào tương ứng với động tác reference trong cả đoạn → gán `missed_move`.

---

## A4. Multi-label — nguyên tắc gán nhiều lỗi trên cùng 1 đoạn

**Nguyên tắc cho phép nhiều label:** Một đoạn (`start_time`–`end_time`) **có thể** được gán nhiều `error_type` cùng lúc nếu các khía cạnh sai lệch đó độc lập với nhau về mặt quan sát.

**Ví dụ 1 đoạn có 2 lỗi đồng thời:** Performer giơ tay (đúng loại động tác) nhưng (a) trễ 0.6 giây so với reference VÀ (b) biên độ chỉ đến ngang vai thay vì giơ thẳng → đoạn này được gán cả `off_beat` và `low_amplitude` (2 dòng riêng trong `error_labels.csv`, cùng `start_time`/`end_time`).

**Khi nào KHÔNG nên gán thêm 1 label chỉ vì nó là "hệ quả" của lỗi kia:**
- Nếu `off_beat` nghiêm trọng đến mức performer hoàn toàn bỏ lỡ động tác (không còn kịp thực hiện) → chỉ gán `missed_move`, **không** gán thêm `off_beat` (vì bản chất đoạn đó không còn "có động tác nhưng lệch thời điểm" nữa — xem Boundary ở mục off_beat).
- Nguyên tắc chung: chỉ gán thêm label thứ 2 nếu nó mô tả **một khía cạnh quan sát được độc lập**, không phải diễn giải lại cùng 1 hiện tượng bằng 2 cái tên.

**Khi annotator không chắc chắn:**
- `[NEEDS REVIEW]` — tài liệu hiện tại chưa quy định quy trình xử lý "uncertain case". Đề xuất 2 phương án:
  - *Phương án A (đề xuất):* annotator vẫn chọn `error_type` có khả năng cao nhất, ghi thêm vào **cột `note`** (text tự do, thêm ngoài 6 cột bắt buộc, không ảnh hưởng schema) để ghi lý do phân vân — dùng để nhóm review lại sau, không dùng để train model.
  - *Phương án B:* thêm cột `confidence` (0-1, annotator tự đánh giá độ chắc chắn) — có thể dùng để lọc dữ liệu "sạch" khi train, nhưng annotator phải đánh giá thêm 1 con số mỗi lần gán nhãn.

**Priority/hierarchy giữa 5 lỗi:** `[NEEDS REVIEW]` — tài liệu gốc chưa quy định thứ tự ưu tiên giữa 5 loại lỗi khi khó phân biệt. Không tự tạo hierarchy — đề xuất mặc định: annotator dựa vào mục Boundary/distinction của từng cặp lỗi để quyết định, nếu vẫn không rõ thì áp dụng Phương án A/B ở trên (uncertain case).

---

## A5. Severity — DRAFT, cần team review

**`[NEEDS REVIEW]`** — specification hiện tại chỉ có field `severity` (kiểu dữ liệu số, theo `error_labels.csv`), chưa định nghĩa thang đo chính thức. Đề xuất 2 phương án để nhóm thảo luận:

**Phương án A (đề xuất): 3 mức rời rạc**

| Mức | Giá trị số | Ý nghĩa | Ví dụ |
|---|---|---|---|
| Nhẹ | 0.3 | Sai lệch có thể nhận ra nhưng không ảnh hưởng nhiều đến cảm nhận tổng thể | off_beat trễ ~0.3–0.5s |
| Trung bình | 0.6 | Sai lệch rõ ràng, ảnh hưởng đến cảm nhận động tác | off_beat trễ ~0.5–1s |
| Nặng | 0.9 | Sai lệch nghiêm trọng, gần như phá vỡ hoàn toàn động tác dự kiến | off_beat trễ >1s hoặc gần như mất nhịp cả đoạn |

Ưu điểm: annotator dễ quyết định nhanh, nhất quán hơn giữa 2 người (không phải ước lượng số thực).

**Phương án B: thang liên tục 0.0–1.0 tự do**

Ưu điểm: chi tiết hơn. Nhược điểm: annotator khó nhất quán (vd. 1 người cho 0.55, người khác cho 0.65 cho cùng mức độ, không có ý nghĩa khác biệt thực sự).

**Vấn đề cần nhóm quyết định:** chọn A hay B; nếu chọn A, threshold định lượng cho từng error_type khác ngoài `off_beat` (`wrong_move`, `low_amplitude`...) cũng cần bổ sung ví dụ tương tự.

---

## A6. Bảng tổng hợp

### Bảng taxonomy

| error_type | Tên tiếng Việt | Definition ngắn | Ví dụ | Dễ nhầm với |
|---|---|---|---|---|
| off_beat | Trễ/sớm nhịp | Đúng động tác, lệch thời điểm | Giơ tay đúng kiểu nhưng trễ 0.6s | missed_move |
| wrong_move | Sai động tác | Thực hiện động tác khác bản chất | Vỗ tay thay vì đưa tay ngang | missed_move, wrong_direction |
| low_amplitude | Biên độ thấp | Đúng động tác/hướng, biên độ nhỏ hơn | Giơ tay chỉ đến ngang vai thay vì giơ thẳng | missed_move, wrong_move |
| wrong_direction | Sai hướng | Đúng loại động tác, sai hướng không gian | Xoay ngược chiều kim đồng hồ | wrong_move |
| missed_move | Bỏ sót động tác | Không thực hiện động tác nào tương ứng | Đứng yên khi reference giậm chân | (điểm neo để phân biệt các lỗi khác) |

### Bảng phân biệt cặp lỗi dễ nhầm

| Cặp lỗi | Điểm khác biệt chính |
|---|---|
| off_beat vs missed_move | off_beat: động tác có xuất hiện, chỉ lệch thời điểm. missed_move: động tác hoàn toàn không xuất hiện. |
| wrong_move vs missed_move | wrong_move: có 1 động tác thay thế (dù sai) xuất hiện. missed_move: không có động tác nào xuất hiện. |
| wrong_move vs wrong_direction | wrong_direction: cùng nhóm chuyển động cơ bản, chỉ khác hướng. wrong_move: khác hẳn nhóm chuyển động. |
| low_amplitude vs missed_move | low_amplitude: có dấu hiệu/ý định động tác dù yếu. missed_move: hoàn toàn không có dấu hiệu. |
| low_amplitude vs wrong_move | low_amplitude: đúng loại/hướng, chỉ thiếu biên độ. wrong_move: biên độ thấp đến mức không còn nhận ra là cùng động tác. |

---

## Checklist review cho phần Taxonomy

- [x] 5 error_type đã có tên và định nghĩa draft
- [x] Mỗi lỗi có ≥2 positive example, ≥1 negative example
- [x] Boundary/distinction đã có draft cho tất cả cặp lỗi dễ nhầm được liệt kê
- [x] Multi-label: nguyên tắc cho phép/không cho phép đã có draft
- [ ] Severity: còn `[NEEDS REVIEW]` — đã có 2 phương án đề xuất (A: 3 mức rời rạc, B: liên tục), cần nhóm chọn
- [ ] Đã chạy pilot test với ≥2 annotator trên ít nhất 1 video
- [ ] Xử lý "uncertain case" còn `[NEEDS REVIEW]` — đã có 2 phương án đề xuất (A: cột note, B: cột confidence)
- [ ] Priority/hierarchy giữa các lỗi: còn `[NEEDS REVIEW]`

**Trạng thái:** Đây vẫn là DRAFT v0.1 cho nhóm review — cần họp để chọn phương án cho các mục `[NEEDS REVIEW]` trước khi coi là v1.0 chính thức.
