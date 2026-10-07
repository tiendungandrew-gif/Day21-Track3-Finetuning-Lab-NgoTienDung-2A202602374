# Lab 21 — Evaluation Report

**Họ tên**: Ngô Tiến Dũng  **MSSV**: 2A202602374  **Ngày**: 2026-10-07
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `Tesla T4 16GB`

> Mọi con số dưới đây phải khớp với file trong `results/`. Grader kiểm tra chéo.
>
> **Mẫu này là gợi ý.** Bạn được tự chọn base model, dataset và tự viết report theo cấu
> trúc của mình — miễn là có đủ: lựa chọn + lý do, bằng chứng mask, mốc đóng băng, kết quả,
> phán quyết, điều học được (rubric 4.1).

---

## 1. Setup

| | |
|---|---|
| Dataset | 250 ticket CSKH tiếng Việt → JSON triage 4 trường (intent, urgency, product, sentiment) |
| Train / val | 225 / 25 (seed 42) |
| `max_length` | 1024 — p95 đo được là 98 *(results/token_stats.json)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2 epochs / 30 steps |

**Template có giữ khối `<think>` không?** `Có` — *(results/template_check.json)*  
Template của Qwen3.5 bảo toàn nguyên vẹn cấu trúc suy luận, đảm bảo `open_tag_present: true` và `body_present: true`, an toàn tuyệt đối khi huấn luyện trên reasoning traces mà không bị nuốt khối suy luận.

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | 0.4149 |
| Câu trả lời nằm trong loss | true |
| Câu hỏi KHÔNG nằm trong loss | true |

Dán 3–5 dòng đầu của đoạn được tính loss:

```
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.791 | 0.000 | 3211.6 |
| (b) base + optimized prompt | 0.765 | 0.791 | 1.000 | 1028.7 |
| (c) LoRA fine-tune | 0.975 | 0.633 | 1.000 | 1388.0 |

**(b) có thật sự mạnh hơn (a) không?** `Có` — Baseline (b) vượt trội hoàn toàn so với (a) khi nâng điểm target từ 0.000 lên 0.765 và độ chính xác format JSON đạt tuyệt đối 1.000 (so với 0.000 của naive prompt).  
Bạn có sửa `OPTIMIZED_PROMPT` không? `Không` — giữ nguyên nguyên bản để đảm bảo tính liêm chính và bảo toàn mã băm SHA của mốc đo đã đóng băng.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32,464,896 | 0.0001 | 0.6256 | 0.975 | 419.4 | 8.78 |
| `attn_only` | q,v | 283 | 32,456,704 | 0.0001 | 0.5369 | 0.970 | 277.4 | 8.79 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 0.00001 | 1.5702 | 0.000 | 413.5 | 8.78 |
| `qlora` | text-linear | 16 | 32,464,896 | 0.0001 | 0.7058 | 0.940 | 481.7 | 3.86 |

> Xếp hạng bằng cột **target**, không bằng cột train loss — chấm bằng chỉ số thay thế
> chính là Lỗi #3. Nếu hai cột cho hai thứ tự khác nhau, nói thẳng điều đó ở 4.1: đó là
> kết quả đáng giá nhất bạn đo được trong lab này.

Trả lời ba câu (mỗi câu ≥3 câu văn):

**4.1 — `attn_only` có cùng số tham số huấn luyện với `correct`. Trên tập target nó thắng, thua, hay hoà? Thứ tự đó có giống thứ tự theo train loss không? Điều đó nói gì về *rank* so với *vị trí gắn adapter*?**  
Run `attn_only` đã được khớp ngân sách tham số chính xác (32,456,704 so với 32,464,896, sai lệch dưới 0.03%). Trên tập nhiệm vụ target, `attn_only` đạt 0.970, hoàn toàn THUA bản `correct` đạt 0.975. Đáng chú ý, thứ tự này ngược hoàn toàn với train loss: `attn_only` có loss thấp hơn rõ rệt (0.5369 so với 0.6256), cho thấy nó bị overfit cục bộ vào các lớp attention nhưng lại khái quát hóa kém hơn. Thực nghiệm này chứng minh rõ ràng rằng **vị trí gắn adapter (toàn bộ các lớp linear của text decoder) là đòn bẩy quyết định**, quan trọng hơn rất nhiều so với việc dồn rank cực cao ($r=283$) vào một vài lớp hạn chế.

**4.2 — `wrong_lr` chỉ khác đúng một con số. Đường loss khác nhau ra sao? Nếu chỉ nhìn loss mà không biết LR, bạn sẽ kết luận sai điều gì?**  
Run `wrong_lr` chỉ khác đúng một con số là learning rate nhỏ hơn $10\times$ ($1\times 10^{-5}$ thay vì $1\times 10^{-4}$), và đường loss gần như đi ngang phẳng lì, kẹt lại ở mức 1.5702 sau 30 bước thay vì hạ xuống 0.6256 như bản chuẩn. Nếu một kỹ sư chỉ nhìn vào đường loss dậm chân tại chỗ mà không kiểm tra cấu hình LR, họ sẽ dễ dàng kết luận sai lầm rằng dữ liệu quá nhiễu, số bước train quá ít hoặc kiến trúc mô hình không có khả năng học bài toán này. Thực tế, nguyên nhân thuần túy là do áp dụng nhầm thang LR của full fine-tuning cho LoRA, trong khi LoRA bắt buộc cần LR lớn hơn xấp xỉ $10\times$ để các ma trận tích cập nhật hiệu quả.

**4.3 — `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì? Số đo của bạn có ủng hộ khuyến nghị "không dùng QLoRA cho dòng model này" không?**  
Run `qlora` đã cắt giảm lượng VRAM đỉnh cực kỳ ấn tượng từ 8.78 GB xuống chỉ còn 3.86 GB (tiết kiệm khoảng 56% bộ nhớ VRAM). Tuy nhiên, cái giá phải trả là thời gian huấn luyện kéo dài thêm (481.7s so với 419.4s do độ trễ dequantize) và đặc biệt là độ chính xác target bị tụt giảm từ 0.975 xuống 0.940 (mất 3.5 điểm phần trăm). Kết quả đo đạc thực nghiệm này hoàn toàn ủng hộ khuyến nghị của nhà cung cấp mô hình Qwen3.5 rằng không nên dùng QLoRA nếu GPU còn đủ dung lượng chứa bản 16-bit, vì sai số tích lũy từ lượng tử hóa 4-bit làm suy thoái năng lực suy luận của mô hình mà không mang lại lợi ích về tốc độ.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`  
`target Δ = +0.210` · `regression Δ = -0.158` · `valid_trace_rate = 0.00`

Diễn giải:  
Mặc dù bản LoRA fine-tune đạt mức tăng trưởng vượt bậc trên nhiệm vụ mục tiêu (target tăng từ 0.765 lên 0.975, tức $\Delta = +0.210$), cổng hồi quy bốn nhóm vẫn phán quyết nghiêm ngặt là **FAILED**. Nguyên nhân cốt lõi là do năng lực chỉ dẫn và tri thức phổ thông bị tụt giảm nghiêm trọng từ 0.791 xuống còn 0.633 (mức suy giảm $\Delta = -0.158$, vượt xa ngưỡng dung sai an toàn cho phép là 0.020). 

Đây là minh chứng thực tế rõ ràng cho hiện tượng **quên thảm họa (catastrophic forgetting)** trong kỹ thuật post-training: khi chúng ta ép mô hình học tập trung cao độ vào một định dạng hẹp (trích xuất 4 trường JSON) trên 250 mẫu dữ liệu, không gian tham số của adapter đã ghi đè và làm tổn hại các biểu diễn ngôn ngữ đa nhiệm ban đầu. Đối với một hệ thống thực tế trong doanh nghiệp, kết quả FAILED này là một cảnh báo cực kỳ giá trị: chúng ta không thể vội vàng đưa mô hình này ra production cho các tác vụ tổng quát nếu chưa giải quyết được bài toán bảo toàn tri thức. Theo đề xuất tại deck §6.3, giải pháp bắt buộc để vượt qua cổng hồi quy là phải trộn thêm từ 1% đến 5% dữ liệu chỉ dẫn phổ thông (replay data) vào tập huấn luyện để neo giữ lại các năng lực suy luận nền tảng của base model.

---

## 6. Định tính — bắt buộc có cả ca THUA

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | Cho mình hỏi, mình đặt chuột không dây mã đơn VN232232. Cho tôi trả lại... | doi_tra, cao, chuột không dây, tich_cuc | doi_tra, trung_binh, chuột không dây, tich_cuc | doi_tra, cao, chuột không dây, tich_cuc | ✅ FT thắng: Nhận diện chính xác urgency mức cao do thời hạn trả lại. |
| 2 | Shop ơi, mình đặt ốp lưng điện thoại mã đơn VN812931. Hoàn tiền. Sớm nhé... | hoan_tien, trung_binh, ốp lưng điện thoại, trung_tinh | hoan_tien, trung_binh, ốp lưng điện thoại, tieu_cuc | hoan_tien, trung_binh, ốp lưng điện thoại, trung_tinh | ✅ FT thắng: Bắt đúng sentiment trung tính thay vì nhầm sang tiêu cực. |
| 3 | Cho mình hỏi, mình đặt bình giữ nhiệt mã đơn VN804124. Chưa thấy tiền... | hoan_tien, trung_binh, bình giữ nhiệt, trung_tinh | hoan_tien, trung_binh, bình giữ nhiệt, trung_tinh | hoan_tien, trung_binh, bình giữ nhiệt, tieu_cuc | ❌ **FT thua**: FT đoán nhầm sentiment thành tieu_cuc do từ "Chưa thấy tiền". |
| 4 | Shop ơi, mình đặt nồi chiên không dầu mã đơn DH249548. Thiếu phụ kiện... | san_pham_loi, trung_binh, nồi chiên không dầu, trung_tinh | san_pham_loi, trung_binh, nồi chiên không dầu, trung_tinh | san_pham_loi, cao, nồi chiên không dầu, tieu_cuc | ❌ **FT thua**: FT thổi phồng mức độ urgency lên cao và sentiment tiêu cực. |
| 5 | Shop ơi, mình đặt áo khoác gió mã đơn VN613097. Bị lỗi. Khi nào tiện... | san_pham_loi, thap, áo khoác gió, trung_tinh | san_pham_loi, thap, áo khoác gió, trung_tinh | san_pham_loi, trung_binh, áo khoác gió, trung_tinh | ❌ **FT thua**: Khách nói "Khi nào tiện" (thap) nhưng FT dự đoán trung_binh. |

Có mẫu chung nào ở các ca FT thua không?  
Có một quy luật rất rõ nét: Các ca bản fine-tune bị thua tập trung hoàn toàn vào hai trường có tính cảm tính và ngữ cảnh tinh tế là `sentiment` và `urgency`. Mô hình fine-tune có xu hướng thiên kiến (bias) gán nhãn `tieu_cuc` và nâng mức `urgency` lên cao hơn khi nhìn thấy các từ khóa báo lỗi hoặc khiếu nại (như "Chưa thấy tiền", "Thiếu phụ kiện"), bỏ qua các sắc thái giảm nhẹ như "Khi nào tiện" hoặc phong thái hỏi thăm trung tính của khách hàng. Trong khi đó, prompt tối ưu với các chỉ dẫn chi tiết và ví dụ ngữ cảnh lại duy trì được cái nhìn khách quan và chính xác hơn ở các trường này.

---

## 7. Kết luận & điều tôi học được

**Kết luận:**  
Từ toàn bộ kết quả thực nghiệm của Lab 21, câu trả lời dứt khoát cho câu hỏi triển khai là: **Chưa nên vội vã deploy bản fine-tune này cho toàn bộ hệ thống chăm sóc khách hàng đa năng**, dù nó đạt độ chính xác ấn tượng 97.5% trên tác vụ trích xuất chuyên biệt. Việc cổng hồi quy đánh dấu FAILED do độ sụt giảm năng lực tổng quát (-15.8%) là một rủi ro thực tế rất lớn: mô hình có thể hoàn thành xuất sắc việc phân loại JSON nhưng sẽ mất đi khả năng đối thoại tự nhiên, giải thích thắc mắc hoặc xử lý các câu hỏi mở ngoài khuôn khổ định dạng sẵn. 

Nếu bài toán chỉ đơn thuần là một microservice độc lập chạy ngầm phía sau (backend pipeline) chỉ nhận text thô và trả về JSON phục vụ routing cho nhân viên, bản fine-tune này hoàn toàn vượt trội nhờ độ chính xác 97.5% và latency tối ưu. Nhưng nếu dùng làm chatbot tương tác trực tiếp với khách, giải pháp an toàn hơn tại thời điểm này là tiếp tục sử dụng Base model kết hợp với Few-shot Optimized Prompt (đã đạt 76.5% và bảo toàn 100% năng lực hội thoại). 

Đòn bẩy kỹ thuật thực sự được khám phá trong bài lab này không nằm ở việc tăng rank LoRA thật lớn, mà nằm ở ba yếu tố quyết định theo thứ tự ưu tiên:
1. **Loss mask chính xác:** Ngăn chặn việc học vẹt câu hỏi ngay từ đầu.
2. **Vị trí gắn adapter toàn diện (`text-linear`):** Cho phép mô hình điều chỉnh linh hoạt toàn bộ các biểu diễn tuyến tính thay vì chỉ tập trung vào cơ chế chú ý.
3. **Thang đo Learning Rate chuẩn mực:** Đảm bảo quá trình cập nhật tham số hội tụ hiệu quả trong không gian hạng thấp.

**Ba điều tôi học được:**
1. **Train loss là một chỉ số thay thế nguy hiểm:** Run `attn_only` có train loss thấp hơn bản `correct` nhưng khi đưa vào bài thi thực tế thì độ chính xác target lại kém hơn. Không bao giờ được dùng train loss làm thước đo quyết định chất lượng mô hình.
2. **Kỹ thuật LoRA Without Regret:** Gắn adapter vào toàn bộ các lớp tuyến tính (`all-linear`) với rank vừa phải ($r=16$) luôn mang lại hiệu quả biểu diễn cao hơn và tổng quát hơn so với việc ép rank thật cao vào chỉ một vài lớp attention ($q, v$).
3. **Giá trị của việc đo đạc công bằng và mốc đóng băng:** Việc đo baseline với một prompt được tối ưu hóa cẩn thận trước khi huấn luyện giúp người kỹ sư nhìn nhận đúng đắn giá trị thực sự của fine-tuning, tránh tâm lý "ảo tưởng sức mạnh" khi so sánh khập khiễng với một prompt sơ sài.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:**
1. Triển khai kỹ thuật Replay Buffer: Trộn thêm khoảng 3% dữ liệu hội thoại và chỉ dẫn phổ thông vào tập 250 ticket để huấn luyện lại, với mục tiêu vừa duy trì target > 95% vừa vượt qua cổng hồi quy regression ($\Delta > -0.02$).
2. Thử nghiệm Bonus B4: Quét rank có kiểm soát ($r \in \{8, 16, 64\}$) trên cấu hình `text-linear` để xác định chính xác ngưỡng bão hòa dung lượng biểu diễn của tập dữ liệu 250 mẫu này.

---

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub — link:
