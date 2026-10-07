# Reflection — Lab 21

*Ngắn gọn, thành thật. Phần này chấm theo độ cụ thể, không theo độ dài.*

**1. Điều gì làm bạn ngạc nhiên nhất?**  
Điều làm tôi ngạc nhiên nhất là hiện tượng train loss "đánh lừa" ở NB4: cấu hình `attn_only` dù chỉ gắn vào 2 lớp attention nhưng khi nâng rank lên 283 lại đạt train loss thấp hơn hẳn bản chuẩn `correct` (0.5369 so với 0.6256), nhưng khi bước vào bài thi thực tế trên tập target thì độ chính xác lại thấp hơn (0.970 so với 0.975). Nó chứng minh trực quan rằng loss thấp không đồng nghĩa với mô hình giỏi hơn.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**  
Tôi mất nhiều thời gian nhất ở khâu chờ Colab huấn luyện và đánh giá nối tiếp 4 run ở NB4 và NB5 (~80 phút). Ban đầu tôi tưởng thời gian lâu nhất sẽ nằm ở khâu nạp dữ liệu hay chỉnh template, nhưng thực tế việc đánh giá sinh văn bản greedy trên toàn bộ 50 mẫu test cho 3 baseline và 4 mô hình mới là phần ngốn tài nguyên và thời gian thực tế nhiều nhất.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**  
Trước lab này, tôi từng tin rằng: "Muốn mô hình mạnh hơn thì cứ tăng rank LoRA thật to và chỉ cần gắn vào các lớp attention là đủ, vừa nhẹ vừa nhanh". Sau khi trực tiếp đo đạc bài lab, tôi nhận ra việc phủ adapter lên toàn bộ các lớp tuyến tính (`all-linear`) với rank vừa phải ($r=16$) mang lại hiệu quả biểu diễn vượt trội hơn nhiều so với việc ép rank cực lớn vào một vài lớp cục bộ.

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**  
Tôi dùng AI assistant để giải thích luồng hoạt động của từng notebook, hỗ trợ thiết lập môi trường chạy trên Google Colab và phân tích nguyên nhân các lỗi cú pháp/mã hóa font Windows. Chỗ AI dễ nhầm lẫn nhất là khi tư vấn lệnh clone code trên Colab, nếu không chỉ định rõ tên thư mục đích thì script sẽ clone theo tên repo dài trên GitHub và làm gãy đường dẫn `os.chdir` ở các dòng lệnh tiếp theo.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**  
Bước đầu tiên tôi làm sẽ KHÔNG PHẢI là mở GPU lên train ngay, mà là **đóng băng một bộ dữ liệu đánh giá và đo trước baseline bằng một prompt được tối ưu hóa cẩn thận (few-shot prompt)**. Nếu prompt tối ưu đã đáp ứng được yêu cầu nghiệp vụ với chi phí và độ trễ chấp nhận được, tôi sẽ tư vấn khách hàng dùng prompt engineering thay vì tốn kém chi phí huấn luyện và bảo trì adapter fine-tune.
