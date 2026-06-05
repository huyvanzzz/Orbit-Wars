# Prompt cải tiến Orbit Wars Agent

Bạn là AI coding agent. Hãy đọc kỹ `main.py` đang cải tiến, và `orbit_wars_benchmark.ipynb` để xem kết quả của các agent trước khi sửa code.
Không sửa, đọc các file agent khác. Chỉ sửa file agent đang cải tiến.

## Quy trình tự động bắt buộc

Agent phải tự lặp:

đọc code → phân tích điểm yếu → sửa `main.py` → chạy `orbit_wars_benchmark.ipynb` → đọc file `replays/last_match.html` → phân tích vì sao thắng/thua → chỉnh tiếp → benchmark lại.

Không được sửa một lần rồi dừng. Nếu yếu hơn, phải tìm nguyên nhân và cải tiến lại logic, không chỉ rollback.

## Mục tiêu

Mục tiêu là đọc code, phân tích, hiểu điểm yếu thật sự của agent, rồi thiết kế cải tiến một cách có kiểm chứng.

Agent cần tự phân tích xem hướng nào phù hợp với code hiện tại, hướng nào có thể gây mất ổn định, hướng nào nên làm trước, hướng nào chỉ nên dùng như fallback hoặc layer phụ.


Mỗi cải tiến phải được thiết kế theo kiểu khoa học:

- có giả thuyết rõ ràng: cải tiến này giải quyết vấn đề gì
- có vùng code cần tác động
- có cách tích hợp không phá logic cũ
- có fallback khi confidence thấp
- có benchmark so với các agent khác
- có phân tích nếu kết quả yếu hơn

Mục tiêu cuối cùng là làm agent mạnh hơn và ổn định hơn tất cả các agent trong benchmark, không phải làm code phức tạp hơn hoặc “đúng lý thuyết” hơn nhưng chơi yếu hơn.

---

## Benchmark và debug

Sau mỗi nhóm thay đổi, chạy `orbit_wars_benchmark.ipynb` kiểm tra kết quả.

Nếu yếu hơn, phải phân tích nguyên nhân từ file `replays/last_match.html` rồi chỉnh tiếp, ví dụ:
- bắn ít hơn?
- giữ quân quá nhiều?
- angle đoán sai?
- project state tính quá cao số ship cần?
- combined attack lấy quân quá nhiều?
- early game quá thụ động?

Không được chỉ rollback rồi dừng. Hãy chỉnh ngưỡng, thêm fallback, giới hạn phase sử dụng, hoặc giảm mức can thiệp để cải tiến ổn định hơn.

## Output cần trả

- Tóm tắt vấn đề tìm thấy.
- Những hàm/vùng code đã sửa.
- Kết quả so với các agent khác.
- Nếu có bước yếu hơn, giải thích đã chỉnh lại thế nào.
- Giải thích các version đã làm vào trong file `orbit-wars.md`.
- Viết ngắn gọn, logic.

## Điều kiện dừng:
- Khi kết quả của agent file main.py đạt kết quả trên 5/10 match, chạy ipynb trên 20 match khác và chạy lại nếu kết quả 10/20 trở lên thì mới được dừng. Không thì phải lặp lại như trên để cải tiến tiếp.