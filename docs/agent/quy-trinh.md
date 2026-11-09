# Quy trình A2 / B / C

> Tài liệu chi tiết cho agent Labrat. Luật chung và thứ tự ưu tiên nằm trong AGENT.md; nếu có mâu thuẫn, AGENT.md thắng.

## Quy trình A2 — Cuộc thi NGOÀI Kaggle (trang web riêng)

Người dùng gửi link trang cuộc thi (vd `https://<site-cuộc-thi>/description.html`). Bridge KHÔNG tải link trang web;
bạn tự đọc:
```
python -m labrat init <slug> --kind custom --metric "<metric>" --direction ... --title "<tên cuộc thi>" --gpu-hours <N>
python -m labrat page <url-trang-mô-tả> --slug <slug> --follow 1     # đọc trang + các trang con cùng site
```
- Đọc mọi file trong `competitions/<slug>/pages/*.md` (mô tả, dữ liệu, đánh giá, luật, mốc thời gian) → điền
  `problem_card.md`: nhiệm vụ, **metric chính xác như ban tổ chức tính**, baseline của ban tổ chức (điểm để so sánh),
  luật về dữ liệu ngoài/pretrained, deadline.
- `pages/links.json` liệt kê link dữ liệu tìm thấy. Tải bằng `python -m labrat get <url> --name <slug>-data`
  (Drive/HF/zip...). Dữ liệu cần đăng ký / đăng nhập → nhắn người dùng gửi link tải hoặc thả vào dropbox.
- Nếu ban tổ chức có repo baseline (GitHub) → `labrat get <repo-url>` rồi đọc code: đó là mốc CV đầu tiên cần tái hiện.
- Đưa dữ liệu lên Kaggle (`labrat dataset … --name <slug>-data`) rồi thực nghiệm như Quy trình A. Không có leaderboard
  tự động: so với baseline công bố của ban tổ chức, dùng đúng metric của họ.
- Trang quá ít chữ (dựng bằng JavaScript) → `warnings` sẽ báo; khi đó nhờ người dùng gửi nội dung hoặc PDF.

## Quy trình B — Nghiên cứu một paper

`prior_work.md` (bắt buộc trước thí nghiệm thứ 2 ở mọi quy trình): ở đây là chính paper đang nghiên cứu, các
phương pháp nó so sánh và các công trình mạnh hơn ra sau nó.

```
python -m labrat init <slug> --kind custom --title "<tên>"   # nếu chưa có dự án phù hợp
python -m labrat paper <slug> <arxiv-id | url | đường-dẫn-pdf>
```
1. Đọc `papers/<pid>/text.txt` (đọc từng đoạn nếu dài). Nếu `warning` báo trích xuất kém: tìm bản HTML
   (arxiv.org/html/<id>) hoặc code chính thức (GitHub); bảng/hình quan trọng thì render đúng trang đó thành PNG bằng
   `pdftoppm` rồi đọc ảnh (xem docs/agent/llm.md mục "Ảnh, PDF scan").
2. Điền đầy đủ `papers/<pid>/analysis.md` theo khung có sẵn. Phần quan trọng nhất là **mục 6 (khả năng tái hiện
   trên 2×T4)** và **mục 7 (bảng đề xuất thực nghiệm)**: mỗi đề xuất có mục tiêu, thay đổi, cách dùng 2 GPU,
   ước tính giờ GPU, tiêu chí thành công. Ưu tiên: tái hiện cốt lõi thu nhỏ → ablation → áp dụng vào dữ liệu dự án.
3. Có thể tìm thêm paper liên quan (`labrat research`) để so sánh, ghi vào mục 8.
4. `python -m labrat paper-report <slug> <pid> --send` rồi trả lời: 3–5 câu tóm tắt + 3 đề xuất hàng đầu + tổng giờ GPU
   ước tính, và hỏi người dùng có muốn chạy không (trừ khi họ đã bảo chạy luôn).
5. Nếu chạy: mỗi đề xuất thành một thí nghiệm (`new-exp`, giả thuyết ghi rõ "P1: ..."), theo Quy trình A.

## Quy trình C — Dữ liệu người dùng gửi

`prior_work.md`: các cách tiếp cận mạnh nhất đã công bố cho loại bài toán / dữ liệu này.

Dữ liệu đến theo ba đường:
- **Thư mục thả dữ liệu (dropbox)** — đường chính cho dữ liệu lớn. Người dùng chép vào `dropbox/<tên>/`
  (scp/WinSCP/rsync). Khi đã chép xong, cron hoặc lệnh `/scan` giao cho bạn một lượt nguồn `dropbox` kèm đường dẫn,
  danh sách file và nội dung `task.md` (nếu có). Tên thư mục trùng dự án có sẵn → dữ liệu bổ sung cho dự án đó.
  PDF đặt ở gốc dropbox → Quy trình B. **Không sửa/xoá/di chuyển dữ liệu trong dropbox**; dùng
  `labrat data-add <slug> <đường-dẫn> --link` (symlink, không tốn thêm dung lượng). `labrat dropbox` liệt kê trạng thái.
- **Link trang web** (không phải file): bridge chuyển nguyên tin nhắn cho bạn → đọc bằng `labrat page`.
- **Link gửi qua Telegram** (Google Drive/Sheets, Hugging Face, Kaggle dataset, Dropbox, GitHub, URL trực tiếp):
  bridge TỰ TẢI vào `dropbox/<tên>/`, giải nén, ghi phần chữ của tin nhắn thành `task.md`, rồi giao bạn một lượt
  nguồn `download` (cùng định dạng như dropbox). Bạn không cần tải lại. Nếu cần tự tải thêm từ một link:
  `labrat get <url> --name <tên>` (vào dropbox) hoặc `labrat data-add <slug> <url>` (vào data/raw của dự án).
  Link cuộc thi Kaggle, arXiv và PDF không bị tải như dữ liệu mà đến với bạn như tin nhắn thường.
- File qua Telegram (≤ 20 MB) được bridge lưu vào `inbox/<ngày>/…` và báo đường dẫn trong tin nhắn.
```
python -m labrat init <slug> --kind custom --metric "<metric>" --direction ... --title "<tên>"
python -m labrat data-add <slug> <đường-dẫn-inbox | link>      # vào data/raw/, tự giải nén
#   khám phá dữ liệu cục bộ (pandas), điền problem_card.md, chuẩn bị thư mục tải lên (vd data/upload/)
python -m labrat dataset <slug> competitions/<slug>/data/upload --name <slug>-data
python -m labrat dataset-status <slug> <slug>-data              # chờ "ready" trên MỌI tài khoản
```
`labrat dataset` tải lên TẤT CẢ tài khoản (dataset private của tài khoản A không đọc được từ kernel của B) và ghi vào
`own_datasets` trong `project.yaml`; bundler tự gắn đúng bản theo tài khoản chạy. Trong kernel, dùng
`find_input("<tên file>")` (template GBDT) để tìm file dưới `/kaggle/input`.
Nếu người dùng không nói metric/mục tiêu → hỏi trước khi thực nghiệm.
