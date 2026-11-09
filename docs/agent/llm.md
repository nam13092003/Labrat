# LLM qua OpenRouter

> Tài liệu chi tiết cho agent Labrat. Luật chung và thứ tự ưu tiên nằm trong AGENT.md; nếu có mâu thuẫn, AGENT.md thắng.


Bạn được dùng key OpenRouter của người dùng để gọi BẤT KỲ model nào, theo hai cách:

## 1. LLM là MỘT PHƯƠNG PHÁP THỰC NGHIỆM (`llm batch`)
Khi bài toán có thể hưởng lợi, hãy cân nhắc các hướng sau như mọi giả thuyết khác trong `plan.md` (xếp theo lợi ích ÷ chi phí):

| Hướng | Làm gì | Kiểm chứng |
|---|---|---|
| **Zero-shot / few-shot** | LLM dự đoán trực tiếp trên tập VALIDATION | tính đúng metric cuộc thi → `labrat record` để so với mô hình huấn luyện |
| **Tăng cường dữ liệu** | paraphrase, sinh mẫu tổng hợp cho lớp hiếm | A/B: huấn luyện có vs không có dữ liệu tổng hợp (2 GPU song song) |
| **Dịch dữ liệu** | dịch sang ngôn ngữ mô hình mạnh hơn, hoặc dịch ngược để tăng cường | so CV trước/sau |
| **Gán nhãn giả / làm sạch nhãn** | LLM gán nhãn dữ liệu chưa nhãn, phát hiện nhãn sai | đo độ khớp với nhãn thật trên một tập con có nhãn TRƯỚC |
| **Đặc trưng / giải thích** | LLM sinh đặc trưng văn bản, rationale để chưng cất | ablation |

Quy trình chuẩn:
```
# 1) chuẩn bị file dữ liệu .jsonl/.csv có cột id (ví dụ val.jsonl) + template prompt dùng {tên_cột}
python -m labrat llm models --search "<họ model>" --max-price 1          # chọn model, xem giá
python -m labrat llm batch val.jsonl --model <id> --template-file tpl.txt --out competitions/<slug>/llm/zs_val.jsonl \
       --limit 20 --purpose "zero-shot thử"                              # 2) CHẠY THỬ 20 dòng, tự đọc kết quả
python -m labrat llm batch val.jsonl --model <id> --template-file tpl.txt --out ...  --dry-run   # 3) ước tính toàn bộ
python -m labrat llm batch val.jsonl --model <id> --template-file tpl.txt --out ... \
       [--examples shots.jsonl] [--json] --purpose "..."                 # 4) chạy hết (tự tiếp tục nếu dừng)
python -m labrat record <slug> --name "zero-shot <model>" --hypothesis "..." --cv <điểm> --method zero-shot \
       --cost-usd <chi phí> --artifact competitions/<slug>/llm/zs_val.jsonl                  # 5) ghi vào sổ
```
- **Ngân sách:** batch ước tính > $2 (`batch_auto_usd`) sẽ chạy thử vài dòng, ngoại suy chi phí rồi DỪNG.
  Khi đó nhắn người dùng: model, số dòng, chi phí dự kiến, lợi ích kỳ vọng. Chỉ chạy lại với
  `--approved --budget-usd <số người dùng duyệt>` sau khi họ đồng ý. Không tự ý đặt `--approved`.
- **Chống rò rỉ:** ví dụ few-shot chỉ lấy từ TRAIN, không bao giờ từ val/test. Không đưa nhãn val/test vào prompt.
  Dữ liệu tổng hợp chỉ sinh từ train, đánh dấu cột `synthetic=1`, không đưa vào tập val.
- **Tái lập:** mặc định `temperature 0`, `seed 42`. File `<out>.meta.json` lưu model, template, tham số; cache
  `.cache/llm/` giúp chạy lại không tốn tiền.
- **Đọc kết quả:** `<out>.jsonl` có `id`, `output`, `parsed` (khi `--json`), `error`, `truncated`. Dòng lỗi được thử lại
  khi chạy lại cùng lệnh. Với gán nhãn, dùng `--json` và yêu cầu model trả về `{"label": ...}`.
- **Đưa vào huấn luyện:** chuyển kết quả thành csv trong `data/`, rồi `labrat dataset ...` (tải lên mọi tài khoản),
  gắn vào kernel. LLM chạy trên VPS; KHÔNG gọi API từ trong kernel Kaggle, KHÔNG đưa key vào kernel.
- **Luật cuộc thi:** kiểm tra `problem_card.md`. Dữ liệu do LLM sinh thường bị coi là dữ liệu ngoài; một số cuộc thi cấm
  LLM API hoặc model đóng. Nếu luật cấm hay không rõ, chỉ dùng cho nghiên cứu nội bộ và ghi rõ trong báo cáo.

## 2. Ảnh, PDF scan: dùng khả năng CÓ SẴN trước, `llm ask` chỉ là dự phòng
- **Ảnh / biểu đồ** (loss.png, hình mẫu dữ liệu…): đọc trực tiếp bằng công cụ Read của bạn. Model của harness
  (`deepseek-flash`) nhìn được ảnh. Không cần OpenRouter.
- **PDF scan / bảng, công thức trích kém** (`labrat paper` báo `warning`): endpoint của DeepSeek không nhận file PDF,
  nên đổi trang cần đọc thành ảnh rồi đọc bằng Read:
  `pdftoppm -r 110 -png -f <trang> -l <trang> paper.pdf papers/<pid>/page` (gói `poppler-utils`), chỉ render các trang cần,
  không render cả paper.
- **`llm ask` chỉ dùng khi:** harness không có thị giác (ví dụ người dùng đổi sang `deepseek-v4-pro`: ảnh thành
  `[Unsupported Image]`), hoặc khi người dùng yêu cầu xin ý kiến model khác. Không tự ý gọi model khác để "hỏi cho chắc".
```
python -m labrat llm ask "<câu hỏi>" --model <id> --file <ảnh|pdf> --purpose "<lý do>"
```

## Chung
- Chọn **model rẻ nhất đủ dùng**; với batch, so 2–3 model trên 20 dòng trước khi chạy hết.
- Trần chi phí: `labrat llm spend`. Bị chặn → chọn model rẻ hơn hoặc hỏi người dùng.
- Ghi chi phí API vào `report_notes.json` (insights) và cột `--cost-usd` của `labrat record`.
- Riêng tư: mặc định chỉ định tuyến tới nhà cung cấp **không lưu và không huấn luyện** trên dữ liệu
  (`data_collection: deny`). Được gửi dữ liệu người dùng khi tác vụ cần (dịch, gán nhãn...), nhưng không gửi
  `.env`, token, khoá, `.kaggle_accounts/`.
