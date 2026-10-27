# Labrat 🐀

*Chuột thí nghiệm chạy thí nghiệm không ngừng nghỉ — trợ lý nghiên cứu học máy tự động trên GPU Kaggle.*

Bạn giao việc qua **Telegram**, agent tự làm phần còn lại trên GPU Kaggle (2×T4):

- **Nghiên cứu một cuộc thi Kaggle:** đọc đề, tìm paper và lời giải, lập kế hoạch, thực nghiệm, tự cải thiện tới khi
  hết ngân sách GPU, rồi gửi thẻ báo cáo HTML kèm hướng đi tiếp theo.
- **Nghiên cứu một paper:** phân tích sâu, khả năng tái hiện trên 2×T4, bảng đề xuất thực nghiệm (chạy luôn nếu bạn muốn).
- **Dữ liệu của bạn:** gửi file hoặc link, agent tự đưa lên Kaggle (private) và thực nghiệm.

Agent không bao giờ nộp bài và chỉ đánh giá bằng CV. **Nhiều tài khoản Kaggle** chạy song song: mỗi tài khoản giữ một
kernel GPU, có quota tuần riêng, và agent tự chọn tài khoản đang rảnh.

## Thành phần

| Thành phần | Vai trò |
|---|---|
| `install.py` | Trình cài đặt hỏi đáp: harness, API provider, tài khoản Kaggle, Telegram, systemd, cron |
| `labrat/bridge.py` | Service Telegram (long polling, không mở cổng): nhận tin/file → gọi harness → trả lời |
| `labrat/agent.py` + `harness.yaml` | Gọi harness ở chế độ headless: Claude Code, Codex, OpenCode, DeepSeek Harness, hoặc lệnh tùy chỉnh |
| `AGENT.md` | Sổ tay vận hành, nạp làm system prompt (Claude Code) hoặc `AGENTS.md` (Codex/OpenCode) |
| `python -m labrat …` | Bộ lệnh agent gọi: dự án, tài khoản, nghiên cứu, paper, dữ liệu, thực nghiệm, báo cáo |
| `templates/runner.py` | Chạy trong kernel: chia việc cho 2 GPU (`parallel` / `ddp`), đo % sử dụng GPU |
| `scripts/tick_all.py` | Cron 15 phút: thu kết quả kernel, đánh thức agent khi có sự kiện hoặc còn GPU rảnh |

## Cài đặt trên VPS (Ubuntu 22.04/24.04, user thường)

```bash
sudo apt update && sudo apt install -y python3-venv python3-pip git unzip curl
git clone https://github.com/nam13092003/Labrat.git && cd Labrat
python3 install.py
```

Cập nhật sau này: `git pull`. Các file cấu hình, dữ liệu và kết quả (`.env`, `harness.yaml`, `competitions/`,
`dropbox/`, `inbox/`) nằm trong `.gitignore` nên không bị ghi đè. Sau đó chạy
`.venv/bin/pip install -r requirements.txt && sudo systemctl restart labrat-bridge`.

Trình cài đặt hỏi lần lượt:

1. **Harness:** Claude Code, Codex CLI, OpenCode, DeepSeek Harness, hoặc lệnh tùy chỉnh. Nếu cần, trình cài đặt tự
   cài Node.js và gói npm vào `~/.local`, không cần quyền root cho npm.
2. **API provider** theo harness:
   - **Claude Code:** Anthropic API key · DeepSeek (endpoint Anthropic-compatible) · gateway khác · đăng nhập tài khoản Claude.
   - **Codex:** OpenAI API key · đăng nhập ChatGPT · provider OpenAI-compatible (DeepSeek, OpenRouter…, ghi vào `~/.codex/config.toml`).
   - **OpenCode:** provider bất kỳ qua biến `<PROVIDER>_API_KEY` và model dạng `provider/model`.
   - **DeepSeek Harness:** `DEEPSEEK_API_KEY`.
3. **Tài khoản Kaggle:** số lượng, username và token của từng tài khoản.
4. **Telegram:** bot token. Chat id được tự tìm sau khi bạn nhắn bot một tin.
5. **Dịch vụ:** cài service `labrat-bridge` (systemd) và cron 15 phút, rồi kiểm tra đăng nhập Kaggle, gửi tin Telegram
   thử, và chạy một lượt harness thử.

Đổi harness hoặc provider sau này:

```bash
python3 install.py --only harness && sudo systemctl restart labrat-bridge
```

Cờ dòng lệnh của các harness thay đổi theo phiên bản. Nếu `python -m labrat.agent --test` báo lỗi, sửa `cmd` hoặc
`first_cmd` trong `harness.yaml`, và xem lệnh đã sinh bằng `.venv/bin/python -m labrat.agent --show-cmd`.

## Dùng qua Telegram

```
/help  /status  /accounts  /usage  /disk  /clean  /new  /stop <slug>  /resume <slug>  /dropbox  /scan  /get <link>  /cancel

Nghiên cứu cuộc thi https://www.kaggle.com/competitions/playground-series-s6e1, ngân sách 15h GPU
Nghiên cứu cuộc thi https://xacle.org/2027/description.html, ngân sách 15h GPU   (cuộc thi ngoài Kaggle)
Đọc paper 2406.12345 và đề xuất hướng thực nghiệm cho bài phân loại ảnh y tế
[gửi file train.csv] Đây là dữ liệu dự đoán churn, metric AUC, ngân sách 10h
https://huggingface.co/datasets/<org>/<tên>  phân loại cảm xúc, metric F1-macro
Báo cáo dự án playground-series-s6e1
```

## Gửi link dữ liệu qua Telegram

Chỉ cần dán link kèm mô tả bài toán, bridge sẽ **tự tải về** mà không cần chờ agent:

```
Dữ liệu churn, nhãn là cột churn, metric AUC, ngân sách 10h
https://drive.google.com/file/d/1AbC.../view?usp=sharing
tên: churn-2026
```

| Nguồn | Ghi chú |
|---|---|
| Google Drive (file / thư mục) | Phải chia sẻ ở chế độ "Bất kỳ ai có đường liên kết". Thư mục tối đa 50 file, nhiều hơn thì nén thành zip |
| Google Sheets | Tải về dạng CSV, lấy sheet theo `gid` trong link |
| Hugging Face | Tải cả repo dataset hoặc một file. Dataset gated cần `HF_TOKEN` trong `.env` |
| Kaggle dataset | Dùng tài khoản Kaggle đầu tiên |
| Dropbox, GitHub, link trực tiếp | Link Dropbox được tự đổi sang `dl=1`. Link GitHub repo tải về dạng zip, link blob tải file gốc |

- **Tiến độ:** hiện ngay trong một tin nhắn (dung lượng, tốc độ). Muốn dừng thì nhắn `/cancel`.
- **Sau khi tải xong:** tự giải nén zip/tar, phần chữ của tin nhắn được ghi thành `task.md`, rồi dữ liệu được giao
  cho agent.
- **Giới hạn:** mặc định 50 GB mỗi lần (`downloads.max_gb`). Trước khi tải còn kiểm tra ổ trống.
- **Link không phải dữ liệu:** trang web (vd trang mô tả cuộc thi), link cuộc thi Kaggle, arXiv và PDF được chuyển
  cho agent như tin nhắn thường. Agent đọc trang web bằng `labrat page`, tự đi tiếp sang các trang con (Data, Rules...)
  và gom link dữ liệu tìm thấy. Chỉ link rõ ràng là file (đuôi `.csv`, `.zip`, `.parquet`…, hoặc Drive/HF/Kaggle dataset)
  mới được tự tải.
- Có thể dùng lệnh rõ ràng: `/get <link> [tên-thư-mục]`.

## Thư mục thả dữ liệu (dropbox)

Dữ liệu lớn không cần đi qua Telegram. Hãy chép vào `dropbox/` trên VPS, **mỗi bộ dữ liệu là một thư mục con**, và
nên kèm một file `task.md` mô tả bài toán (xem mẫu `dropbox/_mau-task.md`):

```
dropbox/
  churn-2026/
    train.csv
    test.csv
    task.md        # mục tiêu, cột nhãn, metric, ngân sách GPU
  paper.pdf        # PDF ở gốc → agent nghiên cứu paper
```

Từ Windows:

```powershell
scp -r .\churn-2026 labrat@<IP>:~/Labrat/dropbox/
```

Nếu muốn kéo thả, dùng WinSCP (giao thức SFTP, user `labrat`) và mở thư mục `Labrat/dropbox`.

Sau khi chép xong, cron sẽ phát hiện trong vòng 15–30 phút. Muốn xử lý ngay thì nhắn `/scan`.

- **Nhận dữ liệu:** hệ thống chỉ nhận khi file không còn thay đổi, nên dữ liệu đang chép dở chưa bị xử lý. File
  `.part`/`.tmp` bị bỏ qua.
- **Thông báo:** bạn nhận tin Telegram "📂 Dữ liệu mới", và agent bắt đầu làm việc (hỏi lại nếu thiếu mô tả).
- **Thêm file sau này:** được báo lại như một bản cập nhật. Tên thư mục trùng tên dự án có sẵn thì được coi là dữ liệu
  bổ sung cho dự án đó.
- **Dữ liệu gốc:** không bị di chuyển hay xoá, agent chỉ đọc hoặc tạo symlink.
- **Xem trạng thái:** nhắn `/dropbox`.
- **Đổi vị trí:** sửa `paths.dropbox` trong `config.yaml`, ví dụ `/home/labrat/labrat-dropbox`. Thư mục nằm ngoài project
  thì khi cập nhật code không phải lo giữ lại.

Cần nhớ:

- Với mỗi cuộc thi, **mọi tài khoản Kaggle phải bấm chấp nhận luật trên web**. Agent sẽ báo tài khoản nào chưa.
- File gửi qua Telegram tối đa 20 MB, do giới hạn của Bot API. Dữ liệu lớn hơn hãy gửi link.

## Cách tận dụng GPU

**Trong một phiên (2×T4).** Quota tính theo thời gian phiên, nên runner giữ cả hai GPU bận:

- `mode: parallel` (template `gbdt_folds`): hàng đợi job, mỗi GPU một slot. Dùng cho fold, A/B, seed.
- `mode: ddp` (template `torch_ddp`): `torchrun --nproc_per_node=2`, AMP fp16, checkpoint, deadline đồng bộ giữa các rank.

**Giữa các tài khoản.** `labrat push` tự chọn tài khoản rảnh còn nhiều quota nhất. Với 3 tài khoản có thể chạy 3 thí
nghiệm đồng thời, tức 6 GPU T4.

`labrat/budget.py` cộng giờ GPU theo từng tài khoản (quota reset 00:00 UTC thứ Bảy). Trần dự án được tính tổng trên mọi
tài khoản, và kernel đang chạy được giữ chỗ theo thời gian ước tính.

## Chi phí LLM và cache

DeepSeek tự động cache **tiền tố** của request, khớp từ đầu. Cache hit được tính giá rẻ hơn nhiều, nên hệ thống được
sắp xếp để giữ phần đầu request ổn định:

- **Một phiên dài:** tin Telegram và sự kiện cron đều nối tiếp vào cùng một phiên harness (`--continue` hoặc
  `resume`). Lượt sau mang nguyên văn hội thoại cũ làm tiền tố.
- **Phần động đặt cuối:** giờ, nguồn và tin mới nằm ở cuối prompt, không bao giờ ở đầu.
- **Claude Code:** tắt git status snapshot, attribution block và auto memory. Đây là các phần thay đổi giữa các lượt
  nằm trong system prompt (`CLAUDE_CODE_DISABLE_GIT_INSTRUCTIONS`, `CLAUDE_CODE_ATTRIBUTION_HEADER=0`,
  `CLAUDE_CODE_DISABLE_AUTO_MEMORY`). Project cũng không khởi tạo git, trừ khi dùng Codex.
- **Harness không nhớ phiên** (DeepSeek Harness, lệnh tùy chỉnh): lịch sử được gửi theo kiểu chỉ nối thêm. Cứ khoảng
  12 lượt mới bắt đầu lại một lần, nên đa số lượt vẫn cache hit.
- **Xem số liệu:** nhắn `/usage` để xem token và tỉ lệ cache hit trong 24h. Số liệu chỉ có với Claude Code; với harness
  khác, xem trên dashboard của provider.

`/new` xoá ngữ cảnh, nên lượt ngay sau đó sẽ cache miss toàn bộ. Chỉ dùng khi thật sự muốn bắt đầu lại.

## Ổ đĩa

Kernel có thể ghi hàng GB vào output (mô hình pretrained, checkpoint). Labrat chặn việc này ở ba lớp:

1. **Trong kernel:** cuối phiên, file > 200 MB bị chuyển khỏi `/kaggle/working`, nên Kaggle không lưu thành output.
   Muốn giữ một file thì khai báo `keep_outputs` trong `job.yaml`.
2. **Khi tải về VPS:** chỉ tải metrics, log, csv, ảnh và OOF (`kaggle kernels output --file-pattern`), sau đó xoá
   file > 50 MB.
3. **Cron:** ổ còn dưới 5 GB thì tự chạy `labrat clean`. Nếu vẫn thiếu, bot cảnh báo qua Telegram.

Mỗi lệnh `kaggle` chạy với thư mục tạm riêng, xoá ngay sau khi xong, nên file nén dataset hàng GB không còn bị bỏ lại
trong `/tmp`.

Lệnh hữu ích: `/disk` và `/clean` trên Telegram; `labrat disk` và `labrat clean --dry-run` trên VPS.

## Vận hành

```bash
journalctl -u labrat-bridge -f          # log bridge + agent
tail -f ~/labrat-tick.log               # log cron
.venv/bin/python -m labrat accounts --check
.venv/bin/python scripts/selftest.py # kiểm tra offline
.venv/bin/python -m labrat.bridge --diagnose   # bot không trả lời? (dừng service trước để kiểm tra đầy đủ)
```

- **Bí mật:** nằm trong `.env` (chmod 600) và `.kaggle_accounts/`, cả hai đều trong `.gitignore`. Harness chạy với
  quyền của user VPS và đọc được `.env`, nên hãy dùng một user riêng cho agent.
- **Cập nhật code:** `git pull && .venv/bin/pip install -r requirements.txt && sudo systemctl restart labrat-bridge`.
