# GPU, tài khoản Kaggle, hợp đồng script

> Tài liệu chi tiết cho agent Labrat. Luật chung và thứ tự ưu tiên nằm trong AGENT.md; nếu có mâu thuẫn, AGENT.md thắng.

## Nhiều tài khoản Kaggle = nhiều thí nghiệm song song

`python -m labrat accounts` cho biết từng tài khoản: quota tuần đã dùng / còn lại, có đang chạy kernel GPU không.

- Mỗi tài khoản chạy **tối đa 1 kernel GPU** cùng lúc và có quota tuần riêng (~30h, reset 00:00 UTC thứ Bảy).
- `labrat push` (không có `--account`) **tự chọn tài khoản rảnh còn nhiều quota nhất**.
- Với N tài khoản, hãy giữ N kernel chạy song song khi có đủ giả thuyết đáng thử:
  chuẩn bị nhiều thí nghiệm độc lập, smoke từng cái, rồi đẩy GPU lần lượt — mỗi lệnh push lấy một tài khoản rảnh.
- Dùng `labrat wait <slug> --any` để quay lại ngay khi MỘT thí nghiệm xong, phân tích, rồi lấp chỗ trống.
- Thí nghiệm song song phải **độc lập** (không chờ kết quả của nhau). Thí nghiệm cần kết quả cái trước
  thì xếp sau.
- Trần ngân sách dự án tính tổng trên mọi tài khoản; kernel đang chạy được giữ chỗ theo ước tính
  (`--est-hours`), nên hãy ước tính sát.
- Với mỗi cuộc thi, MỌI tài khoản phải chấp nhận luật trên web. `labrat fetch <slug>` báo tài khoản nào chưa —
  nhắn người dùng ngay nếu có.

## Tận dụng 2 GPU T4 trong mỗi phiên

Quota tính theo **thời gian phiên**, không theo số GPU. Một phiên T4 x2 trống một GPU = phí nửa ngân sách.

| Tình huống | Cách dùng 2 GPU | `job.yaml` |
|---|---|---|
| GBDT / mô hình nhỏ, K-fold | 2 fold chạy đồng thời | `mode: parallel`, mỗi fold một job |
| A/B hai biến thể | 2 biến thể song song trong 1 phiên | `mode: parallel`, 2 job, không `post` |
| Ablation | bản đầy đủ và bản bỏ một thành phần | `mode: parallel`, 2 job |
| Mạng sâu lớn, 1 mô hình | DDP trên cả 2 GPU | `mode: ddp` |

Phần cứng T4: fp16 (không bf16), không FlashAttention-2 (dùng SDPA), ~15 GB/GPU, ~4 nhân CPU cho cả phiên
(2 job song song → `n_jobs=2`, `num_workers=2` mỗi job). Sau mỗi lần chạy xem `gpu_util`: GPU < 40% là lãng phí;
`n_gpu < 2` → báo người dùng.

## Bố cục thư mục

```
competitions/<slug>/
  project.yaml        kind (competition|custom), metric, chiều tốt, ngân sách, datasets, own_datasets, paused
  problem_card.md     thẻ bài toán          plan.md       hàng đợi giả thuyết
  prior_work.md       lời giải mạnh nhất    eda.md        hiểu dữ liệu
  _archive/           lịch sử trước khi reset (đọc để kế thừa)
  literature.json     tài liệu              registry.json sổ thí nghiệm (máy ghi)
  report_notes.json   nhận định báo cáo     reports/      thẻ HTML
  papers/<pid>/       paper.pdf, text.txt, meta.json, analysis.md, analysis.html
  data/raw/           dữ liệu người dùng (không commit)
  experiments/exp-00X-<tên>/   job.yaml, src/, NOTES.md, out/{smoke,quick,gpu,seed<N>}/
inbox/                file người dùng gửi qua Telegram
dropbox/<tên>/        dữ liệu người dùng thả vào (chỉ đọc với agent), task.md = mô tả bài toán
```

## Hợp đồng với script huấn luyện

- Ghi kết quả vào `$LABRAT_METRICS_PATH` (JSON, khóa `cv`, tùy chọn `cv_std`).
- File kết quả nhỏ (oof, dự đoán, biểu đồ) vào `$LABRAT_OUTPUT_DIR`. File nặng (pretrained, checkpoint trung gian,
  cache đặc trưng) vào `$LABRAT_SCRATCH`. Cuối phiên, runner tự chuyển mọi file > 200 MB ra khỏi output (xem
  `outputs_pruned` trong metrics.json). Cần giữ một file lớn trên Kaggle (vd `best_model.pt` để phân tích lỗi hay tiếp tục huấn luyện) → khai báo
  trong job.yaml: `keep_outputs: ["outputs/best_model.pt"]`.
- VPS chỉ tải về metrics/log/csv/ảnh/oof nhỏ (≤ 50 MB/file). Cần thêm file → job.yaml `fetch_outputs: "<regex>"`.
- `time.time() > $LABRAT_DEADLINE` → lưu checkpoint và thoát.
- `LABRAT_SMOKE=1` → dữ liệu nhỏ, vài bước. `LABRAT_QUICK=1` → bản rẻ (nếu không dùng `quick_args`).
- Seed khởi tạo mô hình / thứ tự dữ liệu đọc từ `LABRAT_SEED` (mặc định 42). Cách chia fold KHÔNG theo seed này —
  fold cố định giữa mọi thí nghiệm để so sánh cặp được.
- `mode: parallel`: mỗi job thấy đúng 1 GPU (`CUDA_VISIBLE_DEVICES`) → luôn dùng `cuda:0`.
