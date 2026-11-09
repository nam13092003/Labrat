# Phương pháp nghiên cứu — chi tiết

> Tài liệu chi tiết cho agent Labrat. Luật chung và thứ tự ưu tiên nằm trong AGENT.md; nếu có mâu thuẫn, AGENT.md thắng.

## Vòng nghiên cứu
1. **Hiểu dữ liệu trước khi mô hình** (`competitions/<slug>/eda.md`, viết bằng script CPU trên VPS hoặc kernel CPU):
   phân phối nhãn và đặc trưng, giá trị thiếu, nhiễu/mâu thuẫn nhãn (mẫu trùng khác nhãn, độ đồng thuận người gán),
   cấu trúc nhóm/thời gian/người dùng (→ chọn validation), lệch train/test (adversarial validation: AUC > 0.6 là có
   lệch — tìm đặc trưng gây lệch), rò rỉ (leakage), độ dài / độ khó / miền con, mất cân bằng.
2. **Phân tích lỗi sau MỖI thí nghiệm** (lưu OOF để làm được): lỗi theo lát cắt (độ dài, lớp, miền, nguồn, độ khó),
   các mẫu sai nặng nhất và điểm chung của chúng, sai số hệ thống (lệch về giữa, bão hoà, nhầm cặp lớp), so sánh OOF
   giữa thí nghiệm mới và cha (cải thiện ở đâu, tệ đi ở đâu). Ghi vào NOTES.md phần "Phân tích & bài học".
3. **Mỗi giả thuyết có đủ 5 phần** (ghi trong `plan.md` và NOTES.md):
   *căn cứ* (quan sát cụ thể, có số liệu, từ bước 1–2 hoặc paper) → *cơ chế* (vì sao thay đổi này tác động đúng chỗ
   đó) → *dự đoán* (nếu đúng thì lát cắt nào cải thiện, CV đổi khoảng bao nhiêu) → *kiểm tra* (thí nghiệm, đổi MỘT
   yếu tố — với luồng nghiên cứu) → *chi phí* (giờ GPU). Căn cứ truyền vào `labrat new-exp --evidence "..."`.
4. **Kết luận cả khi thất bại**: dự đoán sai thì cơ chế nào bị bác bỏ, rút ra điều gì về dữ liệu. Ghi vào insights.

## Nguồn ý tưởng cho luồng nghiên cứu

Mỗi chu kỳ lập kế hoạch có ít nhất MỘT ý tưởng sáng tạo (không phải công thức sẵn):
- *Đặt lại bài toán*: hàm mất mát / đầu ra khớp với CÁCH metric của bài chấm điểm (metric đo thứ tự → học thứ tự;
  metric có ngưỡng → tối ưu ngưỡng; metric bất đối xứng → loss bất đối xứng...), biến đổi nhãn cho hợp phân phối,
  dùng thông tin nhãn phong phú hơn nếu có (nhiều người gán, nhãn phụ → multi-task), chia bài toán thành các bước.
  Các ví dụ ở đây chỉ minh hoạ; ý tưởng cụ thể phải xuất phát từ dữ liệu và metric của chính bài đang làm.
- *Thiên kiến quy nạp của mô hình*: chọn kiến trúc khớp cấu trúc dữ liệu (tuần tự, tập hợp, đồ thị, đa phương thức,
  thứ bậc), cách trộn modality (late/early/cross-attention), pooling, độ phân giải đầu vào, pretrained phù hợp miền.
- *Dữ liệu*: augmentation dựa trên **bất biến thật** của bài toán (không đổi nhãn), xử lý nhiễu nhãn
  (loss bền vững, label smoothing, lọc theo đồng thuận), sửa lệch phân phối (trọng số mẫu, chuẩn hoá theo nhóm),
  đặc trưng miền từ hiểu biết chuyên môn, curriculum theo độ khó, pseudo-label có kiểm soát.
- *Từ paper / lời giải top*: lấy **cơ chế** rồi giải thích vì sao áp vào được dữ liệu này, không chép nguyên công thức.
- *Ablation*: khi một hướng tốt lên, bỏ từng thành phần để biết phần nào thực sự đóng góp.

Dùng 2 GPU cho **2 giả thuyết khác nhau** (A/B) hoặc các fold — không bao giờ cho 2 seed.

## Ba mức chạy
| Mức | Lệnh | Dùng khi | So sánh với |
|---|---|---|---|
| smoke (CPU) | `push --smoke` | kiểm tra code chạy được | — |
| bản rẻ (GPU, tuỳ chọn) | `push --quick` | ý tưởng tốn GPU (bản đầy đủ > ~3h) cần sàng lọc: ít fold (`quick_max_jobs`) / ít epoch (`quick_args`), ≤ `quick_max_hours` | `quick_cv` của thí nghiệm cha (chạy cha bằng `--quick` một lần nếu chưa có) |
| đầy đủ (GPU) | `push` | mặc định | `cv` của cha |

Bản rẻ phải giữ đúng thứ hạng giữa các ý tưởng (cùng fold, cùng dữ liệu con); nếu một ý tưởng tốt ở bản rẻ mà kém ở
bản đầy đủ, ghi lại — đó là thông tin về bài toán (vd ý tưởng chỉ có ích khi huấn luyện ngắn).

## Chốt kết quả

`labrat report-seeds <slug> <exp> [--n 3]`: chỉ cho thí nghiệm đã chạy đầy đủ, code chưa đổi
sau lần chạy đó (code chặn nếu đổi), tối đa `research.max_report_experiments` thí nghiệm: phương pháp cuối, lời giải
nền mạnh nhất đã tái hiện, và các ablation chính. Mỗi seed một kernel trên tài khoản rảnh; gọi lại để đẩy nốt seed
còn chờ. Kết quả `report` = mean ± std trên mọi seed, đưa vào bảng so sánh. So sánh phương pháp với nền theo
mean ± std, không theo một lần chạy.

Bản rẻ chỉ chạy 2 fold nên dao động lớn: coi chênh lệch < 2×`quick_cv_std` là "chưa rõ" chứ không phải "thất bại" —
ý tưởng có cơ sở mạnh thì chạy đầy đủ luôn.

## Đánh giá cải thiện

Cải thiện là thật khi > `min_delta` và đáng kể so với `cv_std` (so trên cùng các fold). Ý tưởng thất bại vẫn ghi lại
cơ chế bị bác bỏ — đó là dữ liệu cho phần phân tích của paper.

## Kinh nghiệm cũ

Dự án có `_archive/<thời điểm>/` (người dùng đã reset): đọc `registry.json`, các `experiments/*/NOTES.md`,
`prior_work.md`, `report_notes.json` cũ để biết hướng nào đã hiệu quả / thất bại. Dùng làm căn cứ ("archive: …"),
không lặp lại hướng đã thất bại nếu không có lý do mới.
