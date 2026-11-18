# CAMO-FS Few-Shot YOLO Pilot (Tuần 1 — Chưa cần Camera vật lý)

Thư mục này chứa trọn bộ mã nguồn thực thi **Tuần 1** của đồ án: kiểm tra dữ liệu CAMO-FS, sinh tập $k$-shot lồng nhau có tô xám `pixel = 114` chống đè vùng giao, và **khảo sát chọn mô hình YOLO ổn định nhất có thể triển khai lên phần cứng (`Orange Pi 5 Pro` + `OAK-D-S2`)**.

## 1. Danh sách các file trong `code/`

| File | Chức năng |
| :--- | :--- |
| `step0_audit_data.py` | **Bước 0:** Tự động quét thư mục dữ liệu CAMO-FS, đếm số lớp/ảnh/instance, truy vết nguyên nhân lệch 6 ảnh (197 + 2.655 = 2.852 vs 2.858), đối chiếu 47 lớp với 80 lớp COCO và xuất `DATA_CHECK.md` + `class_overlap.json`. |
| `step1_build_kshot.py` | **Bước 1:** Tự động phát hiện file Support JSON (235 instances) & Test JSON (3.107 instances), tạo các tập $k$-shot lồng nhau ($1 \subset 2 \subset 3 \subset 5$) qua các seeds, tô xám `114` an toàn (2 pha) và sinh file cấu hình `.yaml`. |
| `step2_screen_models.py` | **Bước 2 & 3:** Chạy khảo sát độ hiệu quả & độ ổn định (`Mean ± Std` trên `last.pt`) của 5 cấu hình YOLO triển khai được trên phần cứng (`v8n_baseline`, `v8n_nomosaic_stable`, `v8s_rich_backbone`, `yolo11n_spatial_attn`, `yolo11s_spatial_attn`), tách điểm **All (47 lớp)**, **Pure Novel**, và **COCO-Overlap**. |
| `step4_cameraless_trap_demo.py` | **Bước 4 (Tuần 1):** Mô phỏng "Trạm Bẫy Ảnh Sinh Thái Thông Minh" trên tập ảnh/video có sẵn khi chưa cắm camera, tự động xuất 5 ảnh dự đoán tốt nhất (`top5_good`) + 5 ảnh lỗi (`top5_bad`) và kiểm tra xuất mô hình sang **ONNX / NCNN** cho `Orange Pi 5 Pro`. |
| `test_week1_pipeline.py` | Bộ kiểm thử tự động trên CPU (Unit Test) xác nhận logic Bước 0 và Bước 1. |

---

## 2. Lệnh chạy tuần tự (Ngắn gọn, chạy trực tiếp)

Giả sử dữ liệu CAMO-FS tải từ Kaggle (`danhnt/camo-fs-dataset`) được giải nén vào `<CAMO_DATA_ROOT>`:

```bash
# 0. Kiểm tra dữ liệu & tách danh sách lớp trùng COCO
python step0_audit_data.py --data-root <CAMO_DATA_ROOT> --out-md ../results/DATA_CHECK.md --out-json ../results/class_overlap.json

# 1. Sinh tập k-shot lồng nhau (k = 1, 2, 3, 5; seed = 0, 1, 2) & tô xám 114 an toàn
python step1_build_kshot.py --data-root <CAMO_DATA_ROOT> --out-dir ../runs/processed_data --config-dir ../runs/configs

# 2. Sàng lọc mô hình YOLO ổn định nhất (Chạy nhanh k=1 và k=5 trên 3 seeds trước)
python step2_screen_models.py --config-dir ../runs/configs --runs-dir ../runs/models --results-dir ../results --overlap-json ../results/class_overlap.json --k-shots 1 5 --seeds 0 1 2

# 3. Mô phỏng Bẫy ảnh (chưa cần camera) + Lọc 5 ảnh tốt / 5 ảnh sai + Kiểm tra xuất ONNX/NCNN cho Orange Pi 5 Pro
python step4_cameraless_trap_demo.py --weights ../runs/models/yolo11n_spatial_attn_k5_seed0/weights/last.pt --test-images ../runs/processed_data/test/images --out-dir ../results/qualitative_analysis --export-edge
```
