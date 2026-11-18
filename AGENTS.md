# CAMO-FS Few-Shot YOLO Course Project Instructions

- **Mục tiêu dự án:** Đồ án môn học "Nhận diện sinh vật ngụy trang ít mẫu bằng YOLO" trên bộ dữ liệu CAMO-FS (47 lớp, 197 ảnh Support / 235 instances, 2.655 ảnh Test / 3.108 instances), kết hợp triển khai lên phần cứng biên **Orange Pi 5 Pro (`OPI-01`)** + **Camera Luxonis OAK-D-S2 (`OAK-01`)** theo kịch bản *Trạm Bẫy Ảnh Sinh Thái Thông Minh (Smart Edge Camera Trap)*.
- **Cho phép chạy Local GPU:** Được phép chạy huấn luyện (`train`), đánh giá (`val`), xuất mô hình (`export` ONNX/NCNN/RKNN) và chạy demo trực tiếp trên GPU cục bộ của máy (**NVIDIA GeForce RTX 5060 Laptop GPU**).
- **Quy chuẩn thực nghiệm nghiêm ngặt:**
  1. Không bao giờ chọn `best.pt` theo tập Test; luôn cố định `epochs=50`, tắt `val=False` trong lúc train, và chỉ chấm điểm duy nhất trên `weights/last.pt`.
  2. Giữ nguyên tính chất lồng nhau của tập Support ($1\text{-shot} \subset 2\text{-shot} \subset 3\text{-shot} \subset 5\text{-shot}$) và tô xám 2 pha (`pixel = 114`) cho các instance không được chọn trong cùng bức ảnh.
  3. Luôn tách kết quả đánh giá thành 3 nhóm: **All (47 lớp)**, **COCO-Overlap (8 lớp trùng trực tiếp)**, và **Pure Novel**.
- **Cấu trúc thư mục:**
  - `code/`: Chứa các script Bước 0 (`step0_audit_data.py`), Bước 1 (`step1_build_kshot.py`), Bước 2 & 3 (`step2_screen_models.py`), Bước 4 (`step4_cameraless_trap_demo.py`) và `run_week1_benchmark.bat`.
  - `runs/processed_data/`: Đã sinh sẵn 12 tập $k$-shot lồng nhau (`k={1,2,3,5}`, `seed={0,1,2}`) và tập `test/`.
  - `runs/configs/`: Chứa 13 file cấu hình `.yaml` của YOLO.
  - `runs/models/`: Chứa kết quả huấn luyện và trọng số `last.pt`.
  - `results/`: Chứa báo cáo `DATA_CHECK.md`, `class_overlap.json`, `camo5_support_5shot_merged.json`, và bảng tổng hợp `model_stability_report.md`.
