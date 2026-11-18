# CAMO-FS Few-Shot YOLO — Trạm Bẫy Ảnh Sinh Thái Thông Minh

Dự án nhận diện sinh vật ngụy trang ít mẫu (**CAMO-FS**, 47 lớp, `split1`) bằng YOLO, hướng tới triển khai trên **Orange Pi 5 Pro (`OPI-01`)** + **Camera Luxonis OAK-D-S2 (`OAK-01`)**.

---

## 1. Thành phần đóng gói trong Git Repository

Do giới hạn dung lượng của Git, repository này chỉ đóng gói **mã nguồn, cấu hình, báo cáo và 2 mô hình đã chọn** (chấm điểm chuẩn hóa trên `weights/last.pt`):

| Thành phần | Đường dẫn | Mô tả |
| :--- | :--- | :--- |
| **Model chính (Đề xuất)** | `runs/models/v8s_rich_backbone_k5_seed0/weights/last.pt` | **YOLOv8s** (`11,15M` params, `freeze=10`, `mosaic=0.0`, `lr0=0.0005`, `50 epochs`, `val=False`). Đạt **`13.70` Box AP** tại `k5_seed0` (`13.56 ± 0.12` trung bình 3 seeds) trên tập Test (`2.655` ảnh / `3.108` instances). |
| **Model dự phòng nhẹ** | `runs/models/v8n_nomosaic_stable_k5_seed0/weights/last.pt` | **YOLOv8n** (`3,15M` params, `freeze=10`, `mosaic=0.0`, `lr0=0.0005`). Đạt **`8.64` Box AP** tại `k5_seed0` (`8.92 ± 0.31` trung bình 3 seeds), phù hợp cho VPU Myriad X của OAK-D-S2 nếu cần tải nhẹ. |
| **Mã nguồn pipeline** | `code/` | `step0_audit_data.py`, `step1_build_kshot.py`, `step2_screen_models.py`, `step4_cameraless_trap_demo.py`, `test_week1_pipeline.py`. |
| **Báo cáo & Số liệu** | `results/` | `BAO_CAO_TOM_TAT.docx`, `class_overlap.json`, `model_stability_raw.csv`, `model_stability_report.md` *(các file kiểm toán dữ liệu `DATA_CHECK.md`, `camo5_support_5shot_merged.json` nằm kèm trong thư mục `audit/` của Kaggle Dataset)*. |
| **Cấu hình YOLO** | `runs/configs/` | 13 file `.yaml` (`camo_test.yaml` + `camo_k{1,2,3,5}_seed{0,1,2}.yaml`) ánh xạ 47 lớp (`0..46`). |

> **Dữ liệu (`runs/processed_data/`):** Không lưu trực tiếp trên Git (~1,01 GB). Dữ liệu đã tiền xử lý (12 tập $k$-shot lồng nhau tô xám `pixel=114` + tập `test/` 2.655 ảnh) được tách riêng sang Kaggle Dataset: **`nguynthphngtuyn/camo-fs-fewshot-yolo-processed`** ([link Kaggle](https://www.kaggle.com/datasets/nguynthphngtuyn/camo-fs-fewshot-yolo-processed)). *(Lưu ý: mặc định dataset ở chế độ Private — cần thêm thành viên vào mục Collaborators hoặc chuyển sang Public trên Kaggle trước khi thành viên tải).*

---

## 2. Hướng dẫn cài đặt và khôi phục dữ liệu sau khi clone

### Bước 1: Cài đặt thư viện Python
```bash
pip install -r requirements.txt
```
*(Đã kiểm thử trên `Python 3.13.15`, `torch 2.14.1+cu130`, `ultralytics 8.4.210`, `opencv-python 5.0.0`, `numpy 2.5.3`, `pyyaml 6.0.3`, `onnxruntime 1.20.1`).*

### Bước 2: Tải dữ liệu từ Kaggle và đặt đúng thư mục
1. Tải dataset từ [`nguynthphngtuyn/camo-fs-fewshot-yolo-processed`](https://www.kaggle.com/datasets/nguynthphngtuyn/camo-fs-fewshot-yolo-processed) (tải trực tiếp qua Web UI hoặc bằng Python `pip install kagglehub` -> `kagglehub.dataset_download("nguynthphngtuyn/camo-fs-fewshot-yolo-processed")`), rồi copy/giải nén thư mục `processed_data/` vào đúng đường dẫn:
   ```text
   runs/processed_data/
   ├── k1_seed0/ ... k5_seed2/   (12 tập k-shot lồng nhau)
   └── test/                     (2.655 ảnh + nhãn YOLO)
   ```
2. **Cập nhật đường dẫn trong `runs/configs/*.yaml`** cho đúng thư mục trên máy hiện tại (do file YAML gốc lưu đường dẫn `D:/camo_yolo_project/runs/processed_data`):
   ```bash
   python -c "from pathlib import Path; target = (Path('runs/processed_data').resolve().as_posix()); [p.write_text('\n'.join([f'path: {target}' if l.startswith('path:') else l for l in p.read_text(encoding='utf-8').splitlines()]) + '\n', encoding='utf-8') for p in Path('runs/configs').glob('*.yaml')]"
   ```

---

## 3. Kiểm tra nhanh model & xuất định dạng biên (tùy chọn)

- **Chạy kiểm thử đơn vị pipeline (CPU):**
  ```bash
  python code/test_week1_pipeline.py
  ```

- **Đánh giá lại `weights/last.pt` trên tập Test (`2.655` ảnh):**
  ```bash
  yolo val model=runs/models/v8s_rich_backbone_k5_seed0/weights/last.pt data=runs/configs/camo_test.yaml split=test imgsz=640
  ```

- **Chạy Bước 4 (tạo ảnh minh họa & export `ONNX` / `NCNN` khi cần triển khai biên):**
  ```bash
  pip install onnx onnxslim ncnn
  python code/step4_cameraless_trap_demo.py \
    --weights runs/models/v8s_rich_backbone_k5_seed0/weights/last.pt \
    --test-images runs/processed_data/test/images \
    --out-dir results/qualitative_analysis_v8s \
    --max-images 2655 \
    --export-edge
  ```

> **Lưu ý mã nguồn:** Trong `code/step1_build_kshot.py` (dòng 347), chế độ tự dò đường dẫn đang ghi cứng `d:/ifs_workspace/...`; nếu chạy lại Bước 1 trên máy mới từ bộ dữ liệu gốc, truyền tường minh `--support-json results/camo5_support_5shot_merged.json --test-json <path_to_camo5_test_split1.json>`.
