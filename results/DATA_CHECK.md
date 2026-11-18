# DATA_CHECK.md — Báo Cáo Kiểm Chứng Dữ Liệu CAMO-FS (Bước 0 & Bước 1)

## 0. Tóm tắt Kết luận Kiểm chứng (Executive Summary cho cả nhóm)
1. **Giải mã nguyên nhân lệch 6 ảnh (197 + 2.655 = 2.852 vs 2.858 trong bài báo):**
   - Trong bản phát hành chính thức trên Kaggle (`danhnt/camo-fs-dataset`, version 4), thư mục `images/images/` chứa **đúng 2.852 file ảnh vật lý** (`camouflage_00000000.jpg` đến `camouflage_00002852.jpg`, khuyết 1 chỉ số `00001601`).
   - Tập Support (`split1` gồm 47 file `*_5shot_split1.json`) chứa **đúng 197 ảnh duy nhất** (235 instances).
   - Tập Test (`camo5_test_split1.json`) chứa **đúng 2.655 ảnh duy nhất** (3.108 instances — nhiều hơn 1 instance so với con số 3.107 ước tính).
   - **Không có bất kỳ ảnh nào trùng giữa Support và Test (`197 + 2.655 = 2.852` khớp 100% số file trên ổ cứng).** Con số 2.858 / 3.342 trong bài báo là số trước khi nhóm tác giả lọc bỏ 6 ảnh khi đóng gói bản release `split1`.
2. **Không cần chờ hỏi danh sách Support Set chính thức:**
   - Bộ dữ liệu tải về đã có sẵn thư mục `few-shot-annotations/subsplit/split1/` chứa các tập `1shot`, `2shot`, `3shot`, `5shot` chính thức của tác giả, thỏa mãn tuyệt đối tính chất lồng nhau `1-shot (47) ⊂ 2-shot (94) ⊂ 3-shot (141) ⊂ 5-shot (235)`.
3. **Phát hiện & Xử lý 2 lỗi ngầm cực kỳ quan trọng trong dữ liệu gốc:**
   - **Lỗi trùng `annotation_id` giữa các file lớp:** Trong `subsplit/split1/`, các cặp lớp (`Fish` & `Moth` cùng có `ann_id=826`; `Lizard` & `Snake` cùng có `ann_id=386, 387`) bị trùng ID. Script `step1_build_kshot.py` đã tự động đánh lại ID duy nhất `1..235` để không bị mất 3 instances này khi gộp file.
   - **Box lồng nhau 100% trong ảnh đa vật thể:** Có 38 ảnh Support chứa từ 2 instances trở lên (ví dụ `camouflage_00002787.jpg` của lớp `Camel` có box con nằm trọn bên trong box mẹ). Thuật toán tô xám 2 pha (`mask_unselected_instances_safe`) đã bảo vệ nguyên vẹn vùng pixel của vật thể được chọn.

## 1. Thống kê tổng quan trên ổ cứng & File nhãn
- **Thư mục gốc (`data_root`):** `C:\Users\nguye\.cache\kagglehub\datasets\danhnt\camo-fs-dataset\versions\4`
- **Tổng số file ảnh tìm thấy trên ổ cứng:** `2852` (Số tên file duy nhất: `2852`)
- **Tổng số tên ảnh duy nhất khai báo trong các file JSON:** `2852`
- **Tổng số lớp phát hiện được:** `47` (Mốc chuẩn bài báo: `47`)

### Chi tiết từng file JSON tìm thấy:
| File JSON | Số lớp | Số ảnh khai báo | Số ảnh duy nhất | Số instances (bboxes) | Số ảnh 0 instance |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `few-shot-annotations\camo5_test_split1.json` | 47 | 2655 | 2655 | 3108 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Bat_1shot_split1.json` | 47 | 1 | 1 | 1 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Bat_2shot_split1.json` | 47 | 2 | 2 | 2 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Bat_3shot_split1.json` | 47 | 3 | 3 | 3 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Bat_5shot_split1.json` | 47 | 4 | 4 | 5 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Bear_1shot_split1.json` | 47 | 1 | 1 | 1 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Bear_2shot_split1.json` | 47 | 2 | 2 | 2 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Bear_3shot_split1.json` | 47 | 3 | 3 | 3 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Bear_5shot_split1.json` | 47 | 5 | 5 | 5 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Bird_1shot_split1.json` | 47 | 1 | 1 | 1 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Bird_2shot_split1.json` | 47 | 2 | 2 | 2 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Bird_3shot_split1.json` | 47 | 3 | 3 | 3 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Bird_5shot_split1.json` | 47 | 4 | 4 | 5 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Body_Painting_1shot_split1.json` | 47 | 1 | 1 | 1 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Body_Painting_2shot_split1.json` | 47 | 2 | 2 | 2 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Body_Painting_3shot_split1.json` | 47 | 3 | 3 | 3 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Body_Painting_5shot_split1.json` | 47 | 4 | 4 | 5 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Camel_1shot_split1.json` | 47 | 1 | 1 | 1 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Camel_2shot_split1.json` | 47 | 2 | 2 | 2 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Camel_3shot_split1.json` | 47 | 3 | 3 | 3 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Camel_5shot_split1.json` | 47 | 4 | 4 | 5 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Cat_1shot_split1.json` | 47 | 1 | 1 | 1 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Cat_2shot_split1.json` | 47 | 2 | 2 | 2 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Cat_3shot_split1.json` | 47 | 3 | 3 | 3 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Cat_5shot_split1.json` | 47 | 5 | 5 | 5 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Crab_1shot_split1.json` | 47 | 1 | 1 | 1 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Crab_2shot_split1.json` | 47 | 2 | 2 | 2 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Crab_3shot_split1.json` | 47 | 3 | 3 | 3 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Crab_5shot_split1.json` | 47 | 5 | 5 | 5 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Crocodile_1shot_split1.json` | 47 | 1 | 1 | 1 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Crocodile_2shot_split1.json` | 47 | 2 | 2 | 2 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Crocodile_3shot_split1.json` | 47 | 3 | 3 | 3 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Crocodile_5shot_split1.json` | 47 | 5 | 5 | 5 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Deer_1shot_split1.json` | 47 | 1 | 1 | 1 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Deer_2shot_split1.json` | 47 | 2 | 2 | 2 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Deer_3shot_split1.json` | 47 | 3 | 3 | 3 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Deer_5shot_split1.json` | 47 | 4 | 4 | 5 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Dog_1shot_split1.json` | 47 | 1 | 1 | 1 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Dog_2shot_split1.json` | 47 | 2 | 2 | 2 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Dog_3shot_split1.json` | 47 | 3 | 3 | 3 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Dog_5shot_split1.json` | 47 | 4 | 4 | 5 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Dolphin_1shot_split1.json` | 47 | 1 | 1 | 1 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Dolphin_2shot_split1.json` | 47 | 2 | 2 | 2 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Dolphin_3shot_split1.json` | 47 | 3 | 3 | 3 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Dolphin_5shot_split1.json` | 47 | 4 | 4 | 5 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Elephant_1shot_split1.json` | 47 | 1 | 1 | 1 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Elephant_2shot_split1.json` | 47 | 2 | 2 | 2 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Elephant_3shot_split1.json` | 47 | 3 | 3 | 3 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Elephant_5shot_split1.json` | 47 | 4 | 4 | 5 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Fish_1shot_split1.json` | 47 | 1 | 1 | 1 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Fish_2shot_split1.json` | 47 | 2 | 2 | 2 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Fish_3shot_split1.json` | 47 | 3 | 3 | 3 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Fish_5shot_split1.json` | 47 | 4 | 4 | 5 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Fox_1shot_split1.json` | 47 | 1 | 1 | 1 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Fox_2shot_split1.json` | 47 | 2 | 2 | 2 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Fox_3shot_split1.json` | 47 | 3 | 3 | 3 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Fox_5shot_split1.json` | 47 | 4 | 4 | 5 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Frog_1shot_split1.json` | 47 | 1 | 1 | 1 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Frog_2shot_split1.json` | 47 | 2 | 2 | 2 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Frog_3shot_split1.json` | 47 | 3 | 3 | 3 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Frog_5shot_split1.json` | 47 | 4 | 4 | 5 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Giraffe_1shot_split1.json` | 47 | 1 | 1 | 1 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Giraffe_2shot_split1.json` | 47 | 2 | 2 | 2 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Giraffe_3shot_split1.json` | 47 | 3 | 3 | 3 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Giraffe_5shot_split1.json` | 47 | 4 | 4 | 5 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Goat_1shot_split1.json` | 47 | 1 | 1 | 1 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Goat_2shot_split1.json` | 47 | 2 | 2 | 2 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Goat_3shot_split1.json` | 47 | 3 | 3 | 3 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Goat_5shot_split1.json` | 47 | 4 | 4 | 5 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Hedgehog_1shot_split1.json` | 47 | 1 | 1 | 1 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Hedgehog_2shot_split1.json` | 47 | 2 | 2 | 2 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Hedgehog_3shot_split1.json` | 47 | 3 | 3 | 3 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Hedgehog_5shot_split1.json` | 47 | 5 | 5 | 5 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Horse_1shot_split1.json` | 47 | 1 | 1 | 1 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Horse_2shot_split1.json` | 47 | 2 | 2 | 2 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Horse_3shot_split1.json` | 47 | 3 | 3 | 3 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Horse_5shot_split1.json` | 47 | 4 | 4 | 5 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Insect_1shot_split1.json` | 47 | 1 | 1 | 1 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Insect_2shot_split1.json` | 47 | 2 | 2 | 2 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Insect_3shot_split1.json` | 47 | 3 | 3 | 3 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Insect_5shot_split1.json` | 47 | 4 | 4 | 5 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Kangaroo_1shot_split1.json` | 47 | 1 | 1 | 1 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Kangaroo_2shot_split1.json` | 47 | 2 | 2 | 2 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Kangaroo_3shot_split1.json` | 47 | 3 | 3 | 3 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Kangaroo_5shot_split1.json` | 47 | 4 | 4 | 5 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Leopard_1shot_split1.json` | 47 | 1 | 1 | 1 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Leopard_2shot_split1.json` | 47 | 2 | 2 | 2 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Leopard_3shot_split1.json` | 47 | 3 | 3 | 3 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Leopard_5shot_split1.json` | 47 | 4 | 4 | 5 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Lion_1shot_split1.json` | 47 | 1 | 1 | 1 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Lion_2shot_split1.json` | 47 | 2 | 2 | 2 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Lion_3shot_split1.json` | 47 | 3 | 3 | 3 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Lion_5shot_split1.json` | 47 | 4 | 4 | 5 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Lizard_1shot_split1.json` | 47 | 1 | 1 | 1 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Lizard_2shot_split1.json` | 47 | 2 | 2 | 2 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Lizard_3shot_split1.json` | 47 | 3 | 3 | 3 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Lizard_5shot_split1.json` | 47 | 4 | 4 | 5 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Monkey_1shot_split1.json` | 47 | 1 | 1 | 1 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Monkey_2shot_split1.json` | 47 | 2 | 2 | 2 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Monkey_3shot_split1.json` | 47 | 3 | 3 | 3 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Monkey_5shot_split1.json` | 47 | 4 | 4 | 5 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Moth_1shot_split1.json` | 47 | 1 | 1 | 1 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Moth_2shot_split1.json` | 47 | 2 | 2 | 2 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Moth_3shot_split1.json` | 47 | 3 | 3 | 3 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Moth_5shot_split1.json` | 47 | 4 | 4 | 5 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Mouse_1shot_split1.json` | 47 | 1 | 1 | 1 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Mouse_2shot_split1.json` | 47 | 2 | 2 | 2 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Mouse_3shot_split1.json` | 47 | 3 | 3 | 3 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Mouse_5shot_split1.json` | 47 | 4 | 4 | 5 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Octopus_1shot_split1.json` | 47 | 1 | 1 | 1 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Octopus_2shot_split1.json` | 47 | 2 | 2 | 2 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Octopus_3shot_split1.json` | 47 | 3 | 3 | 3 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Octopus_5shot_split1.json` | 47 | 4 | 4 | 5 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Penguin_1shot_split1.json` | 47 | 1 | 1 | 1 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Penguin_2shot_split1.json` | 47 | 2 | 2 | 2 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Penguin_3shot_split1.json` | 47 | 3 | 3 | 3 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Penguin_5shot_split1.json` | 47 | 4 | 4 | 5 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Pig_1shot_split1.json` | 47 | 1 | 1 | 1 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Pig_2shot_split1.json` | 47 | 2 | 2 | 2 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Pig_3shot_split1.json` | 47 | 3 | 3 | 3 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Pig_5shot_split1.json` | 47 | 4 | 4 | 5 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Rabbit_1shot_split1.json` | 47 | 1 | 1 | 1 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Rabbit_2shot_split1.json` | 47 | 2 | 2 | 2 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Rabbit_3shot_split1.json` | 47 | 3 | 3 | 3 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Rabbit_5shot_split1.json` | 47 | 4 | 4 | 5 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Rhino_1shot_split1.json` | 47 | 1 | 1 | 1 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Rhino_2shot_split1.json` | 47 | 2 | 2 | 2 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Rhino_3shot_split1.json` | 47 | 3 | 3 | 3 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Rhino_5shot_split1.json` | 47 | 4 | 4 | 5 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Scorpion_1shot_split1.json` | 47 | 1 | 1 | 1 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Scorpion_2shot_split1.json` | 47 | 2 | 2 | 2 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Scorpion_3shot_split1.json` | 47 | 3 | 3 | 3 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Scorpion_5shot_split1.json` | 47 | 4 | 4 | 5 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Sea_Lion_1shot_split1.json` | 47 | 1 | 1 | 1 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Sea_Lion_2shot_split1.json` | 47 | 2 | 2 | 2 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Sea_Lion_3shot_split1.json` | 47 | 3 | 3 | 3 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Sea_Lion_5shot_split1.json` | 47 | 4 | 4 | 5 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Seahorse_1shot_split1.json` | 47 | 1 | 1 | 1 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Seahorse_2shot_split1.json` | 47 | 2 | 2 | 2 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Seahorse_3shot_split1.json` | 47 | 3 | 3 | 3 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Seahorse_5shot_split1.json` | 47 | 4 | 4 | 5 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Shrimp_1shot_split1.json` | 47 | 1 | 1 | 1 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Shrimp_2shot_split1.json` | 47 | 2 | 2 | 2 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Shrimp_3shot_split1.json` | 47 | 3 | 3 | 3 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Shrimp_5shot_split1.json` | 47 | 4 | 4 | 5 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Sloth_1shot_split1.json` | 47 | 1 | 1 | 1 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Sloth_2shot_split1.json` | 47 | 2 | 2 | 2 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Sloth_3shot_split1.json` | 47 | 3 | 3 | 3 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Sloth_5shot_split1.json` | 47 | 4 | 4 | 5 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Snail_1shot_split1.json` | 47 | 1 | 1 | 1 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Snail_2shot_split1.json` | 47 | 2 | 2 | 2 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Snail_3shot_split1.json` | 47 | 3 | 3 | 3 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Snail_5shot_split1.json` | 47 | 4 | 4 | 5 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Snake_1shot_split1.json` | 47 | 1 | 1 | 1 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Snake_2shot_split1.json` | 47 | 2 | 2 | 2 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Snake_3shot_split1.json` | 47 | 3 | 3 | 3 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Snake_5shot_split1.json` | 47 | 5 | 5 | 5 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Solider_1shot_split1.json` | 47 | 1 | 1 | 1 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Solider_2shot_split1.json` | 47 | 2 | 2 | 2 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Solider_3shot_split1.json` | 47 | 3 | 3 | 3 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Solider_5shot_split1.json` | 47 | 4 | 4 | 5 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Spider_1shot_split1.json` | 47 | 1 | 1 | 1 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Spider_2shot_split1.json` | 47 | 2 | 2 | 2 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Spider_3shot_split1.json` | 47 | 3 | 3 | 3 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Spider_5shot_split1.json` | 47 | 5 | 5 | 5 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Squirrel_1shot_split1.json` | 47 | 1 | 1 | 1 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Squirrel_2shot_split1.json` | 47 | 2 | 2 | 2 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Squirrel_3shot_split1.json` | 47 | 3 | 3 | 3 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Squirrel_5shot_split1.json` | 47 | 5 | 5 | 5 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Starfish_1shot_split1.json` | 47 | 1 | 1 | 1 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Starfish_2shot_split1.json` | 47 | 2 | 2 | 2 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Starfish_3shot_split1.json` | 47 | 3 | 3 | 3 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Starfish_5shot_split1.json` | 47 | 4 | 4 | 5 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Tiger_1shot_split1.json` | 47 | 1 | 1 | 1 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Tiger_2shot_split1.json` | 47 | 2 | 2 | 2 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Tiger_3shot_split1.json` | 47 | 3 | 3 | 3 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Tiger_5shot_split1.json` | 47 | 5 | 5 | 5 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Turtle_1shot_split1.json` | 47 | 1 | 1 | 1 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Turtle_2shot_split1.json` | 47 | 2 | 2 | 2 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Turtle_3shot_split1.json` | 47 | 3 | 3 | 3 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Turtle_5shot_split1.json` | 47 | 4 | 4 | 5 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Weasel_1shot_split1.json` | 47 | 1 | 1 | 1 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Weasel_2shot_split1.json` | 47 | 2 | 2 | 2 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Weasel_3shot_split1.json` | 47 | 3 | 3 | 3 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Weasel_5shot_split1.json` | 47 | 4 | 4 | 5 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Worm_1shot_split1.json` | 47 | 1 | 1 | 1 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Worm_2shot_split1.json` | 47 | 2 | 2 | 2 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Worm_3shot_split1.json` | 47 | 3 | 3 | 3 | 0 |
| `few-shot-annotations\subsplit\split1\camo5_Worm_5shot_split1.json` | 47 | 4 | 4 | 5 | 0 |
| **TỔNG CỘNG (Cộng ngang)** | — | **3134** | **2852** | **3625** | — |

## 2. Truy vết nguyên nhân chênh lệch số lượng ảnh (Ví dụ: 197 + 2.655 = 2.852 vs 2.858)
- **Ảnh có trên ổ cứng nhưng KHÔNG nằm trong bất kỳ file JSON nào (0 ảnh):** `[]`
- **Ảnh khai báo trong JSON nhưng THIẾU file trên ổ cứng (0 ảnh):** `[]`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Bat_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Bat_2shot_split1.json` (1 ảnh):** `['camouflage_00002816.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Bat_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Bat_3shot_split1.json` (1 ảnh):** `['camouflage_00002816.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Bat_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Bat_5shot_split1.json` (1 ảnh):** `['camouflage_00002816.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Bat_2shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Bat_3shot_split1.json` (2 ảnh):** `['camouflage_00002816.jpg', 'camouflage_00002817.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Bat_2shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Bat_5shot_split1.json` (2 ảnh):** `['camouflage_00002816.jpg', 'camouflage_00002817.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Bat_3shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Bat_5shot_split1.json` (3 ảnh):** `['camouflage_00002816.jpg', 'camouflage_00002817.jpg', 'camouflage_00002818.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Bear_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Bear_2shot_split1.json` (1 ảnh):** `['camouflage_00002820.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Bear_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Bear_3shot_split1.json` (1 ảnh):** `['camouflage_00002820.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Bear_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Bear_5shot_split1.json` (1 ảnh):** `['camouflage_00002820.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Bear_2shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Bear_3shot_split1.json` (2 ảnh):** `['camouflage_00002820.jpg', 'camouflage_00002821.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Bear_2shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Bear_5shot_split1.json` (2 ảnh):** `['camouflage_00002820.jpg', 'camouflage_00002821.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Bear_3shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Bear_5shot_split1.json` (3 ảnh):** `['camouflage_00002820.jpg', 'camouflage_00002821.jpg', 'camouflage_00002822.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Bird_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Bird_2shot_split1.json` (1 ảnh):** `['camouflage_00000628.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Bird_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Bird_3shot_split1.json` (1 ảnh):** `['camouflage_00000628.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Bird_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Bird_5shot_split1.json` (1 ảnh):** `['camouflage_00000628.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Bird_2shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Bird_3shot_split1.json` (2 ảnh):** `['camouflage_00000628.jpg', 'camouflage_00000630.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Bird_2shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Bird_5shot_split1.json` (2 ảnh):** `['camouflage_00000628.jpg', 'camouflage_00000630.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Bird_3shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Bird_5shot_split1.json` (3 ảnh):** `['camouflage_00000628.jpg', 'camouflage_00000630.jpg', 'camouflage_00000640.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Body_Painting_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Body_Painting_2shot_split1.json` (1 ảnh):** `['camouflage_00002048.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Body_Painting_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Body_Painting_3shot_split1.json` (1 ảnh):** `['camouflage_00002048.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Body_Painting_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Body_Painting_5shot_split1.json` (1 ảnh):** `['camouflage_00002048.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Body_Painting_2shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Body_Painting_3shot_split1.json` (2 ảnh):** `['camouflage_00002048.jpg', 'camouflage_00002049.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Body_Painting_2shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Body_Painting_5shot_split1.json` (2 ảnh):** `['camouflage_00002048.jpg', 'camouflage_00002049.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Body_Painting_3shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Body_Painting_5shot_split1.json` (3 ảnh):** `['camouflage_00002048.jpg', 'camouflage_00002049.jpg', 'camouflage_00002050.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Camel_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Camel_2shot_split1.json` (1 ảnh):** `['camouflage_00002784.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Camel_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Camel_3shot_split1.json` (1 ảnh):** `['camouflage_00002784.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Camel_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Camel_5shot_split1.json` (1 ảnh):** `['camouflage_00002784.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Camel_2shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Camel_3shot_split1.json` (2 ảnh):** `['camouflage_00002784.jpg', 'camouflage_00002785.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Camel_2shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Camel_5shot_split1.json` (2 ảnh):** `['camouflage_00002784.jpg', 'camouflage_00002785.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Camel_3shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Camel_5shot_split1.json` (3 ảnh):** `['camouflage_00002784.jpg', 'camouflage_00002785.jpg', 'camouflage_00002786.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Cat_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Cat_2shot_split1.json` (1 ảnh):** `['camouflage_00001137.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Cat_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Cat_3shot_split1.json` (1 ảnh):** `['camouflage_00001137.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Cat_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Cat_5shot_split1.json` (1 ảnh):** `['camouflage_00001137.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Cat_2shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Cat_3shot_split1.json` (2 ảnh):** `['camouflage_00001137.jpg', 'camouflage_00001138.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Cat_2shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Cat_5shot_split1.json` (2 ảnh):** `['camouflage_00001137.jpg', 'camouflage_00001138.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Cat_3shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Cat_5shot_split1.json` (3 ảnh):** `['camouflage_00001137.jpg', 'camouflage_00001138.jpg', 'camouflage_00001139.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Crab_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Crab_2shot_split1.json` (1 ảnh):** `['camouflage_00001792.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Crab_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Crab_3shot_split1.json` (1 ảnh):** `['camouflage_00001792.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Crab_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Crab_5shot_split1.json` (1 ảnh):** `['camouflage_00001792.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Crab_2shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Crab_3shot_split1.json` (2 ảnh):** `['camouflage_00001792.jpg', 'camouflage_00001793.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Crab_2shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Crab_5shot_split1.json` (2 ảnh):** `['camouflage_00001792.jpg', 'camouflage_00001793.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Crab_3shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Crab_5shot_split1.json` (3 ảnh):** `['camouflage_00001792.jpg', 'camouflage_00001793.jpg', 'camouflage_00001794.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Crocodile_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Crocodile_2shot_split1.json` (1 ảnh):** `['camouflage_00001474.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Crocodile_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Crocodile_3shot_split1.json` (1 ảnh):** `['camouflage_00001474.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Crocodile_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Crocodile_5shot_split1.json` (1 ảnh):** `['camouflage_00001474.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Crocodile_2shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Crocodile_3shot_split1.json` (2 ảnh):** `['camouflage_00001474.jpg', 'camouflage_00001475.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Crocodile_2shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Crocodile_5shot_split1.json` (2 ảnh):** `['camouflage_00001474.jpg', 'camouflage_00001475.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Crocodile_3shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Crocodile_5shot_split1.json` (3 ảnh):** `['camouflage_00001474.jpg', 'camouflage_00001475.jpg', 'camouflage_00001476.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Deer_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Deer_2shot_split1.json` (1 ảnh):** `['camouflage_00001120.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Deer_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Deer_3shot_split1.json` (1 ảnh):** `['camouflage_00001120.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Deer_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Deer_5shot_split1.json` (1 ảnh):** `['camouflage_00001120.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Deer_2shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Deer_3shot_split1.json` (2 ảnh):** `['camouflage_00001120.jpg', 'camouflage_00001121.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Deer_2shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Deer_5shot_split1.json` (2 ảnh):** `['camouflage_00001120.jpg', 'camouflage_00001121.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Deer_3shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Deer_5shot_split1.json` (3 ảnh):** `['camouflage_00001120.jpg', 'camouflage_00001121.jpg', 'camouflage_00001122.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Dog_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Dog_2shot_split1.json` (1 ảnh):** `['camouflage_00001247.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Dog_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Dog_3shot_split1.json` (1 ảnh):** `['camouflage_00001247.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Dog_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Dog_5shot_split1.json` (1 ảnh):** `['camouflage_00001247.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Dog_2shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Dog_3shot_split1.json` (2 ảnh):** `['camouflage_00001247.jpg', 'camouflage_00001248.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Dog_2shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Dog_5shot_split1.json` (2 ảnh):** `['camouflage_00001247.jpg', 'camouflage_00001248.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Dog_3shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Dog_5shot_split1.json` (3 ảnh):** `['camouflage_00001230.jpg', 'camouflage_00001247.jpg', 'camouflage_00001248.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Dolphin_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Dolphin_2shot_split1.json` (1 ảnh):** `['camouflage_00002690.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Dolphin_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Dolphin_3shot_split1.json` (1 ảnh):** `['camouflage_00002690.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Dolphin_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Dolphin_5shot_split1.json` (1 ảnh):** `['camouflage_00002690.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Dolphin_2shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Dolphin_3shot_split1.json` (2 ảnh):** `['camouflage_00002690.jpg', 'camouflage_00002691.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Dolphin_2shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Dolphin_5shot_split1.json` (2 ảnh):** `['camouflage_00002690.jpg', 'camouflage_00002691.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Dolphin_3shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Dolphin_5shot_split1.json` (3 ảnh):** `['camouflage_00002690.jpg', 'camouflage_00002691.jpg', 'camouflage_00002693.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Elephant_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Elephant_2shot_split1.json` (1 ảnh):** `['camouflage_00002807.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Elephant_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Elephant_3shot_split1.json` (1 ảnh):** `['camouflage_00002807.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Elephant_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Elephant_5shot_split1.json` (1 ảnh):** `['camouflage_00002807.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Elephant_2shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Elephant_3shot_split1.json` (2 ảnh):** `['camouflage_00002795.jpg', 'camouflage_00002807.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Elephant_2shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Elephant_5shot_split1.json` (2 ảnh):** `['camouflage_00002795.jpg', 'camouflage_00002807.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Elephant_3shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Elephant_5shot_split1.json` (3 ảnh):** `['camouflage_00002795.jpg', 'camouflage_00002796.jpg', 'camouflage_00002807.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Fish_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Fish_2shot_split1.json` (1 ảnh):** `['camouflage_00001738.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Fish_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Fish_3shot_split1.json` (1 ảnh):** `['camouflage_00001738.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Fish_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Fish_5shot_split1.json` (1 ảnh):** `['camouflage_00001738.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Fish_2shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Fish_3shot_split1.json` (2 ảnh):** `['camouflage_00001738.jpg', 'camouflage_00001773.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Fish_2shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Fish_5shot_split1.json` (2 ảnh):** `['camouflage_00001738.jpg', 'camouflage_00001773.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Fish_3shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Fish_5shot_split1.json` (3 ảnh):** `['camouflage_00001738.jpg', 'camouflage_00001743.jpg', 'camouflage_00001773.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Fox_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Fox_2shot_split1.json` (1 ảnh):** `['camouflage_00001206.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Fox_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Fox_3shot_split1.json` (1 ảnh):** `['camouflage_00001206.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Fox_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Fox_5shot_split1.json` (1 ảnh):** `['camouflage_00001206.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Fox_2shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Fox_3shot_split1.json` (2 ảnh):** `['camouflage_00000434.jpg', 'camouflage_00001206.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Fox_2shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Fox_5shot_split1.json` (2 ảnh):** `['camouflage_00000434.jpg', 'camouflage_00001206.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Fox_3shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Fox_5shot_split1.json` (3 ảnh):** `['camouflage_00000434.jpg', 'camouflage_00000435.jpg', 'camouflage_00001206.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Frog_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Frog_2shot_split1.json` (1 ảnh):** `['camouflage_00002560.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Frog_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Frog_3shot_split1.json` (1 ảnh):** `['camouflage_00002560.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Frog_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Frog_5shot_split1.json` (1 ảnh):** `['camouflage_00002560.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Frog_2shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Frog_3shot_split1.json` (2 ảnh):** `['camouflage_00002560.jpg', 'camouflage_00002561.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Frog_2shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Frog_5shot_split1.json` (2 ảnh):** `['camouflage_00002560.jpg', 'camouflage_00002561.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Frog_3shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Frog_5shot_split1.json` (3 ảnh):** `['camouflage_00002560.jpg', 'camouflage_00002561.jpg', 'camouflage_00002562.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Giraffe_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Giraffe_2shot_split1.json` (1 ảnh):** `['camouflage_00001113.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Giraffe_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Giraffe_3shot_split1.json` (1 ảnh):** `['camouflage_00001113.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Giraffe_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Giraffe_5shot_split1.json` (1 ảnh):** `['camouflage_00001113.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Giraffe_2shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Giraffe_3shot_split1.json` (2 ảnh):** `['camouflage_00001112.jpg', 'camouflage_00001113.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Giraffe_2shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Giraffe_5shot_split1.json` (2 ảnh):** `['camouflage_00001112.jpg', 'camouflage_00001113.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Giraffe_3shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Giraffe_5shot_split1.json` (3 ảnh):** `['camouflage_00001102.jpg', 'camouflage_00001112.jpg', 'camouflage_00001113.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Goat_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Goat_2shot_split1.json` (1 ảnh):** `['camouflage_00000352.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Goat_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Goat_3shot_split1.json` (1 ảnh):** `['camouflage_00000352.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Goat_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Goat_5shot_split1.json` (1 ảnh):** `['camouflage_00000352.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Goat_2shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Goat_3shot_split1.json` (2 ảnh):** `['camouflage_00000352.jpg', 'camouflage_00001128.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Goat_2shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Goat_5shot_split1.json` (2 ảnh):** `['camouflage_00000352.jpg', 'camouflage_00001128.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Goat_3shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Goat_5shot_split1.json` (3 ảnh):** `['camouflage_00000352.jpg', 'camouflage_00001128.jpg', 'camouflage_00001129.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Hedgehog_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Hedgehog_2shot_split1.json` (1 ảnh):** `['camouflage_00001312.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Hedgehog_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Hedgehog_3shot_split1.json` (1 ảnh):** `['camouflage_00001312.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Hedgehog_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Hedgehog_5shot_split1.json` (1 ảnh):** `['camouflage_00001312.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Hedgehog_2shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Hedgehog_3shot_split1.json` (2 ảnh):** `['camouflage_00001312.jpg', 'camouflage_00001313.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Hedgehog_2shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Hedgehog_5shot_split1.json` (2 ảnh):** `['camouflage_00001312.jpg', 'camouflage_00001313.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Hedgehog_3shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Hedgehog_5shot_split1.json` (3 ảnh):** `['camouflage_00001312.jpg', 'camouflage_00001313.jpg', 'camouflage_00001314.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Horse_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Horse_2shot_split1.json` (1 ảnh):** `['camouflage_00002702.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Horse_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Horse_3shot_split1.json` (1 ảnh):** `['camouflage_00002702.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Horse_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Horse_5shot_split1.json` (1 ảnh):** `['camouflage_00002702.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Horse_2shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Horse_3shot_split1.json` (2 ảnh):** `['camouflage_00002702.jpg', 'camouflage_00002703.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Horse_2shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Horse_5shot_split1.json` (2 ảnh):** `['camouflage_00002702.jpg', 'camouflage_00002703.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Horse_3shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Horse_5shot_split1.json` (3 ảnh):** `['camouflage_00002702.jpg', 'camouflage_00002703.jpg', 'camouflage_00002704.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Insect_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Insect_2shot_split1.json` (1 ảnh):** `['camouflage_00002137.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Insect_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Insect_3shot_split1.json` (1 ảnh):** `['camouflage_00002137.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Insect_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Insect_5shot_split1.json` (1 ảnh):** `['camouflage_00002137.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Insect_2shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Insect_3shot_split1.json` (2 ảnh):** `['camouflage_00002137.jpg', 'camouflage_00002138.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Insect_2shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Insect_5shot_split1.json` (2 ảnh):** `['camouflage_00002137.jpg', 'camouflage_00002138.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Insect_3shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Insect_5shot_split1.json` (3 ảnh):** `['camouflage_00000528.jpg', 'camouflage_00002137.jpg', 'camouflage_00002138.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Kangaroo_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Kangaroo_2shot_split1.json` (1 ảnh):** `['camouflage_00001052.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Kangaroo_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Kangaroo_3shot_split1.json` (1 ảnh):** `['camouflage_00001052.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Kangaroo_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Kangaroo_5shot_split1.json` (1 ảnh):** `['camouflage_00001052.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Kangaroo_2shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Kangaroo_3shot_split1.json` (2 ảnh):** `['camouflage_00001052.jpg', 'camouflage_00001053.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Kangaroo_2shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Kangaroo_5shot_split1.json` (2 ảnh):** `['camouflage_00001052.jpg', 'camouflage_00001053.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Kangaroo_3shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Kangaroo_5shot_split1.json` (3 ảnh):** `['camouflage_00001052.jpg', 'camouflage_00001053.jpg', 'camouflage_00002718.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Leopard_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Leopard_2shot_split1.json` (1 ảnh):** `['camouflage_00001025.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Leopard_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Leopard_3shot_split1.json` (1 ảnh):** `['camouflage_00001025.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Leopard_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Leopard_5shot_split1.json` (1 ảnh):** `['camouflage_00001025.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Leopard_2shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Leopard_3shot_split1.json` (2 ảnh):** `['camouflage_00001025.jpg', 'camouflage_00001026.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Leopard_2shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Leopard_5shot_split1.json` (2 ảnh):** `['camouflage_00001025.jpg', 'camouflage_00001026.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Leopard_3shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Leopard_5shot_split1.json` (3 ảnh):** `['camouflage_00001025.jpg', 'camouflage_00001026.jpg', 'camouflage_00001027.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Lion_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Lion_2shot_split1.json` (1 ảnh):** `['camouflage_00001024.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Lion_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Lion_3shot_split1.json` (1 ảnh):** `['camouflage_00001024.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Lion_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Lion_5shot_split1.json` (1 ảnh):** `['camouflage_00001024.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Lion_2shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Lion_3shot_split1.json` (2 ảnh):** `['camouflage_00000425.jpg', 'camouflage_00001024.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Lion_2shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Lion_5shot_split1.json` (2 ảnh):** `['camouflage_00000425.jpg', 'camouflage_00001024.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Lion_3shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Lion_5shot_split1.json` (3 ảnh):** `['camouflage_00000425.jpg', 'camouflage_00000426.jpg', 'camouflage_00001024.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Lizard_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Lizard_2shot_split1.json` (1 ảnh):** `['camouflage_00000252.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Lizard_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Lizard_3shot_split1.json` (1 ảnh):** `['camouflage_00000252.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Lizard_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Lizard_5shot_split1.json` (1 ảnh):** `['camouflage_00000252.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Lizard_2shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Lizard_3shot_split1.json` (2 ảnh):** `['camouflage_00000252.jpg', 'camouflage_00000253.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Lizard_2shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Lizard_5shot_split1.json` (2 ảnh):** `['camouflage_00000252.jpg', 'camouflage_00000253.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Lizard_3shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Lizard_5shot_split1.json` (3 ảnh):** `['camouflage_00000252.jpg', 'camouflage_00000253.jpg', 'camouflage_00000254.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Monkey_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Monkey_2shot_split1.json` (1 ảnh):** `['camouflage_00002740.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Monkey_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Monkey_3shot_split1.json` (1 ảnh):** `['camouflage_00002740.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Monkey_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Monkey_5shot_split1.json` (1 ảnh):** `['camouflage_00002740.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Monkey_2shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Monkey_3shot_split1.json` (2 ảnh):** `['camouflage_00002740.jpg', 'camouflage_00002741.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Monkey_2shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Monkey_5shot_split1.json` (2 ảnh):** `['camouflage_00002740.jpg', 'camouflage_00002741.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Monkey_3shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Monkey_5shot_split1.json` (3 ảnh):** `['camouflage_00002740.jpg', 'camouflage_00002741.jpg', 'camouflage_00002742.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Moth_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Moth_2shot_split1.json` (1 ảnh):** `['camouflage_00002272.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Moth_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Moth_3shot_split1.json` (1 ảnh):** `['camouflage_00002272.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Moth_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Moth_5shot_split1.json` (1 ảnh):** `['camouflage_00002272.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Moth_2shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Moth_3shot_split1.json` (2 ảnh):** `['camouflage_00002272.jpg', 'camouflage_00002273.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Moth_2shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Moth_5shot_split1.json` (2 ảnh):** `['camouflage_00002272.jpg', 'camouflage_00002273.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Moth_3shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Moth_5shot_split1.json` (3 ảnh):** `['camouflage_00002234.jpg', 'camouflage_00002272.jpg', 'camouflage_00002273.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Mouse_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Mouse_2shot_split1.json` (1 ảnh):** `['camouflage_00001259.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Mouse_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Mouse_3shot_split1.json` (1 ảnh):** `['camouflage_00001259.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Mouse_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Mouse_5shot_split1.json` (1 ảnh):** `['camouflage_00001259.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Mouse_2shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Mouse_3shot_split1.json` (2 ảnh):** `['camouflage_00000493.jpg', 'camouflage_00001259.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Mouse_2shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Mouse_5shot_split1.json` (2 ảnh):** `['camouflage_00000493.jpg', 'camouflage_00001259.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Mouse_3shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Mouse_5shot_split1.json` (3 ảnh):** `['camouflage_00000493.jpg', 'camouflage_00000494.jpg', 'camouflage_00001259.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Octopus_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Octopus_2shot_split1.json` (1 ảnh):** `['camouflage_00000000.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Octopus_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Octopus_3shot_split1.json` (1 ảnh):** `['camouflage_00000000.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Octopus_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Octopus_5shot_split1.json` (1 ảnh):** `['camouflage_00000000.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Octopus_2shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Octopus_3shot_split1.json` (2 ảnh):** `['camouflage_00000000.jpg', 'camouflage_00000001.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Octopus_2shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Octopus_5shot_split1.json` (2 ảnh):** `['camouflage_00000000.jpg', 'camouflage_00000001.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Octopus_3shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Octopus_5shot_split1.json` (3 ảnh):** `['camouflage_00000000.jpg', 'camouflage_00000001.jpg', 'camouflage_00000002.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Penguin_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Penguin_2shot_split1.json` (1 ảnh):** `['camouflage_00000694.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Penguin_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Penguin_3shot_split1.json` (1 ảnh):** `['camouflage_00000694.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Penguin_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Penguin_5shot_split1.json` (1 ảnh):** `['camouflage_00000694.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Penguin_2shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Penguin_3shot_split1.json` (2 ảnh):** `['camouflage_00000694.jpg', 'camouflage_00002757.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Penguin_2shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Penguin_5shot_split1.json` (2 ảnh):** `['camouflage_00000694.jpg', 'camouflage_00002757.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Penguin_3shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Penguin_5shot_split1.json` (3 ảnh):** `['camouflage_00000694.jpg', 'camouflage_00002757.jpg', 'camouflage_00002758.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Pig_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Pig_2shot_split1.json` (1 ảnh):** `['camouflage_00000512.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Pig_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Pig_3shot_split1.json` (1 ảnh):** `['camouflage_00000512.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Pig_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Pig_5shot_split1.json` (1 ảnh):** `['camouflage_00000512.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Pig_2shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Pig_3shot_split1.json` (2 ảnh):** `['camouflage_00000512.jpg', 'camouflage_00000513.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Pig_2shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Pig_5shot_split1.json` (2 ảnh):** `['camouflage_00000512.jpg', 'camouflage_00000513.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Pig_3shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Pig_5shot_split1.json` (3 ảnh):** `['camouflage_00000512.jpg', 'camouflage_00000513.jpg', 'camouflage_00000514.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Rabbit_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Rabbit_2shot_split1.json` (1 ảnh):** `['camouflage_00001059.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Rabbit_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Rabbit_3shot_split1.json` (1 ảnh):** `['camouflage_00001059.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Rabbit_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Rabbit_5shot_split1.json` (1 ảnh):** `['camouflage_00001059.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Rabbit_2shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Rabbit_3shot_split1.json` (2 ảnh):** `['camouflage_00001059.jpg', 'camouflage_00001060.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Rabbit_2shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Rabbit_5shot_split1.json` (2 ảnh):** `['camouflage_00001059.jpg', 'camouflage_00001060.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Rabbit_3shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Rabbit_5shot_split1.json` (3 ảnh):** `['camouflage_00001059.jpg', 'camouflage_00001060.jpg', 'camouflage_00001061.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Rhino_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Rhino_2shot_split1.json` (1 ảnh):** `['camouflage_00001288.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Rhino_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Rhino_3shot_split1.json` (1 ảnh):** `['camouflage_00001288.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Rhino_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Rhino_5shot_split1.json` (1 ảnh):** `['camouflage_00001288.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Rhino_2shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Rhino_3shot_split1.json` (2 ảnh):** `['camouflage_00001288.jpg', 'camouflage_00001290.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Rhino_2shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Rhino_5shot_split1.json` (2 ảnh):** `['camouflage_00001288.jpg', 'camouflage_00001290.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Rhino_3shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Rhino_5shot_split1.json` (3 ảnh):** `['camouflage_00001288.jpg', 'camouflage_00001290.jpg', 'camouflage_00001291.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Scorpion_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Scorpion_2shot_split1.json` (1 ảnh):** `['camouflage_00002600.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Scorpion_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Scorpion_3shot_split1.json` (1 ảnh):** `['camouflage_00002600.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Scorpion_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Scorpion_5shot_split1.json` (1 ảnh):** `['camouflage_00002600.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Scorpion_2shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Scorpion_3shot_split1.json` (2 ảnh):** `['camouflage_00002600.jpg', 'camouflage_00002601.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Scorpion_2shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Scorpion_5shot_split1.json` (2 ảnh):** `['camouflage_00002600.jpg', 'camouflage_00002601.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Scorpion_3shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Scorpion_5shot_split1.json` (3 ảnh):** `['camouflage_00002600.jpg', 'camouflage_00002601.jpg', 'camouflage_00002602.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Sea_Lion_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Sea_Lion_2shot_split1.json` (1 ảnh):** `['camouflage_00000414.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Sea_Lion_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Sea_Lion_3shot_split1.json` (1 ảnh):** `['camouflage_00000414.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Sea_Lion_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Sea_Lion_5shot_split1.json` (1 ảnh):** `['camouflage_00000414.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Sea_Lion_2shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Sea_Lion_3shot_split1.json` (2 ảnh):** `['camouflage_00000414.jpg', 'camouflage_00000415.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Sea_Lion_2shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Sea_Lion_5shot_split1.json` (2 ảnh):** `['camouflage_00000414.jpg', 'camouflage_00000415.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Sea_Lion_3shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Sea_Lion_5shot_split1.json` (3 ảnh):** `['camouflage_00000414.jpg', 'camouflage_00000415.jpg', 'camouflage_00000416.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Seahorse_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Seahorse_2shot_split1.json` (1 ảnh):** `['camouflage_00001536.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Seahorse_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Seahorse_3shot_split1.json` (1 ảnh):** `['camouflage_00001536.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Seahorse_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Seahorse_5shot_split1.json` (1 ảnh):** `['camouflage_00001536.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Seahorse_2shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Seahorse_3shot_split1.json` (2 ảnh):** `['camouflage_00001536.jpg', 'camouflage_00001537.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Seahorse_2shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Seahorse_5shot_split1.json` (2 ảnh):** `['camouflage_00001536.jpg', 'camouflage_00001537.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Seahorse_3shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Seahorse_5shot_split1.json` (3 ảnh):** `['camouflage_00001536.jpg', 'camouflage_00001537.jpg', 'camouflage_00001538.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Shrimp_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Shrimp_2shot_split1.json` (1 ảnh):** `['camouflage_00001655.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Shrimp_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Shrimp_3shot_split1.json` (1 ảnh):** `['camouflage_00001655.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Shrimp_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Shrimp_5shot_split1.json` (1 ảnh):** `['camouflage_00001655.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Shrimp_2shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Shrimp_3shot_split1.json` (2 ảnh):** `['camouflage_00001655.jpg', 'camouflage_00001656.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Shrimp_2shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Shrimp_5shot_split1.json` (2 ảnh):** `['camouflage_00001655.jpg', 'camouflage_00001656.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Shrimp_3shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Shrimp_5shot_split1.json` (3 ảnh):** `['camouflage_00001655.jpg', 'camouflage_00001656.jpg', 'camouflage_00001657.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Sloth_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Sloth_2shot_split1.json` (1 ảnh):** `['camouflage_00000523.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Sloth_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Sloth_3shot_split1.json` (1 ảnh):** `['camouflage_00000523.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Sloth_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Sloth_5shot_split1.json` (1 ảnh):** `['camouflage_00000523.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Sloth_2shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Sloth_3shot_split1.json` (2 ảnh):** `['camouflage_00000523.jpg', 'camouflage_00000525.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Sloth_2shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Sloth_5shot_split1.json` (2 ảnh):** `['camouflage_00000523.jpg', 'camouflage_00000525.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Sloth_3shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Sloth_5shot_split1.json` (3 ảnh):** `['camouflage_00000523.jpg', 'camouflage_00000525.jpg', 'camouflage_00000526.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Snail_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Snail_2shot_split1.json` (1 ảnh):** `['camouflage_00002080.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Snail_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Snail_3shot_split1.json` (1 ảnh):** `['camouflage_00002080.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Snail_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Snail_5shot_split1.json` (1 ảnh):** `['camouflage_00002080.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Snail_2shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Snail_3shot_split1.json` (2 ảnh):** `['camouflage_00002080.jpg', 'camouflage_00002082.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Snail_2shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Snail_5shot_split1.json` (2 ảnh):** `['camouflage_00002080.jpg', 'camouflage_00002082.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Snail_3shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Snail_5shot_split1.json` (3 ảnh):** `['camouflage_00002080.jpg', 'camouflage_00002082.jpg', 'camouflage_00002083.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Snake_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Snake_2shot_split1.json` (1 ảnh):** `['camouflage_00001316.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Snake_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Snake_3shot_split1.json` (1 ảnh):** `['camouflage_00001316.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Snake_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Snake_5shot_split1.json` (1 ảnh):** `['camouflage_00001316.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Snake_2shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Snake_3shot_split1.json` (2 ảnh):** `['camouflage_00001316.jpg', 'camouflage_00001317.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Snake_2shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Snake_5shot_split1.json` (2 ảnh):** `['camouflage_00001316.jpg', 'camouflage_00001317.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Snake_3shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Snake_5shot_split1.json` (3 ảnh):** `['camouflage_00001316.jpg', 'camouflage_00001317.jpg', 'camouflage_00001318.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Solider_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Solider_2shot_split1.json` (1 ảnh):** `['camouflage_00001934.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Solider_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Solider_3shot_split1.json` (1 ảnh):** `['camouflage_00001934.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Solider_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Solider_5shot_split1.json` (1 ảnh):** `['camouflage_00001934.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Solider_2shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Solider_3shot_split1.json` (2 ảnh):** `['camouflage_00001909.jpg', 'camouflage_00001934.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Solider_2shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Solider_5shot_split1.json` (2 ảnh):** `['camouflage_00001909.jpg', 'camouflage_00001934.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Solider_3shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Solider_5shot_split1.json` (3 ảnh):** `['camouflage_00001909.jpg', 'camouflage_00001934.jpg', 'camouflage_00001935.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Spider_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Spider_2shot_split1.json` (1 ảnh):** `['camouflage_00002606.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Spider_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Spider_3shot_split1.json` (1 ảnh):** `['camouflage_00002606.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Spider_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Spider_5shot_split1.json` (1 ảnh):** `['camouflage_00002606.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Spider_2shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Spider_3shot_split1.json` (2 ảnh):** `['camouflage_00002606.jpg', 'camouflage_00002607.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Spider_2shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Spider_5shot_split1.json` (2 ảnh):** `['camouflage_00002606.jpg', 'camouflage_00002607.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Spider_3shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Spider_5shot_split1.json` (3 ảnh):** `['camouflage_00002606.jpg', 'camouflage_00002607.jpg', 'camouflage_00002608.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Squirrel_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Squirrel_2shot_split1.json` (1 ảnh):** `['camouflage_00000517.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Squirrel_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Squirrel_3shot_split1.json` (1 ảnh):** `['camouflage_00000517.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Squirrel_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Squirrel_5shot_split1.json` (1 ảnh):** `['camouflage_00000517.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Squirrel_2shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Squirrel_3shot_split1.json` (2 ảnh):** `['camouflage_00000517.jpg', 'camouflage_00000518.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Squirrel_2shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Squirrel_5shot_split1.json` (2 ảnh):** `['camouflage_00000517.jpg', 'camouflage_00000518.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Squirrel_3shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Squirrel_5shot_split1.json` (3 ảnh):** `['camouflage_00000517.jpg', 'camouflage_00000518.jpg', 'camouflage_00000519.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Starfish_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Starfish_2shot_split1.json` (1 ảnh):** `['camouflage_00001664.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Starfish_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Starfish_3shot_split1.json` (1 ảnh):** `['camouflage_00001664.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Starfish_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Starfish_5shot_split1.json` (1 ảnh):** `['camouflage_00001664.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Starfish_2shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Starfish_3shot_split1.json` (2 ảnh):** `['camouflage_00001664.jpg', 'camouflage_00001665.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Starfish_2shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Starfish_5shot_split1.json` (2 ảnh):** `['camouflage_00001664.jpg', 'camouflage_00001665.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Starfish_3shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Starfish_5shot_split1.json` (3 ảnh):** `['camouflage_00001664.jpg', 'camouflage_00001665.jpg', 'camouflage_00001666.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Tiger_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Tiger_2shot_split1.json` (1 ảnh):** `['camouflage_00001282.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Tiger_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Tiger_3shot_split1.json` (1 ảnh):** `['camouflage_00001282.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Tiger_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Tiger_5shot_split1.json` (1 ảnh):** `['camouflage_00001282.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Tiger_2shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Tiger_3shot_split1.json` (2 ảnh):** `['camouflage_00001282.jpg', 'camouflage_00001283.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Tiger_2shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Tiger_5shot_split1.json` (2 ảnh):** `['camouflage_00001282.jpg', 'camouflage_00001283.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Tiger_3shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Tiger_5shot_split1.json` (3 ảnh):** `['camouflage_00001282.jpg', 'camouflage_00001283.jpg', 'camouflage_00001285.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Turtle_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Turtle_2shot_split1.json` (1 ảnh):** `['camouflage_00001482.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Turtle_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Turtle_3shot_split1.json` (1 ảnh):** `['camouflage_00001482.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Turtle_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Turtle_5shot_split1.json` (1 ảnh):** `['camouflage_00001482.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Turtle_2shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Turtle_3shot_split1.json` (2 ảnh):** `['camouflage_00001482.jpg', 'camouflage_00001483.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Turtle_2shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Turtle_5shot_split1.json` (2 ảnh):** `['camouflage_00001482.jpg', 'camouflage_00001483.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Turtle_3shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Turtle_5shot_split1.json` (3 ảnh):** `['camouflage_00001482.jpg', 'camouflage_00001483.jpg', 'camouflage_00001484.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Weasel_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Weasel_2shot_split1.json` (1 ảnh):** `['camouflage_00001084.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Weasel_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Weasel_3shot_split1.json` (1 ảnh):** `['camouflage_00001084.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Weasel_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Weasel_5shot_split1.json` (1 ảnh):** `['camouflage_00001084.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Weasel_2shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Weasel_3shot_split1.json` (2 ảnh):** `['camouflage_00001084.jpg', 'camouflage_00001085.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Weasel_2shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Weasel_5shot_split1.json` (2 ảnh):** `['camouflage_00001084.jpg', 'camouflage_00001085.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Weasel_3shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Weasel_5shot_split1.json` (3 ảnh):** `['camouflage_00001084.jpg', 'camouflage_00001085.jpg', 'camouflage_00001086.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Worm_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Worm_2shot_split1.json` (1 ảnh):** `['camouflage_00000319.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Worm_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Worm_3shot_split1.json` (1 ảnh):** `['camouflage_00000319.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Worm_1shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Worm_5shot_split1.json` (1 ảnh):** `['camouflage_00000319.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Worm_2shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Worm_3shot_split1.json` (2 ảnh):** `['camouflage_00000319.jpg', 'camouflage_00000320.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Worm_2shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Worm_5shot_split1.json` (2 ảnh):** `['camouflage_00000319.jpg', 'camouflage_00000320.jpg']`
- **Ảnh xuất hiện trùng ở cả 2 file `few-shot-annotations\subsplit\split1\camo5_Worm_3shot_split1.json ∩ few-shot-annotations\subsplit\split1\camo5_Worm_5shot_split1.json` (3 ảnh):** `['camouflage_00000319.jpg', 'camouflage_00000320.jpg', 'camouflage_00000321.jpg']`

## 3. Phân nhóm 47 Lớp CAMO-FS so với 80 Lớp MS-COCO (`yolov8n.pt`)
- **Nhóm 1 — Trùng tên trực tiếp hoặc từ khóa với COCO (8 lớp):**
  - `[yolo_id=10]` **Bird** (khớp `bird` - kiểu `exact`)
  - `[yolo_id=16]` **Bear** (khớp `bear` - kiểu `exact`)
  - `[yolo_id=18]` **Cat** (khớp `cat` - kiểu `exact`)
  - `[yolo_id=20]` **Dog** (khớp `dog` - kiểu `exact`)
  - `[yolo_id=21]` **Elephant** (khớp `elephant` - kiểu `exact`)
  - `[yolo_id=23]` **Giraffe** (khớp `giraffe` - kiểu `exact`)
  - `[yolo_id=26]` **Horse** (khớp `horse` - kiểu `exact`)
  - `[yolo_id=31]` **Mouse** (khớp `mouse` - kiểu `exact`)
- **Nhóm 2 — Cùng họ động vật lớn với COCO (7 lớp, tùy chọn tách riêng):**
  - `[yolo_id=19]` **Deer** (họ `sheep/cow/horse (ungulate)`)
  - `[yolo_id=22]` **Fox** (họ `dog`)
  - `[yolo_id=24]` **Goat** (họ `sheep`)
  - `[yolo_id=28]` **Leopard** (họ `cat`)
  - `[yolo_id=29]` **Lion** (họ `cat`)
  - `[yolo_id=35]` **Sea_Lion** (họ `cat`)
  - `[yolo_id=38]` **Tiger** (họ `cat`)
- **Nhóm 3 — Lớp Novel hoàn toàn không có trong COCO (32 lớp):**
  `Frog`, `Crab`, `Dolphin`, `Fish`, `Octopus`, `Seahorse`, `Shrimp`, `Starfish`, `Scorpion`, `Spider`, `Penguin`, `Snail`, `Insect`, `Moth`, `Bat`, `Camel`, `Hedgehog`, `Kangaroo`, `Monkey`, `Pig`, `Rabbit`, `Rhino`, `Sloth`, `Squirrel`, `Weasel`, `Body_Painting`, `Solider`, `Crocodile`, `Lizard`, `Snake`, `Turtle`, `Worm`
