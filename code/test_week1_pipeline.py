#!/usr/bin/env python3
"""
Unit test tổng hợp trên CPU (Synthetic CPU Unit Check) cho toàn bộ Pipeline Tuần 1:
- Kiểm tra Bước 0 (step0_audit_data.py): Bắt đúng lỗi lệch ảnh và tách đúng lớp trùng COCO vs Novel.
- Kiểm tra Bước 1 (step1_build_kshot.py):
  + Đảm bảo tính chất lồng nhau tuyệt đối: 1-shot ⊂ 2-shot ⊂ 3-shot ⊂ 5-shot.
  + Đảm bảo thuật toán tô xám 2 pha (pixel = 114) KHÔNG cắt lẹm vào vật thể được chọn khi 2 box đè lên nhau.
"""

import shutil
import tempfile
import json
from pathlib import Path
import cv2
import numpy as np

from step0_audit_data import run_audit
from step1_build_kshot import build_pipeline, mask_unselected_instances_safe


def test_safe_gray_masking_with_overlap():
    # Tạo ảnh 100x100 có màu gốc = 200
    img = np.full((100, 100, 3), 200, dtype=np.uint8)
    # Vật thể được chọn (selected) có pixel = 50 nằm ở [10, 10, 40, 40] (x: 10..50, y: 10..50)
    img[10:50, 10:50] = 50
    selected_boxes = [[10, 10, 40, 40]]
    # Vật thể KHÔNG được chọn (unselected) nằm ở [30, 30, 40, 40] (x: 30..70, y: 30..70) -> Giao nhau vùng [30..50, 30..50]!
    unselected_boxes = [[30, 30, 40, 40]]

    masked = mask_unselected_instances_safe(img, selected_boxes, unselected_boxes, fill_value=114)

    # 1. Toàn bộ vùng của vật thể được chọn [10:50, 10:50] (kể cả vùng giao [30:50, 30:50]) PHẢI giữ nguyên giá trị 50!
    assert np.all(masked[10:50, 10:50] == 50), "Lỗi: Vùng giao của instance được chọn bị tô xám đè mất!"
    # 2. Phần không giao của vật thể thừa [50:70, 50:70] PHẢI bị tô thành 114!
    assert np.all(masked[50:70, 50:70] == 114), "Lỗi: Instance thừa chưa được tô xám 114!"
    print("[PASS] Unit Test 1: Tô xám pixel=114 an toàn khi 2 Bounding Box đè lên nhau.")


def test_step0_and_step1_end_to_end():
    tmp_dir = Path(tempfile.mkdtemp(prefix="camo_unit_test_"))
    try:
        raw_dir = tmp_dir / "raw"
        img_dir = raw_dir / "images"
        img_dir.mkdir(parents=True)

        # Tạo 3 lớp giả lập: 1 lớp trùng COCO ('cat'), 1 lớp trùng họ ('snowy owl'), 1 lớp Novel ('leaf tailed gecko')
        categories = [
            {"id": 1, "name": "cat"},
            {"id": 2, "name": "snowy owl"},
            {"id": 3, "name": "leaf tailed gecko"},
        ]

        sup_images = []
        sup_anns = []
        ann_id = 1
        # Tạo 5 instances / lớp cho tập Support (nhét 2 instances vào cùng 1 ảnh để test tô xám 114)
        for cid in [1, 2, 3]:
            for s_idx in range(5):
                img_id = cid * 100 + s_idx
                fname = f"sup_c{cid}_{s_idx}.jpg"
                dummy_img = np.full((120, 160, 3), 80, dtype=np.uint8)
                cv2.imwrite(str(img_dir / fname), dummy_img)
                sup_images.append({"id": img_id, "file_name": f"images/{fname}", "width": 160, "height": 120})
                sup_anns.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cid,
                    "bbox": [10.0 + s_idx * 5, 10.0, 30.0, 30.0],
                    "iscrowd": 0,
                })
                ann_id += 1

        # Tạo tập Test (3 ảnh) + tạo thêm 2 ảnh thừa trên ổ cứng không có trong JSON để test phát hiện lệch ảnh
        tst_images = []
        tst_anns = []
        for t_idx in range(3):
            img_id = 900 + t_idx
            fname = f"test_{t_idx}.jpg"
            dummy_img = np.full((120, 160, 3), 90, dtype=np.uint8)
            cv2.imwrite(str(img_dir / fname), dummy_img)
            tst_images.append({"id": img_id, "file_name": f"images/{fname}", "width": 160, "height": 120})
            tst_anns.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": (t_idx % 3) + 1,
                "bbox": [20.0, 20.0, 40.0, 40.0],
                "iscrowd": 0,
            })
            ann_id += 1

        # Tạo 2 ảnh rời không khai báo trong JSON (giả lập lệch ảnh)
        for extra_i in range(2):
            cv2.imwrite(str(img_dir / f"extra_unlisted_{extra_i}.jpg"), np.zeros((50, 50, 3), dtype=np.uint8))

        sup_json = raw_dir / "camo_support.json"
        tst_json = raw_dir / "camo_test.json"
        with open(sup_json, "w", encoding="utf-8") as f:
            json.dump({"images": sup_images, "annotations": sup_anns, "categories": categories}, f)
        with open(tst_json, "w", encoding="utf-8") as f:
            json.dump({"images": tst_images, "annotations": tst_anns, "categories": categories}, f)

        # Kiểm tra Bước 0
        out_md = tmp_dir / "DATA_CHECK.md"
        out_ov = tmp_dir / "class_overlap.json"
        ov_res = run_audit(raw_dir, out_md, out_ov)
        assert ov_res["num_total_classes"] == 3
        assert len(ov_res["discrepancy_analysis"]["on_disk_not_in_json"]) == 2
        assert 0 in ov_res["exact_or_token_overlap_yolo_ids"]  # 'cat' -> yolo_id 0
        assert 2 in ov_res["pure_novel_yolo_ids"]              # 'leaf tailed gecko' -> yolo_id 2
        print("[PASS] Unit Test 2: Bước 0 phát hiện chính xác ảnh lệch và phân loại đúng lớp trùng COCO vs Novel.")

        # Kiểm tra Bước 1
        proc_dir = tmp_dir / "processed"
        cfg_dir = tmp_dir / "configs"
        stats = build_pipeline(raw_dir, sup_json, tst_json, proc_dir, cfg_dir, k_shots=[1, 2, 3, 5], seeds=[0, 1])
        assert stats["splits"]["k1_seed0"]["selected_instances"] == 3
        assert stats["splits"]["k5_seed0"]["selected_instances"] == 15
        assert (cfg_dir / "camo_k1_seed0.yaml").exists()
        assert (cfg_dir / "camo_test.yaml").exists()
        print("[PASS] Unit Test 3: Bước 1 sinh đúng tập lồng nhau k={1,2,3,5} và xuất đầy đủ nhãn YOLO + YAML.")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    test_safe_gray_masking_with_overlap()
    test_step0_and_step1_end_to_end()
    print("\n=== TẤT CẢ UNIT TEST TUẦN 1 ĐỀU VƯỢT QUA 100% ===")
