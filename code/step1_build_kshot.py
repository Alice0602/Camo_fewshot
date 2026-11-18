#!/usr/bin/env python3
"""
Bước 1: Xây dựng Pipeline dữ liệu k-shot lồng nhau (1 ⊂ 2 ⊂ 3 ⊂ 5) & Tô xám pixel=114 an toàn.
- Chống âm tính giả (Anti-False-Negative): Tô xám (RGB = 114, 114, 114) các instance không được chọn
  trong cùng một bức ảnh mẫu, đồng thời KHÔI PHỤC lại vùng pixel của các instance được chọn nếu
  2 bounding box giao nhau (IoU > 0).
- Sinh nhãn định dạng chuẩn YOLO .txt và các file cấu hình YAML cho từng mức k và seed.
"""

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def mask_unselected_instances_safe(
    img: np.ndarray,
    selected_bboxes: List[List[float]],
    unselected_bboxes: List[List[float]],
    fill_value: int = 114,
) -> np.ndarray:
    """
    Tô xám 2 pha an toàn:
      - Pha 1: Tô xám (pixel = 114) toàn bộ các bounding box KHÔNG được chọn.
      - Pha 2: Đắp lại vùng pixel gốc của các bounding box ĐƯỢC CHỌN để chống cắt lẹm
               khi 2 con vật đứng chồng/giao nhau (overlapping boxes).
    Bbox đầu vào theo chuẩn COCO: [x_min, y_min, width, height].
    """
    if not unselected_bboxes:
        return img

    h_img, w_img = img.shape[:2]
    masked = img.copy()

    # Pha 1: Tô xám các instance thừa
    for box in unselected_bboxes:
        x, y, w, h = box
        x1 = max(0, int(round(x)))
        y1 = max(0, int(round(y)))
        x2 = min(w_img, int(round(x + w)))
        y2 = min(h_img, int(round(y + h)))
        if x2 > x1 and y2 > y1:
            masked[y1:y2, x1:x2] = fill_value

    # Pha 2: Khôi phục nguyên vẹn vùng pixel của các instance được chọn
    for box in selected_bboxes:
        x, y, w, h = box
        x1 = max(0, int(round(x)))
        y1 = max(0, int(round(y)))
        x2 = min(w_img, int(round(x + w)))
        y2 = min(h_img, int(round(y + h)))
        if x2 > x1 and y2 > y1:
            masked[y1:y2, x1:x2] = img[y1:y2, x1:x2]

    return masked


def coco_bbox_to_yolo(bbox: List[float], img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    """Chuyển [x_min, y_min, w, h] sang [x_center, y_center, w, h] chuẩn hóa trong [0, 1]."""
    x, y, w, h = bbox
    x1 = max(0.0, min(float(img_w), float(x)))
    y1 = max(0.0, min(float(img_h), float(y)))
    x2 = max(0.0, min(float(img_w), float(x + w)))
    y2 = max(0.0, min(float(img_h), float(y + h)))

    bw = max(1e-6, (x2 - x1) / float(img_w))
    bh = max(1e-6, (y2 - y1) / float(img_h))
    xc = max(0.0, min(1.0, ((x1 + x2) / 2.0) / float(img_w)))
    yc = max(0.0, min(1.0, ((y1 + y2) / 2.0) / float(img_h)))
    return xc, yc, bw, bh


def index_disk_images(data_root: Path) -> Dict[str, Path]:
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    mapping: Dict[str, Path] = {}
    for p in data_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in img_exts:
            mapping[p.name] = p
            mapping[str(p.relative_to(data_root)).replace("\\", "/")] = p
    return mapping


def resolve_image_path(file_name: str, img_index: Dict[str, Path], data_root: Path) -> Path:
    direct = data_root / file_name
    if direct.exists():
        return direct
    norm_key = file_name.replace("\\", "/")
    if norm_key in img_index:
        return img_index[norm_key]
    base_name = Path(file_name).name
    if base_name in img_index:
        return img_index[base_name]
    raise FileNotFoundError(f"Không tìm thấy file ảnh '{file_name}' trong {data_root}")


def write_yolo_yaml(yaml_path: Path, dataset_root: Path, train_rel: str, val_rel: str, test_rel: str, names: Dict[int, str]):
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"path: {dataset_root.resolve().as_posix()}",
        f"train: {train_rel}",
        f"val: {val_rel}",
        f"test: {test_rel}",
        f"nc: {len(names)}",
        "names:",
    ]
    for idx in sorted(names.keys()):
        safe_name = str(names[idx]).replace("'", "")
        lines.append(f"  {idx}: '{safe_name}'")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def build_pipeline(
    data_root: Path,
    support_json: Path,
    test_json: Path,
    out_dir: Path,
    config_dir: Path,
    k_shots: List[int],
    seeds: List[int],
) -> Dict:
    with open(support_json, "r", encoding="utf-8") as f:
        sup_data = json.load(f)
    with open(test_json, "r", encoding="utf-8") as f:
        tst_data = json.load(f)

    img_index = index_disk_images(data_root)

    # Ánh xạ category_id -> yolo_id (0 .. nc-1)
    sorted_cats = sorted(sup_data["categories"], key=lambda c: c["id"])
    cat_id_to_yolo = {c["id"]: i for i, c in enumerate(sorted_cats)}
    yolo_names = {i: c["name"] for i, c in enumerate(sorted_cats)}

    # 1. Xuất tập Test chuẩn (chỉ làm 1 lần, không bao giờ trùng ảnh mẫu)
    test_img_dir = out_dir / "test" / "images"
    test_lbl_dir = out_dir / "test" / "labels"
    test_img_dir.mkdir(parents=True, exist_ok=True)
    test_lbl_dir.mkdir(parents=True, exist_ok=True)

    tst_img_map = {img["id"]: img for img in tst_data["images"]}
    tst_anns_by_img = defaultdict(list)
    for ann in tst_data["annotations"]:
        if ann.get("iscrowd", 0) == 1:
            continue
        tst_anns_by_img[ann["image_id"]].append(ann)

    sup_filenames = {Path(img["file_name"]).name for img in sup_data["images"]}
    leaked_test_imgs = 0

    for img_id, img_info in tst_img_map.items():
        fname = Path(img_info["file_name"]).name
        if fname in sup_filenames:
            leaked_test_imgs += 1
            continue
        src_path = resolve_image_path(img_info["file_name"], img_index, data_root)
        dst_img_path = test_img_dir / fname
        if not dst_img_path.exists():
            shutil.copy2(src_path, dst_img_path)

        w_img, h_img = int(img_info["width"]), int(img_info["height"])
        lbl_lines = []
        for ann in tst_anns_by_img[img_id]:
            y_id = cat_id_to_yolo[ann["category_id"]]
            xc, yc, bw, bh = coco_bbox_to_yolo(ann["bbox"], w_img, h_img)
            lbl_lines.append(f"{y_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        lbl_path = test_lbl_dir / f"{Path(fname).stem}.txt"
        with open(lbl_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lbl_lines) + ("\n" if lbl_lines else ""))

    test_yaml = config_dir / "camo_test.yaml"
    write_yolo_yaml(test_yaml, out_dir, "test/images", "test/images", "test/images", yolo_names)

    # 2. Chuẩn bị cấu trúc tập Support (5 instances / lớp)
    sup_img_map = {img["id"]: img for img in sup_data["images"]}
    sup_anns_by_cat = defaultdict(list)
    sup_anns_by_img = defaultdict(list)
    for ann in sup_data["annotations"]:
        if ann.get("iscrowd", 0) == 1:
            continue
        sup_anns_by_cat[ann["category_id"]].append(ann)
        sup_anns_by_img[ann["image_id"]].append(ann)

    # Đảm bảo thứ tự gốc cố định theo official_rank trước khi shuffle theo seed
    for cid in sup_anns_by_cat:
        sup_anns_by_cat[cid].sort(key=lambda a: (a.get("official_rank", 5), a["id"]))

    summary_stats = {
        "num_classes": len(sorted_cats),
        "leaked_test_imgs_removed": leaked_test_imgs,
        "splits": {},
    }

    # 3. Sinh các tập k-shot lồng nhau (1 ⊂ 2 ⊂ 3 ⊂ 5) cho từng seed
    for seed in seeds:
        # Với seed=0: Giữ nguyên thứ tự split1 chính thức của bài báo FS-CDIS; với seed>0: tạo hoán vị ngẫu nhiên cố định
        permuted_by_cat: Dict[int, List[Dict]] = {}
        for cid, anns in sup_anns_by_cat.items():
            anns_copy = list(anns)
            if seed > 0:
                rng = random.Random(seed * 10007 + cid)
                rng.shuffle(anns_copy)
            permuted_by_cat[cid] = anns_copy

        prev_selected_ids: Optional[set] = None
        for k in sorted(k_shots):
            split_name = f"k{k}_seed{seed}"
            sp_img_dir = out_dir / split_name / "images"
            sp_lbl_dir = out_dir / split_name / "labels"
            sp_img_dir.mkdir(parents=True, exist_ok=True)
            sp_lbl_dir.mkdir(parents=True, exist_ok=True)

            # Chọn đúng k instances đầu tiên của hoán vị -> Đảm bảo 1-shot ⊂ 2-shot ⊂ 3-shot ⊂ 5-shot
            selected_ann_ids = set()
            for cid, perm_anns in permuted_by_cat.items():
                for ann in perm_anns[:k]:
                    selected_ann_ids.add(ann["id"])

            if prev_selected_ids is not None:
                assert prev_selected_ids.issubset(selected_ann_ids), "Lỗi: Vi phạm tính chất lồng nhau!"
            prev_selected_ids = set(selected_ann_ids)

            # Tìm các bức ảnh có ít nhất 1 instance được chọn
            active_img_ids = sorted({
                ann["image_id"]
                for cid, perm_anns in permuted_by_cat.items()
                for ann in perm_anns[:k]
            })

            masked_instances_count = 0
            for img_id in active_img_ids:
                img_info = sup_img_map[img_id]
                fname = Path(img_info["file_name"]).name
                src_path = resolve_image_path(img_info["file_name"], img_index, data_root)
                img = cv2.imread(str(src_path))
                if img is None:
                    raise RuntimeError(f"Không thể đọc ảnh: {src_path}")

                h_img, w_img = img.shape[:2]
                all_img_anns = sup_anns_by_img[img_id]
                sel_anns = [a for a in all_img_anns if a["id"] in selected_ann_ids]
                unsel_anns = [a for a in all_img_anns if a["id"] not in selected_ann_ids]

                masked_instances_count += len(unsel_anns)
                sel_boxes = [a["bbox"] for a in sel_anns]
                unsel_boxes = [a["bbox"] for a in unsel_anns]

                # Tô xám pixel=114 cho instance không được chọn & bảo vệ vùng giao với instance được chọn
                out_img = mask_unselected_instances_safe(img, sel_boxes, unsel_boxes, fill_value=114)
                cv2.imwrite(str(sp_img_dir / fname), out_img)

                lbl_lines = []
                for a in sel_anns:
                    y_id = cat_id_to_yolo[a["category_id"]]
                    xc, yc, bw, bh = coco_bbox_to_yolo(a["bbox"], w_img, h_img)
                    lbl_lines.append(f"{y_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

                with open(sp_lbl_dir / f"{Path(fname).stem}.txt", "w", encoding="utf-8") as f:
                    f.write("\n".join(lbl_lines) + "\n")

            yaml_file = config_dir / f"camo_k{k}_seed{seed}.yaml"
            # Lưu ý: val trỏ vào chính tập support để tuyệt đối không chạm tập test trong lúc train
            write_yolo_yaml(
                yaml_file,
                out_dir,
                f"{split_name}/images",
                f"{split_name}/images",
                "test/images",
                yolo_names,
            )

            summary_stats["splits"][split_name] = {
                "k": k,
                "seed": seed,
                "selected_instances": len(selected_ann_ids),
                "support_images": len(active_img_ids),
                "masked_unselected_instances": masked_instances_count,
                "yaml": str(yaml_file),
            }
            print(
                f"[OK] {split_name}: {len(selected_ann_ids)} instances trên {len(active_img_ids)} ảnh "
                f"(Đã tô xám 114 cho {masked_instances_count} instances phụ)"
            )

    return summary_stats


def merge_official_subsplits(subsplit_dir: Path) -> Dict:
    """
    Hợp nhất 47 file camo5_<Class>_5shot_split1.json chính thức của tác giả thành 1 cấu trúc COCO duy nhất,
    đồng thời gán lại ann['id'] duy nhất (1..235) để sửa lỗi trùng ID [826, 386, 387] giữa Fish/Moth và Lizard/Snake,
    và đánh dấu thứ tự ưu tiên chính thức (official_rank: 1, 2, 3, 5) cho seed=0.
    """
    def collect_bbox_keys(k: int) -> set:
        keys = set()
        for f in sorted(subsplit_dir.glob(f"*_{k}shot_split1.json")):
            with open(f, "r", encoding="utf-8") as fp:
                d = json.load(fp)
            for a in d["annotations"]:
                keys.add((a["image_id"], a["category_id"], tuple(a["bbox"])))
        return keys

    keys_1 = collect_bbox_keys(1)
    keys_2 = collect_bbox_keys(2)
    keys_3 = collect_bbox_keys(3)

    categories = []
    images_by_id = {}
    merged_anns = []
    next_uid = 1

    for f in sorted(subsplit_dir.glob("*_5shot_split1.json")):
        with open(f, "r", encoding="utf-8") as fp:
            d = json.load(fp)
        if not categories and d.get("categories"):
            categories = d["categories"]
        for im in d.get("images", []):
            images_by_id[im["id"]] = im
        for a in d.get("annotations", []):
            key = (a["image_id"], a["category_id"], tuple(a["bbox"]))
            rank = 1 if key in keys_1 else (2 if key in keys_2 else (3 if key in keys_3 else 5))
            a_copy = dict(a)
            a_copy["orig_id"] = a["id"]
            a_copy["id"] = next_uid
            a_copy["official_rank"] = rank
            next_uid += 1
            merged_anns.append(a_copy)

    return {
        "categories": categories,
        "images": list(images_by_id.values()),
        "annotations": merged_anns,
    }


def auto_discover_support_and_test_jsons(data_root: Path) -> Tuple[Path, Path]:
    """Tự động nhận diện hoặc hợp nhất file Support JSON (235 instances) và Test JSON (3108 instances) trong data_root."""
    subsplit_dir = data_root / "few-shot-annotations" / "subsplit" / "split1"
    test_split1 = data_root / "few-shot-annotations" / "camo5_test_split1.json"
    if subsplit_dir.exists() and test_split1.exists():
        merged_sup_path = Path("d:/ifs_workspace/experiments/camo_yolo_pilot/results/camo5_support_5shot_merged.json")
        merged_sup_path.parent.mkdir(parents=True, exist_ok=True)
        merged_data = merge_official_subsplits(subsplit_dir)
        with open(merged_sup_path, "w", encoding="utf-8") as f:
            json.dump(merged_data, f, indent=2)
        print(
            f"[AUTO-MERGE] Đã hợp nhất 47 file subsplit chính thức -> {len(merged_data['annotations'])} instances "
            f"trên {len(merged_data['images'])} ảnh tại: {merged_sup_path}"
        )
        return merged_sup_path, test_split1

    candidates = []
    for jp in sorted(data_root.rglob("*.json")):
        if jp.name.startswith("class_overlap") or "subsplit" in str(jp):
            continue
        try:
            with open(jp, "r", encoding="utf-8") as f:
                d = json.load(f)
            if "images" in d and "annotations" in d and "categories" in d:
                candidates.append((len(d["annotations"]), jp))
        except Exception:
            continue

    if len(candidates) < 2:
        raise FileNotFoundError(
            f"Cần ít nhất 2 file COCO JSON (Support & Test) trong {data_root}, tìm thấy: {[str(c[1]) for c in candidates]}"
        )
    candidates.sort(key=lambda x: x[0])
    sup_jp = candidates[0][1]
    tst_jp = candidates[-1][1]
    print(f"[AUTO-DETECT] Support JSON ({candidates[0][0]} instances): {sup_jp}")
    print(f"[AUTO-DETECT] Test JSON    ({candidates[-1][0]} instances): {tst_jp}")
    return sup_jp, tst_jp


def main():
    parser = argparse.ArgumentParser(description="Bước 1: Sinh tập k-shot lồng nhau & tô xám 114")
    parser.add_argument("--data-root", type=str, required=True, help="Thư mục gốc chứa ảnh CAMO-FS")
    parser.add_argument("--support-json", type=str, default="", help="File COCO JSON của tập Support (để trống sẽ tự dò)")
    parser.add_argument("--test-json", type=str, default="", help="File COCO JSON của tập Test (để trống sẽ tự dò)")
    parser.add_argument("--out-dir", type=str, default="data/processed", help="Thư mục xuất ảnh và nhãn YOLO")
    parser.add_argument("--config-dir", type=str, default="configs", help="Thư mục xuất các file .yaml")
    parser.add_argument("--k-shots", type=int, nargs="+", default=[1, 2, 3, 5])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()

    data_root = Path(args.data_root)
    if args.support_json and args.test_json:
        sup_json, tst_json = Path(args.support_json), Path(args.test_json)
    else:
        sup_json, tst_json = auto_discover_support_and_test_jsons(data_root)

    build_pipeline(
        data_root,
        sup_json,
        tst_json,
        Path(args.out_dir),
        Path(args.config_dir),
        args.k_shots,
        args.seeds,
    )


if __name__ == "__main__":
    main()
