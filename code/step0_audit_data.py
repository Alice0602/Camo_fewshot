#!/usr/bin/env python3
"""
Bước 0: Kiểm tra và đối chiếu toàn diện bộ dữ liệu CAMO-FS (Data Audit & COCO Overlap).
- Tự động quét cấu trúc thư mục và các file JSON (COCO format) trong --data-root.
- Đếm chính xác số lớp, số ảnh, số instances (Support vs Test) và tìm nguyên nhân lệch 6 ảnh.
- Đối chiếu 47 lớp CAMO-FS với 80 lớp COCO của YOLO để tách nhóm:
  (1) Nhóm trùng trực tiếp / từ khóa với COCO (COCO-Overlap)
  (2) Nhóm loài mới hoàn toàn (Pure Novel)
- Xuất báo cáo DATA_CHECK.md và file cấu hình class_overlap.json cho Bước 1-3.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

# 80 lớp chuẩn của MS-COCO (dùng trong yolov8n.pt / yolo11n.pt)
COCO_80_CLASSES: List[str] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]

# Từ điển ánh xạ ngữ nghĩa bổ trợ (các loài động vật thuộc họ/nhóm con trực tiếp của lớp động vật COCO)
COCO_ANIMAL_SYNONYMS: Dict[str, str] = {
    "owl": "bird",
    "heron": "bird",
    "bittern": "bird",
    "ptarmigan": "bird",
    "nightjar": "bird",
    "potoo": "bird",
    "woodcock": "bird",
    "plover": "bird",
    "pheasant": "bird",
    "quail": "bird",
    "grouse": "bird",
    "sandpiper": "bird",
    "snipe": "bird",
    "frogmouth": "bird",
    "kestrel": "bird",
    "hawk": "bird",
    "eagle": "bird",
    "falcon": "bird",
    "parrot": "bird",
    "sparrow": "bird",
    "pigeon": "bird",
    "duck": "bird",
    "goose": "bird",
    "leopard": "cat",
    "cheetah": "cat",
    "jaguar": "cat",
    "tiger": "cat",
    "lion": "cat",
    "lynx": "cat",
    "puma": "cat",
    "cougar": "cat",
    "bobcat": "cat",
    "wildcat": "cat",
    "fox": "dog",
    "wolf": "dog",
    "coyote": "dog",
    "deer": "sheep/cow/horse (ungulate)",
    "antelope": "sheep/cow/horse (ungulate)",
    "ibex": "sheep",
    "goat": "sheep",
}


def normalize_name(name: str) -> str:
    return name.lower().replace("_", " ").replace("-", " ").strip()


def classify_coco_overlap(cat_name: str) -> Tuple[bool, str, str]:
    """
    Phân loại một lớp của CAMO-FS xem có trùng với 80 lớp COCO hay không.
    Trả về: (is_overlap, match_type, matched_coco_class)
      - match_type: 'exact' (trùng tên trực tiếp), 'token' (chứa từ khóa lớp COCO),
                    'semantic_family' (thuộc họ động vật có trong COCO như owl->bird, leopard->cat),
                    'novel' (hoàn toàn không có trong COCO như stick insect, katydid, flounder, gecko...)
    """
    norm = normalize_name(cat_name)
    coco_set = {normalize_name(c) for c in COCO_80_CLASSES}

    if norm in coco_set:
        return True, "exact", norm

    tokens = norm.split()
    for t in tokens:
        if t in coco_set:
            return True, "token", t
        if t.endswith("s") and t[:-1] in coco_set:
            return True, "token", t[:-1]

    for t in tokens:
        if t in COCO_ANIMAL_SYNONYMS:
            return True, "semantic_family", COCO_ANIMAL_SYNONYMS[t]

    return False, "novel", ""


def inspect_coco_json(json_path: Path) -> Dict:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    images = data.get("images", [])
    annotations = data.get("annotations", [])
    categories = data.get("categories", [])

    img_id_to_info = {img["id"]: img for img in images}
    img_filenames = [Path(img.get("file_name", "")).name for img in images if img.get("file_name")]
    unique_filenames = set(img_filenames)

    ann_per_img = Counter(ann["image_id"] for ann in annotations)
    ann_per_cat = Counter(ann["category_id"] for ann in annotations)
    imgs_per_cat = defaultdict(set)
    for ann in annotations:
        imgs_per_cat[ann["category_id"]].add(ann["image_id"])

    imgs_without_anns = [
        img_id_to_info[iid].get("file_name", str(iid))
        for iid in img_id_to_info
        if ann_per_img[iid] == 0
    ]

    return {
        "path": str(json_path),
        "num_categories": len(categories),
        "num_images_entries": len(images),
        "num_unique_filenames": len(unique_filenames),
        "num_annotations": len(annotations),
        "imgs_without_anns": imgs_without_anns,
        "categories": categories,
        "ann_per_cat": dict(ann_per_cat),
        "imgs_per_cat": {k: len(v) for k, v in imgs_per_cat.items()},
        "image_filenames_set": unique_filenames,
        "raw_data": data,
    }


def run_audit(data_root: Path, out_md: Path, out_json: Path) -> Dict:
    if not data_root.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục dữ liệu: {data_root}")

    # 1. Tìm tất cả các file ảnh và file JSON trong data_root
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    all_disk_imgs = [
        p for p in data_root.rglob("*")
        if p.is_file() and p.suffix.lower() in img_exts
    ]
    disk_img_names = {p.name for p in all_disk_imgs}

    json_files = sorted([
        p for p in data_root.rglob("*.json")
        if p.is_file() and not p.name.startswith("class_overlap")
    ])

    json_summaries = []
    all_categories = {}
    json_union_filenames: Set[str] = set()
    json_file_sets: Dict[str, Set[str]] = {}

    for jp in json_files:
        try:
            info = inspect_coco_json(jp)
            if info["num_categories"] > 0 or info["num_images_entries"] > 0:
                json_summaries.append(info)
                rel_name = str(jp.relative_to(data_root))
                json_file_sets[rel_name] = info["image_filenames_set"]
                json_union_filenames |= info["image_filenames_set"]
                for cat in info["categories"]:
                    all_categories[cat["id"]] = cat["name"]
        except Exception as e:
            print(f"[WARN] Bỏ qua file JSON không phải định dạng COCO ({jp.name}): {e}")

    # 2. Phân tích chênh lệch số lượng ảnh (tìm nguyên nhân lệch 6 ảnh: 197 + 2655 = 2852 vs 2858)
    overlap_between_jsons = {}
    json_names = list(json_file_sets.keys())
    for i in range(len(json_names)):
        for j in range(i + 1, len(json_names)):
            n1, n2 = json_names[i], json_names[j]
            shared = sorted(list(json_file_sets[n1] & json_file_sets[n2]))
            if shared:
                overlap_between_jsons[f"{n1} ∩ {n2}"] = shared

    on_disk_not_in_json = sorted(list(disk_img_names - json_union_filenames))
    in_json_not_on_disk = sorted(list(json_union_filenames - disk_img_names))

    # 3. Phân tích 47 lớp CAMO-FS vs 80 lớp COCO
    sorted_cats = sorted(all_categories.items(), key=lambda x: x[0])
    cat_id_to_yolo_idx = {cat_id: idx for idx, (cat_id, _) in enumerate(sorted_cats)}

    exact_or_token_overlap = []
    semantic_family_overlap = []
    pure_novel = []

    class_records = []
    for cat_id, cat_name in sorted_cats:
        yolo_idx = cat_id_to_yolo_idx[cat_id]
        is_ov, mtype, matched = classify_coco_overlap(cat_name)
        rec = {
            "category_id": cat_id,
            "yolo_id": yolo_idx,
            "name": cat_name,
            "match_type": mtype,
            "matched_coco": matched,
        }
        class_records.append(rec)
        if mtype in ("exact", "token"):
            exact_or_token_overlap.append(rec)
        elif mtype == "semantic_family":
            semantic_family_overlap.append(rec)
        else:
            pure_novel.append(rec)

    overlap_payload = {
        "num_total_classes": len(sorted_cats),
        "exact_or_token_overlap_yolo_ids": [r["yolo_id"] for r in exact_or_token_overlap],
        "All_overlap_including_semantic_yolo_ids": [
            r["yolo_id"] for r in (exact_or_token_overlap + semantic_family_overlap)
        ],
        "pure_novel_yolo_ids": [r["yolo_id"] for r in pure_novel],
        "classes": class_records,
        "discrepancy_analysis": {
            "total_disk_image_files": len(all_disk_imgs),
            "unique_disk_image_names": len(disk_img_names),
            "unique_json_image_names": len(json_union_filenames),
            "on_disk_not_in_json": on_disk_not_in_json,
            "in_json_not_on_disk": in_json_not_on_disk,
            "shared_across_json_splits": overlap_between_jsons,
        },
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(overlap_payload, f, indent=2, ensure_ascii=False)

    # 4. Xuất file báo cáo DATA_CHECK.md
    lines = [
        "# DATA_CHECK.md — Báo Cáo Kiểm Chứng Dữ Liệu CAMO-FS (Bước 0)",
        "",
        "## 1. Thống kê tổng quan trên ổ cứng & File nhãn",
        f"- **Thư mục gốc (`data_root`):** `{data_root}`",
        f"- **Tổng số file ảnh tìm thấy trên ổ cứng:** `{len(all_disk_imgs)}` (Số tên file duy nhất: `{len(disk_img_names)}`)",
        f"- **Tổng số tên ảnh duy nhất khai báo trong các file JSON:** `{len(json_union_filenames)}`",
        f"- **Tổng số lớp phát hiện được:** `{len(sorted_cats)}` (Mốc chuẩn bài báo: `47`)",
        "",
        "### Chi tiết từng file JSON tìm thấy:",
        "| File JSON | Số lớp | Số ảnh khai báo | Số ảnh duy nhất | Số instances (bboxes) | Số ảnh 0 instance |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]

    total_entries = 0
    total_anns = 0
    for js in json_summaries:
        rel_p = Path(js["path"]).relative_to(data_root)
        total_entries += js["num_images_entries"]
        total_anns += js["num_annotations"]
        lines.append(
            f"| `{rel_p}` | {js['num_categories']} | {js['num_images_entries']} | "
            f"{js['num_unique_filenames']} | {js['num_annotations']} | {len(js['imgs_without_anns'])} |"
        )
    lines.append(f"| **TỔNG CỘNG (Cộng ngang)** | — | **{total_entries}** | **{len(json_union_filenames)}** | **{total_anns}** | — |")

    lines.extend([
        "",
        "## 2. Truy vết nguyên nhân chênh lệch số lượng ảnh (Ví dụ: 197 + 2.655 = 2.852 vs 2.858)",
        f"- **Ảnh có trên ổ cứng nhưng KHÔNG nằm trong bất kỳ file JSON nào ({len(on_disk_not_in_json)} ảnh):** "
        f"`{on_disk_not_in_json[:20]}`",
        f"- **Ảnh khai báo trong JSON nhưng THIẾU file trên ổ cứng ({len(in_json_not_on_disk)} ảnh):** "
        f"`{in_json_not_on_disk[:20]}`",
    ])
    for split_pair, shared_list in overlap_between_jsons.items():
        lines.append(
            f"- **Ảnh xuất hiện trùng ở cả 2 file `{split_pair}` ({len(shared_list)} ảnh):** `{shared_list[:20]}`"
        )
    for js in json_summaries:
        if js["imgs_without_anns"]:
            rel_p = Path(js["path"]).relative_to(data_root)
            lines.append(
                f"- **Ảnh trong `{rel_p}` không có bất kỳ instance nào ({len(js['imgs_without_anns'])} ảnh):** "
                f"`{js['imgs_without_anns'][:20]}`"
            )

    lines.extend([
        "",
        "## 3. Phân nhóm 47 Lớp CAMO-FS so với 80 Lớp MS-COCO (`yolov8n.pt`)",
        f"- **Nhóm 1 — Trùng tên trực tiếp hoặc từ khóa với COCO ({len(exact_or_token_overlap)} lớp):**",
    ])
    for r in exact_or_token_overlap:
        lines.append(f"  - `[yolo_id={r['yolo_id']}]` **{r['name']}** (khớp `{r['matched_coco']}` - kiểu `{r['match_type']}`)")

    lines.extend([
        f"- **Nhóm 2 — Cùng họ động vật lớn với COCO ({len(semantic_family_overlap)} lớp, tùy chọn tách riêng):**",
    ])
    for r in semantic_family_overlap:
        lines.append(f"  - `[yolo_id={r['yolo_id']}]` **{r['name']}** (họ `{r['matched_coco']}`)")

    lines.extend([
        f"- **Nhóm 3 — Lớp Novel hoàn toàn không có trong COCO ({len(pure_novel)} lớp):**",
        "  " + ", ".join([f"`{r['name']}`" for r in pure_novel]),
        "",
    ])

    out_md.parent.mkdir(parents=True, exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[OK] Đã xuất báo cáo Bước 0 tại: {out_md}")
    print(f"[OK] Đã xuất cấu hình tách lớp COCO tại: {out_json}")
    return overlap_payload


def main():
    parser = argparse.ArgumentParser(description="Bước 0: Kiểm chứng dữ liệu CAMO-FS & COCO Overlap")
    parser.add_argument("--data-root", type=str, required=True, help="Đường dẫn thư mục chứa CAMO-FS")
    parser.add_argument("--out-md", type=str, default="DATA_CHECK.md", help="File báo cáo Markdown đầu ra")
    parser.add_argument("--out-json", type=str, default="class_overlap.json", help="File JSON lưu nhóm lớp trùng COCO")
    args = parser.parse_args()

    run_audit(Path(args.data_root), Path(args.out_md), Path(args.out_json))


if __name__ == "__main__":
    main()
