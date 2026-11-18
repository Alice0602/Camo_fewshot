#!/usr/bin/env python3
"""
Bước 4 (Tuần 1 - Chưa cần Camera vật lý):
1. Mô phỏng "Trạm Bẫy Ảnh Sinh Thái Thông Minh" trực tiếp trên thư mục ảnh Test hoặc file Video.
2. Tự động lọc và xuất ra 5 ảnh dự đoán TỐT NHẤT và 5 ảnh dự đoán SAI/SÓT (Qualitative Analysis cho TV3 & TV4).
3. Kiểm tra xuất mô hình sang định dạng nhúng (ONNX / NCNN) để xác nhận 100% tương thích với Orange Pi 5 Pro.
"""

import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


def compute_iou_xyxy(box1: np.ndarray, box2: np.ndarray) -> float:
    x1 = max(float(box1[0]), float(box2[0]))
    y1 = max(float(box1[1]), float(box2[1]))
    x2 = min(float(box1[2]), float(box2[2]))
    y2 = min(float(box1[3]), float(box2[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, float(box1[2] - box1[0])) * max(0.0, float(box1[3] - box1[1]))
    area2 = max(0.0, float(box2[2] - box2[0])) * max(0.0, float(box2[3] - box2[1]))
    union = area1 + area2 - inter
    return inter / union if union > 0.0 else 0.0


def load_yolo_gt_boxes(lbl_path: Path, img_w: int, img_h: int) -> List[Tuple[int, np.ndarray]]:
    if not lbl_path.exists():
        return []
    gts = []
    with open(lbl_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cid = int(parts[0])
            xc, yc, bw, bh = [float(x) for x in parts[1:]]
            x1 = (xc - bw / 2.0) * img_w
            y1 = (yc - bh / 2.0) * img_h
            x2 = (xc + bw / 2.0) * img_w
            y2 = (yc + bh / 2.0) * img_h
            gts.append((cid, np.array([x1, y1, x2, y2], dtype=float)))
    return gts


def draw_gt_and_pred(
    img: np.ndarray,
    gts: List[Tuple[int, np.ndarray]],
    preds: List[Tuple[int, float, np.ndarray]],
    names: Dict[int, str],
) -> np.ndarray:
    canvas = img.copy()
    # Vẽ Ground Truth màu Xanh Lá (Green) nét đứt/mỏng
    for cid, box in gts:
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cname = names.get(cid, str(cid))
        cv2.putText(canvas, f"GT: {cname}", (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Vẽ Prediction màu Đỏ/Cam (BGR: 0, 140, 255)
    for cid, conf, box in preds:
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 140, 255), 2)
        cname = names.get(cid, str(cid))
        cv2.putText(
            canvas,
            f"Pred: {cname} ({conf:.2f})",
            (x1, min(canvas.shape[0] - 8, y2 + 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 140, 255),
            2,
        )
    return canvas


def run_cameraless_simulation(
    weights_path: Path,
    test_images_dir: Path,
    out_dir: Path,
    conf_thres: float = 0.25,
    max_images: int = 200,
    export_edge: bool = False,
):
    from ultralytics import YOLO

    model = YOLO(str(weights_path))
    names = model.names

    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    img_files = sorted([p for p in test_images_dir.glob("*") if p.suffix.lower() in img_exts])[:max_images]
    lbl_dir = test_images_dir.parent / "labels"

    latencies_ms = []
    scored_samples = []

    for img_path in img_files:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h_img, w_img = img.shape[:2]
        gts = load_yolo_gt_boxes(lbl_dir / f"{img_path.stem}.txt", w_img, h_img)

        t0 = time.perf_counter()
        res = model.predict(img, imgsz=640, conf=conf_thres, verbose=False)[0]
        dt_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(dt_ms)

        preds = []
        if res.boxes is not None and len(res.boxes) > 0:
            xyxys = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            clss = res.boxes.cls.cpu().numpy().astype(int)
            for c, cf, bx in zip(clss, confs, xyxys):
                preds.append((int(c), float(cf), bx))

        # Tính điểm khớp tốt nhất với Ground Truth để xếp hạng 5 ảnh tốt nhất & 5 ảnh lỗi nặng nhất
        best_correct_score = 0.0
        for gt_cid, gt_box in gts:
            for pr_cid, pr_conf, pr_box in preds:
                iou = compute_iou_xyxy(gt_box, pr_box)
                if pr_cid == gt_cid and iou >= 0.5:
                    best_correct_score = max(best_correct_score, iou * pr_conf)

        scored_samples.append({
            "img_path": img_path,
            "gts": gts,
            "preds": preds,
            "score": best_correct_score,
        })

    scored_samples.sort(key=lambda x: x["score"], reverse=True)
    top5_good = [s for s in scored_samples if s["score"] > 0][:5]
    top5_bad = [s for s in reversed(scored_samples) if len(s["gts"]) > 0][:5]

    good_dir = out_dir / "top5_good"
    bad_dir = out_dir / "top5_bad"
    good_dir.mkdir(parents=True, exist_ok=True)
    bad_dir.mkdir(parents=True, exist_ok=True)

    for idx, item in enumerate(top5_good, 1):
        img = cv2.imread(str(item["img_path"]))
        vis = draw_gt_and_pred(img, item["gts"], item["preds"], names)
        cv2.imwrite(str(good_dir / f"good_{idx}_{item['img_path'].name}"), vis)

    for idx, item in enumerate(top5_bad, 1):
        img = cv2.imread(str(item["img_path"]))
        vis = draw_gt_and_pred(img, item["gts"], item["preds"], names)
        cv2.imwrite(str(bad_dir / f"bad_{idx}_{item['img_path'].name}"), vis)

    mean_ms = float(np.mean(latencies_ms)) if latencies_ms else 0.0
    fps = 1000.0 / max(mean_ms, 0.1)
    print(f"[OK] Đã chạy mô phỏng bẫy ảnh trên {len(latencies_ms)} ảnh.")
    print(f"     Tốc độ trung bình: {fps:.1f} FPS | Độ trễ: {mean_ms:.2f} ms/ảnh")
    print(f"     Đã lưu 5 ảnh tốt nhất tại: {good_dir}")
    print(f"     Đã lưu 5 ảnh dự đoán sai/sót tại: {bad_dir}")

    if export_edge:
        print("\n[EDGE EXPORT] Đang kiểm tra xuất mô hình sang ONNX & NCNN (cho Orange Pi 5 Pro)...")
        onnx_path = model.export(format="onnx", imgsz=640)
        print(f"  -> Xuất ONNX thành công: {onnx_path}")
        try:
            ncnn_path = model.export(format="ncnn", imgsz=640)
            print(f"  -> Xuất NCNN (ARM CPU Orange Pi 5 Pro) thành công: {ncnn_path}")
        except Exception as e:
            print(f"  -> [Lưu ý] NCNN cần gói ncnn cài thêm khi chạy trên Orange Pi ({e})")


def main():
    parser = argparse.ArgumentParser(description="Mô phỏng Bẫy ảnh Tuần 1 (Chưa cần Camera) & Lọc 5 ảnh tốt / 5 ảnh sai")
    parser.add_argument("--weights", type=str, required=True, help="Đường dẫn file last.pt")
    parser.add_argument("--test-images", type=str, default="data/processed/test/images", help="Thư mục ảnh test")
    parser.add_argument("--out-dir", type=str, default="results/qualitative_analysis", help="Thư mục lưu ảnh phân tích")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--max-images", type=int, default=200)
    parser.add_argument("--export-edge", action="store_true", help="Kiểm tra xuất ONNX/NCNN cho Orange Pi 5 Pro")
    args = parser.parse_args()

    run_cameraless_simulation(
        Path(args.weights),
        Path(args.test_images),
        Path(args.out_dir),
        conf_thres=args.conf,
        max_images=args.max_images,
        export_edge=args.export_edge,
    )


if __name__ == "__main__":
    main()
