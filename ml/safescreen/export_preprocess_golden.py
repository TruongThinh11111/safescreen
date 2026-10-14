"""
safescreen/export_preprocess_golden.py — Xuất dữ liệu đối chiếu cho bước tiền xử lý ảnh bản Kotlin.

Với vài ảnh mẫu, lưu: ảnh đầu vào (RGB), ảnh sau normalize_background, ảnh cuối cùng đưa cho Tesseract.
Unit test Android (PreprocessTest.kt) chạy bản Kotlin trên ảnh đầu vào và so TỪNG ĐIỂM ẢNH.

Chạy (đứng ở thư mục ml/), chạy lại mỗi khi sửa preprocess() trong ocr.py:
    python -m safescreen.export_preprocess_golden
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from safescreen.ocr import normalize_background, preprocess

ML_DIR = Path(__file__).resolve().parent.parent
OUT = ML_DIR.parent / "android" / "app" / "src" / "test" / "resources" / "preprocess"

# (tên, ảnh, vùng cắt (x0, y0, x1, y1) hoặc None) — chọn để phủ đủ các nhánh của OpenCV
CASES = [
    ("bank_light_1080", "data/test/images/00010.png", None),        # thu nhỏ đúng 8 lần
    ("chat_dark_1080", "data/test/images/00018.png", None),         # nền tối → đảo màu
    ("idcard_dark_720", "data/test/images/00000.png", None),        # phóng ×2
    ("form_1080x2181", "data/test/images/00050.png", None),         # chiều cao không chia hết cho 8
    ("crop_1077x1501", "data/test/images/00073.png", (3, 300, 1080, 1801)),  # cả 2 chiều lẻ
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta = []
    for name, path, crop in CASES:
        img = Image.open(ML_DIR / path).convert("RGB")
        if crop:
            img = img.crop(crop)
        gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        norm = normalize_background(gray)
        final, scale = preprocess(img)
        img.save(OUT / f"{name}_input.png")
        Image.fromarray(norm).save(OUT / f"{name}_norm.png")
        Image.fromarray(final).save(OUT / f"{name}_final.png")
        meta.append({"name": name, "width": img.width, "height": img.height,
                     "final_width": final.shape[1], "final_height": final.shape[0], "scale": scale})
        print(f"{name}: {img.size} -> {final.shape[1]}x{final.shape[0]}, scale={scale}")
    (OUT / "cases.json").write_text(json.dumps({"opencv": cv2.__version__, "cases": meta}, indent=1), encoding="utf-8")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
