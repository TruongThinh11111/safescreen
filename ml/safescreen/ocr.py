"""
safescreen/ocr.py — Bước 2: OCR bằng Tesseract và ghép nhãn gốc vào kết quả OCR.

Vì sao cần "ghép nhãn"?
    Khi chạy thật, CRF chỉ nhìn thấy những gì OCR đọc được (có thể sai chính tả, gộp 2 từ làm 1,
    bỏ sót từ...). Vì vậy CRF phải được train trên ĐẦU RA OCR chứ không phải trên văn bản gốc.
    Ta chạy OCR trên ảnh tổng hợp, rồi với mỗi từ OCR, tìm token gốc nằm cùng chỗ (bbox chồng nhau)
    để lấy nhãn của nó.

Đầu ra mỗi ảnh: data/<...>/ocr/<tên>.json
    {"text": "0912345678", "bbox": [x0,y0,x1,y1], "conf": 0.93, "line": 4, "label": "B-PHONE"}
    conf = độ tin cậy của Tesseract (0–1) → sau này là c_ocr trong hàm rủi ro.

Chạy (đứng ở thư mục ml/):
    python -m safescreen.ocr --data data/synth --limit 50
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from PIL import Image

ML_DIR = Path(__file__).resolve().parent.parent
WINDOWS_TESSERACT = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")


# =====================================================================
# 1. Cấu hình Tesseract
# =====================================================================

def setup_tesseract() -> str:
    """
    Tìm tesseract và thư mục tessdata. Trả về chuỗi config cần truyền cho pytesseract.
    - Windows: dùng bản cài ở Program Files + ml/tessdata (có vie bản best).
    - Colab/Kaggle: `apt install tesseract-ocr tesseract-ocr-vie` là đủ, tesseract đã nằm trong PATH.
    Có thể ghi đè bằng biến môi trường TESSERACT_CMD và TESSDATA_DIR.
    """
    cmd = os.environ.get("TESSERACT_CMD") or (str(WINDOWS_TESSERACT) if WINDOWS_TESSERACT.exists() else "tesseract")
    pytesseract.pytesseract.tesseract_cmd = cmd

    tessdata = os.environ.get("TESSDATA_DIR")
    if not tessdata and (ML_DIR / "tessdata" / "vie.traineddata").exists():
        tessdata = (ML_DIR / "tessdata").as_posix()
    # Lưu ý: KHÔNG bọc đường dẫn trong dấu nháy — pytesseract sẽ truyền nguyên dấu nháy cho tesseract
    return f"--tessdata-dir {tessdata}" if tessdata else ""


TESS_CONFIG = setup_tesseract()


# =====================================================================
# 2. Tiền xử lý ảnh + chạy OCR
# =====================================================================

def normalize_background(gray: np.ndarray, block: int = 8) -> np.ndarray:
    """
    Đưa mọi vùng về dạng "chữ đen trên nền trắng" — kiểu ảnh Tesseract đọc tốt nhất.
    Ảnh chụp màn hình hay có chữ trắng trên nền tối/màu (chế độ tối, thanh tiêu đề, nút bấm),
    và Tesseract thường bỏ qua cả khối nền xám vì tưởng đó là hình ảnh.

    Làm việc trên ảnh thu nhỏ `block` lần; mỗi điểm ảnh nhỏ nhìn cửa sổ 7x7 (≈ 56x56 px ảnh gốc):
    1. Màu nền = trung vị (median). Nét chữ mảnh chỉ chiếm ít điểm ảnh nên trung vị ≈ màu nền.
    2. Chiều chữ: so (sáng nhất - nền) với (nền - tối nhất). Bên nào xa nền hơn là chữ.
       → chữ sáng hơn nền thì đảo màu vùng đó. Không dùng ngưỡng độ sáng cố định, vì nền cam
       (xám ≈ 133) chữ trắng sẽ bị bỏ qua, còn nền xám nhạt chữ đen lại bị đảo nhầm.
    3. Chia cho độ sáng nền: nền → 255 (trắng tinh), chữ đen vẫn gần 0.
    """
    h, w = gray.shape
    small_size = (max(1, w // block), max(1, h // block))
    k_block, k_win = np.ones((block, block), np.uint8), np.ones((7, 7), np.uint8)

    def shrink(a):
        return cv2.resize(a, small_size, interpolation=cv2.INTER_AREA)

    bg = cv2.medianBlur(shrink(gray), 7).astype(np.int16)
    hi = cv2.dilate(shrink(cv2.dilate(gray, k_block)), k_win).astype(np.int16)  # điểm sáng nhất quanh đó
    lo = cv2.erode(shrink(cv2.erode(gray, k_block)), k_win).astype(np.int16)    # điểm tối nhất quanh đó
    light_text = ((hi - bg) > (bg - lo)).astype(np.uint8)

    def grow(a):
        return cv2.resize(a, (w, h), interpolation=cv2.INTER_NEAREST)

    flip = grow(light_text).astype(bool)
    bg = grow(bg.astype(np.uint8)).astype(np.float32)
    img = np.where(flip, 255 - gray, gray).astype(np.float32)
    bg = np.where(flip, 255 - bg, bg)
    return np.clip(img * 255 / np.maximum(bg, 1), 0, 255).astype(np.uint8)


def preprocess(img: Image.Image, target_width: int = 1440, invert_dark: bool = True):
    """
    Chuẩn bị ảnh cho Tesseract. Trả về (ảnh xám, hệ số phóng).
    Các bước đều đơn giản để sau này viết lại được bằng Kotlin trên Android.
    """
    gray = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    if invert_dark:
        gray = normalize_background(gray)
    # Phóng to cho chữ đủ lớn (Tesseract đọc tốt khi chữ cao ~30px trở lên)
    scale = target_width / gray.shape[1]
    if abs(scale - 1) > 0.05:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    else:
        scale = 1.0
    return gray, scale


def run_ocr(img: Image.Image, lang: str = "vie+eng", psm: int = 3,
            target_width: int = 1440, invert_dark: bool = True) -> list[dict]:
    """
    OCR 1 ảnh → danh sách từ {text, bbox, conf, line} theo thứ tự đọc từng hàng (xem sort_reading_order).
    bbox được đổi về toạ độ của ẢNH GỐC (chia lại cho hệ số phóng).
    lang: "vie+eng" thay vì chỉ "vie" — "vie" đọc tên có dấu tốt hơn nhưng đọc sai CHỮ SỐ
          (vd 3732335983 → 3732330983) và email, mà regex cần số đúng tuyệt đối.
          Tên bị mất dấu (TRẦN → TRAN) thì CRF vẫn học được vì dữ liệu có sẵn tin nhắn không dấu.
    psm (page segmentation mode): 3 = tự động chia khối (mặc định của Tesseract),
                                  11 = "chữ rải rác". Thử trên 80 ảnh tổng hợp: psm 3 phủ 99,2% token
                                  nhạy cảm, psm 11 chỉ 96,8% (hay sót chữ trong bong bóng chat/ô form).
    """
    arr, s = preprocess(img, target_width, invert_dark)
    data = pytesseract.image_to_data(arr, lang=lang, config=f"{TESS_CONFIG} --psm {psm}",
                                     output_type=pytesseract.Output.DICT)
    words = []
    for i, raw in enumerate(data["text"]):
        text, conf = raw.strip(), float(data["conf"][i])
        if not text or conf < 0:  # conf = -1 là dòng mô tả khối/đoạn, không phải từ
            continue
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        words.append({"text": text,
                      "bbox": [round(x / s), round(y / s), round((x + w) / s), round((y + h) / s)],
                      "conf": round(conf / 100, 3)})
    return sort_reading_order(words)  # gán "line" theo hàng ngang trên màn hình


def sort_reading_order(words: list[dict]) -> list[dict]:
    """
    Sắp xếp lại từ theo HÀNG NGANG trên màn hình (trên → dưới, trong hàng thì trái → phải),
    và đánh lại số `line` theo hàng.

    Vì sao: với bố cục 2 cột kiểu app ngân hàng ("Số tài khoản ........ 293409457"), Tesseract đọc
    hết cột trái rồi mới tới cột phải → từ khóa "Số tài khoản" nằm cách xa con số của nó,
    regex/CRF mất ngữ cảnh. Gom theo hàng thì nhãn và giá trị lại đứng cạnh nhau.

    Cách gom: duyệt từ theo tâm dọc; từ nào có tâm nằm trong dải [y0, y1] của hàng hiện tại
    thì cùng hàng, ngược lại mở hàng mới.

    Khóa sắp xếp ĐẦY ĐỦ (tâm dọc, rồi x0, y0, x1, y1, chữ) để kết quả không phụ thuộc thứ tự
    Tesseract trả về: 2 từ cùng tâm dọc mà xếp khác nhau thì từ đứng đầu (quyết định dải của hàng)
    khác nhau → chia hàng khác nhau. Bản Kotlin phải dùng đúng khóa này.
    """
    def row_key(w):
        x0, y0, x1, y1 = w["bbox"]
        return ((y0 + y1) / 2, x0, y0, x1, y1, w["text"])

    rows: list[dict] = []
    for w in sorted(words, key=row_key):
        cy = (w["bbox"][1] + w["bbox"][3]) / 2
        if rows and rows[-1]["y0"] <= cy <= rows[-1]["y1"]:
            rows[-1]["words"].append(w)
        else:
            rows.append({"y0": w["bbox"][1], "y1": w["bbox"][3], "words": [w]})
    out = []
    for i, row in enumerate(rows):
        for w in sorted(row["words"], key=lambda w: (w["bbox"][0], w["bbox"][1], w["bbox"][2], w["bbox"][3], w["text"])):
            w["line"] = i
            out.append(w)
    return out


# =====================================================================
# 3. Ghép nhãn gốc (ground truth) vào từ OCR
# =====================================================================

def area(b) -> float:
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def intersection(a, b) -> float:
    return area([max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])])


def align_labels(words: list[dict], gt_tokens: list[dict], min_cover: float = 0.3) -> list[dict]:
    """
    Gán nhãn BIO cho từng từ OCR dựa vào token gốc có bbox chồng lên nó.

    Quy tắc (nghiêng về RECALL — thà gán nhạy cảm còn hơn bỏ sót):
      - Nếu phần chồng với token THỰC THỂ chiếm ≥ min_cover diện tích từ OCR → lấy nhãn thực thể đó
        (kể cả khi OCR gộp "STK 1025" thành 1 từ, phần số vẫn kéo nhãn ACCOUNT theo).
      - Ngược lại → O.
    B-/I- được tính lại theo thứ tự OCR: từ OCR thuộc cùng 1 thực thể gốc với từ ngay trước → I-, khác → B-.
    """
    # Mỗi token gốc thuộc thực thể nào (đánh số span: span tăng mỗi khi gặp B-)
    span_of, span = [], -1
    for t in gt_tokens:
        if t["label"].startswith("B-"):
            span += 1
        span_of.append(None if t["label"] == "O" else span)

    prev_span = None
    for w in words:
        w_area = area(w["bbox"]) or 1
        best_j, best_ov = None, 0.0
        for j, t in enumerate(gt_tokens):
            if span_of[j] is None:
                continue
            ov = intersection(w["bbox"], t["bbox"])
            if ov > best_ov:
                best_j, best_ov = j, ov
        if best_j is not None and best_ov / w_area >= min_cover:
            ent, sp = gt_tokens[best_j]["label"][2:], span_of[best_j]
            w["label"] = ("I-" if sp == prev_span else "B-") + ent
            prev_span = sp
        else:
            w["label"] = "O"
            prev_span = None
    return words


# =====================================================================
# 4. Đánh giá chất lượng OCR trên các token nhạy cảm
# =====================================================================

def ocr_stats(words: list[dict], gt_tokens: list[dict], min_cover: float = 0.3) -> dict:
    """
    Với mỗi token NHẠY CẢM gốc:
      - covered: có từ OCR nào phủ lên ≥ min_cover diện tích của nó không?
                 Token không được phủ = OCR bỏ sót → hệ thống KHÔNG thể che → giới hạn trên của recall.
      - exact  : có từ OCR nào đọc ĐÚNG Y HỆT chữ của nó không? (ảnh hưởng tới regex/CRF)
    """
    st = defaultdict(Counter)
    for t in gt_tokens:
        if t["label"] == "O":
            continue
        ent = t["label"][2:]
        t_area = area(t["bbox"]) or 1
        hits = [w for w in words if intersection(w["bbox"], t["bbox"]) / t_area >= min_cover]
        st[ent]["total"] += 1
        st[ent]["covered"] += bool(hits)
        st[ent]["exact"] += any(w["text"] == t["text"] for w in hits)
    return st


# =====================================================================
# 5. Chạy cho cả thư mục dữ liệu
# =====================================================================

def process_one(label_path: str, out_dir: str, params: dict) -> dict:
    """Xử lý 1 ảnh: OCR → ghép nhãn → lưu JSON. Trả về thống kê để cộng dồn."""
    label_path = Path(label_path)
    meta = json.loads(label_path.read_text(encoding="utf-8"))
    img = Image.open(label_path.parent.parent / meta["image"])

    t0 = time.perf_counter()
    words = run_ocr(img, **params)
    elapsed = time.perf_counter() - t0
    align_labels(words, meta["tokens"])

    out = {"image": meta["image"], "screen_type": meta["screen_type"],
           "width": meta["width"], "height": meta["height"], "ocr_params": params, "words": words}
    (Path(out_dir) / label_path.name).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return {"stats": ocr_stats(words, meta["tokens"]), "seconds": elapsed, "screen_type": meta["screen_type"]}


def print_report(results: list[dict]):
    total, by_screen = defaultdict(Counter), defaultdict(Counter)
    for r in results:
        for ent, c in r["stats"].items():
            total[ent].update(c)
            by_screen[r["screen_type"]].update(c)

    def row(name, c):
        n = c["total"] or 1
        print(f"  {name:<10} {c['total']:>6} {c['covered'] / n:>9.1%} {c['exact'] / n:>9.1%}")

    print(f"\n  {'Thực thể':<10} {'Số token':>6} {'Phủ (bbox)':>9} {'Đọc đúng':>9}")
    for ent in sorted(total):
        row(ent, total[ent])
    row("TỔNG", sum(total.values(), Counter()))
    print(f"\n  {'Màn hình':<10}")
    for s in sorted(by_screen):
        row(s, by_screen[s])
    secs = [r["seconds"] for r in results]
    print(f"\n  Thời gian OCR trung bình: {np.mean(secs):.2f}s/ảnh")


def main():
    ap = argparse.ArgumentParser(description="OCR ảnh tổng hợp và ghép nhãn gốc")
    ap.add_argument("--data", default="data/synth", help="thư mục do synth.py tạo ra")
    ap.add_argument("--out", default=None, help="thư mục lưu kết quả (mặc định <data>/ocr)")
    ap.add_argument("--limit", type=int, default=0, help="chỉ xử lý N ảnh đầu (0 = tất cả)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--psm", type=int, default=3)
    ap.add_argument("--target-width", type=int, default=1440)
    ap.add_argument("--no-invert", action="store_true", help="tắt chuẩn hoá nền (normalize_background)")
    ap.add_argument("--lang", default="vie+eng")
    args = ap.parse_args()

    data = Path(args.data)
    out = Path(args.out) if args.out else data / "ocr"
    out.mkdir(parents=True, exist_ok=True)
    files = sorted((data / "labels").glob("*.json"))
    if args.limit:
        files = files[:args.limit]
    params = {"lang": args.lang, "psm": args.psm, "target_width": args.target_width,
              "invert_dark": not args.no_invert}

    # Mỗi tiến trình chạy 1 tesseract; giới hạn tesseract dùng 1 luồng để các tiến trình không tranh CPU
    os.environ["OMP_THREAD_LIMIT"] = "1"
    print(f"OCR {len(files)} ảnh, {args.workers} tiến trình, tham số {params}")
    t0 = time.perf_counter()
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(process_one, map(str, files), [str(out)] * len(files), [params] * len(files)))
    print(f"Xong trong {time.perf_counter() - t0:.0f}s → {out.resolve()}")
    print_report(results)


if __name__ == "__main__":
    main()
