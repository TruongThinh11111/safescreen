"""
safescreen/risk_mask.py — Bước 5: hàm rủi ro F, chọn ngưỡng τ, che ảnh, pipeline end-to-end.

Hàm rủi ro cho mỗi token:
    F = σ(λ1·p_crf + λ2·r_regex + λ3·c_ocr + λ4·ctx + β)      che token nếu F ≥ τ

Vì sao LogisticRegression chính là cách "học" hàm tuyến tính F?
    LogisticRegression mô hình hoá xác suất một token là nhạy cảm bằng đúng công thức này:
        P(nhạy cảm | x) = σ(λ·x + β),   với σ(z) = 1 / (1 + e^(−z))
    Khi train, nó tìm λ, β làm CỰC ĐẠI log-likelihood trên dữ liệu có nhãn — tương đương
    cực tiểu log-loss  Σ −[y·log F + (1 − y)·log(1 − F)].
    Nghĩa là: thay vì tự chọn λ bằng cảm tính, ta để dữ liệu chọn bộ trọng số khiến F vừa là
    một tổ hợp tuyến tính đi qua sigmoid, vừa là một XÁC SUẤT được hiệu chỉnh (calibrated):
    trong các token có F ≈ 0.8, khoảng 80% thật sự nhạy cảm.
      - λ_k > 0: đặc trưng k càng lớn → càng nhạy cảm; λ_k < 0: ngược lại.
      - Cả 4 đặc trưng đều nằm trong [0, 1] nên so sánh |λ_k| để biết đặc trưng nào quan trọng.
      - β: "độ lệch" — giá trị (trước sigmoid) khi mọi đặc trưng bằng 0.

Chạy (đứng ở thư mục ml/):
    python -m safescreen.risk_mask fit  --val data/val --test data/test   # học λ, β, τ
    python -m safescreen.risk_mask demo --test data/test --n 5            # ảnh trước/sau
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve

from safescreen.crf import KEYWORDS, load as load_crf, load_docs, predict_tokens
from safescreen.ocr import run_ocr
from safescreen.text_utils import fold

FEATURES = ["p_crf", "r_regex", "c_ocr", "ctx"]
CTX_WINDOW = 3
DEFAULT_CRF = Path("models/crf.joblib")
DEFAULT_RISK = Path("models/risk.json")
ALL_KEYWORDS = [k for kws in KEYWORDS.values() for k in kws]


# =====================================================================
# 1. Vector đặc trưng rủi ro cho từng token
# =====================================================================

def risk_features(words: list[dict]) -> np.ndarray:
    """
    words: danh sách từ ĐÃ qua crf.predict_tokens (có p_sensitive, rx_score). Trả về ma trận (số từ × 4):
      p_crf   = 1 − P(O) từ predict_marginals
      r_regex = điểm regex (1.0 chặt / 0.6 lỏng / 0 không khớp)
      c_ocr   = độ tin cậy OCR (ocr.py đã chia 100 → 0–1)
      ctx     = 1 nếu trong cửa sổ ±3 từ (kể cả chính nó) có từ khóa nhạy cảm ("Họ tên", "STK", ...)
    """
    folded = [re.sub(r"[^\w\s.]", " ", fold(w["text"])) for w in words]
    rows = []
    for i, w in enumerate(words):
        window = re.sub(r"\s+", " ", " ".join(folded[max(0, i - CTX_WINDOW):i + CTX_WINDOW + 1]))
        ctx = any(k in window for k in ALL_KEYWORDS)
        rows.append([w["p_sensitive"], w["rx_score"], min(max(w["conf"], 0.0), 1.0), float(ctx)])
    return np.array(rows, dtype=float).reshape(-1, len(FEATURES))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


@dataclass
class RiskModel:
    lambdas: list[float]   # λ1..λ4 theo thứ tự FEATURES
    beta: float
    tau: float

    def score(self, X: np.ndarray) -> np.ndarray:
        """F = σ(λ·x + β) cho từng dòng của X."""
        return sigmoid(X @ np.array(self.lambdas) + self.beta)

    def save(self, path=DEFAULT_RISK):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # Lưu JSON (không dùng pickle) để app Android đọc thẳng được 6 con số này
        Path(path).write_text(json.dumps({"features": FEATURES, **asdict(self)}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path=DEFAULT_RISK) -> "RiskModel":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(d["lambdas"], d["beta"], d["tau"])


# =====================================================================
# 2. Học λ, β (LogisticRegression) và 3. chọn ngưỡng τ
# =====================================================================

def build_xy(crf, docs: list[list[dict]]) -> tuple[np.ndarray, np.ndarray]:
    """Chạy CRF trên từng ảnh → ghép vector rủi ro của mọi token; nhãn đích y = 1 nếu token nhạy cảm."""
    X, y = [], []
    for words in docs:
        predict_tokens(crf, words)
        X.append(risk_features(words))
        y.extend(w["label"] != "O" for w in words)
    return np.vstack(X), np.array(y, dtype=int)


def choose_tau(y: np.ndarray, F: np.ndarray, min_recall: float, beta: float = 2.0):
    """
    Duyệt mọi ngưỡng có thể. Ràng buộc cứng: recall ≥ min_recall.
    Trong các ngưỡng thoả ràng buộc, chọn ngưỡng có F-beta lớn nhất (beta=2: recall quan trọng gấp đôi
    precision).

    Vì sao không lấy "ngưỡng LỚN NHẤT còn recall ≥ 0.95"? Khi mô hình tốt (recall đạt ~99.9%), quy tắc đó
    đẩy τ lên sát 1 và tự hạ recall xuống đúng 95% chỉ để bớt vài token che nhầm — ngược với mục tiêu
    ưu tiên recall. F2 chỉ chịu hy sinh recall khi đổi lại được nhiều precision.
    """
    precision, recall, thresholds = precision_recall_curve(y, F)
    # precision/recall có n+1 phần tử, phần tử cuối không ứng với ngưỡng nào → bỏ
    precision, recall = precision[:-1], recall[:-1]
    f_beta = (1 + beta**2) * precision * recall / np.maximum(beta**2 * precision + recall, 1e-12)
    f_beta[recall < min_recall] = -1  # loại các ngưỡng vi phạm ràng buộc recall
    k = int(np.argmax(f_beta))
    return float(thresholds[k]), precision, recall, thresholds


def plot_pr_vs_tau(thresholds, precision, recall, tau, min_recall, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, ink2, grid, surface = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    blue, orange = "#2a78d6", "#eb6834"   # 2 màu đầu của bộ màu phân loại (đã kiểm chứng cho người mù màu)
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=150, facecolor=surface)
    ax.set_facecolor(surface)
    ax.plot(thresholds, precision, color=blue, lw=2, label="Precision")
    ax.plot(thresholds, recall, color=orange, lw=2, label="Recall")
    ax.axhline(min_recall, color=ink2, lw=1, ls=(0, (4, 3)))
    ax.axvline(tau, color=ink, lw=1)
    k = np.searchsorted(thresholds, tau)
    ax.annotate(f"τ = {tau:.3f}\nP = {precision[k]:.3f}, R = {recall[k]:.3f}", (tau, 0.5),
                xytext=(8, 0), textcoords="offset points", color=ink, fontsize=9, va="center")
    ax.text(0.01, min_recall, f"recall mục tiêu {min_recall:.2f}", color=ink2, fontsize=8,
            va="bottom", transform=ax.get_yaxis_transform())
    # Nhãn trực tiếp ở đầu phải mỗi đường (màu chữ trung tính, không dùng màu đường)
    for ys, name in [(precision, "Precision"), (recall, "Recall")]:
        ax.text(1.005, ys[-1], name, color=ink, fontsize=9, va="center", transform=ax.get_yaxis_transform())
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Ngưỡng τ (che token nếu F ≥ τ)", color=ink2)
    ax.set_ylabel("Giá trị trên tập val", color=ink2)
    ax.set_title("Precision và recall theo ngưỡng τ", color=ink, loc="left", fontsize=11)
    ax.grid(True, color=grid, lw=0.8)
    ax.tick_params(colors=ink2)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.legend(loc="lower left", frameon=False, labelcolor=ink, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, facecolor=surface)
    plt.close(fig)


def fit_risk(crf, val_docs, min_recall: float) -> tuple[RiskModel, dict]:
    X, y = build_xy(crf, val_docs)
    lr = LogisticRegression(C=10.0, max_iter=1000)  # C lớn = phạt nhẹ, vì chỉ có 4 đặc trưng
    lr.fit(X, y)
    F = lr.predict_proba(X)[:, 1]
    tau, precision, recall, thresholds = choose_tau(y, F, min_recall)
    model = RiskModel([float(v) for v in lr.coef_[0]], float(lr.intercept_[0]), tau)
    return model, {"precision": precision, "recall": recall, "thresholds": thresholds,
                   "n_tokens": len(y), "n_sensitive": int(y.sum())}


# =====================================================================
# Đánh giá trên test: so sánh regex / CRF / hàm rủi ro F
# =====================================================================

def mask_metrics(gold: np.ndarray, pred: np.ndarray) -> dict:
    tp = int((gold & pred).sum())
    return {"recall": tp / max(gold.sum(), 1), "precision": tp / max(pred.sum(), 1),
            "over_masking": int((~gold & pred).sum()) / max((~gold).sum(), 1)}


def evaluate_test(crf, risk: RiskModel, test_docs):
    gold, rx, crf_pred, F, ents = [], [], [], [], []
    for words in test_docs:
        predict_tokens(crf, words)
        F.extend(risk.score(risk_features(words)))
        for w in words:
            gold.append(w["label"] != "O")
            ents.append(w["label"][2:] if w["label"] != "O" else "O")
            rx.append(w["rx_score"] > 0)
            crf_pred.append(w["crf_label"] != "O")
    gold, F, ents = np.array(gold), np.array(F), np.array(ents)
    print(f"\n=== Che token trên tập test ({len(test_docs)} ảnh, {len(gold)} token, {gold.sum()} nhạy cảm) ===")
    print(f"  {'Hệ thống':<26}{'Recall':>9}{'Precision':>11}{'Over-masking':>14}")
    for name, pred in [("Chỉ regex (score ≥ 0.6)", np.array(rx)),
                       ("Chỉ CRF (nhãn ≠ O)", np.array(crf_pred)),
                       (f"Hàm rủi ro F ≥ τ={risk.tau:.3f}", F >= risk.tau)]:
        m = mask_metrics(gold, pred)
        print(f"  {name:<26}{m['recall']:>9.2%}{m['precision']:>11.2%}{m['over_masking']:>14.3%}")
    print("\n  Recall của hàm rủi ro theo từng loại thực thể:")
    for ent in sorted(set(ents) - {"O"}):
        sel = ents == ent
        print(f"    {ent:<9} {(F[sel] >= risk.tau).mean():.2%}  ({sel.sum()} token)")


# =====================================================================
# 4. Che ảnh
# =====================================================================

def _same_line_adjacent(a, b) -> bool:
    """2 box nằm trên cùng dòng (chồng nhau ≥ 50% chiều cao) và cách nhau không quá ~1 chữ."""
    ha, hb = a[3] - a[1], b[3] - b[1]
    v_overlap = min(a[3], b[3]) - max(a[1], b[1])
    if v_overlap < 0.5 * min(ha, hb):
        return False
    h_gap = max(a[0], b[0]) - min(a[2], b[2])  # âm nếu 2 box chồng nhau theo chiều ngang
    return h_gap <= 0.8 * max(ha, hb)


def merge_regions(items: list[dict]) -> list[dict]:
    """
    Gộp các box liền nhau trên cùng dòng thành 1 vùng (vd "0912 345 678" → 1 vùng thay vì 3).
    items: dict có "bbox", tuỳ chọn "label", "F", "text". Vùng gộp giữ danh sách nhãn và F lớn nhất.
    """
    regions = [{"bbox": list(it["bbox"]), "labels": [it.get("label", "SENSITIVE")],
                "F": float(it.get("F", 1.0)), "text": it.get("text", "")} for it in items]
    merged = True
    while merged:  # lặp tới khi không gộp thêm được (gộp A+B có thể làm AB chạm C)
        merged = False
        for i in range(len(regions)):
            for j in range(i + 1, len(regions)):
                a, b = regions[i], regions[j]
                if _same_line_adjacent(a["bbox"], b["bbox"]):
                    first, second = (a, b) if a["bbox"][0] <= b["bbox"][0] else (b, a)
                    a["bbox"] = [min(a["bbox"][0], b["bbox"][0]), min(a["bbox"][1], b["bbox"][1]),
                                 max(a["bbox"][2], b["bbox"][2]), max(a["bbox"][3], b["bbox"][3])]
                    a["labels"] = sorted(set(a["labels"]) | set(b["labels"]))
                    a["F"] = max(a["F"], b["F"])
                    a["text"] = f"{first['text']} {second['text']}".strip()
                    del regions[j]
                    merged = True
                    break
            if merged:
                break
    return sorted(regions, key=lambda r: (r["bbox"][1], r["bbox"][0]))


MASK_MODES = ("gaussian_blur", "pixelate", "black_box")


def mask(img: Image.Image, boxes, mode: str = "black_box", pad: int = 4) -> Image.Image:
    """
    Che các vùng trên ảnh. boxes: list bbox [x0,y0,x1,y1] hoặc list dict có "bbox".
    Box liền nhau trên cùng dòng được gộp, rồi nới rộng `pad` px mỗi phía (3–5 px là vừa).

    Lưu ý bảo mật cho báo cáo: blur/pixelate với chữ số đôi khi có thể khôi phục lại được
    (thử từng chuỗi số rồi so ảnh mờ). black_box là cách duy nhất xoá hẳn thông tin → mặc định.
    """
    if mode not in MASK_MODES:
        raise ValueError(f"mode phải là một trong {MASK_MODES}")
    items = [b if isinstance(b, dict) else {"bbox": b} for b in boxes]
    out = img.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    for r in merge_regions(items):
        x0, y0, x1, y1 = r["bbox"]
        x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
        x1, y1 = min(out.width, x1 + pad), min(out.height, y1 + pad)
        if x1 <= x0 or y1 <= y0:
            continue
        h = y1 - y0
        if mode == "black_box":
            draw.rectangle([x0, y0, x1, y1], fill=(0, 0, 0))
            continue
        crop = out.crop((x0, y0, x1, y1))
        if mode == "gaussian_blur":
            crop = crop.filter(ImageFilter.GaussianBlur(radius=max(6, h * 0.6)))  # bán kính theo cỡ chữ
        else:  # pixelate: thu nhỏ rồi phóng lại bằng NEAREST → các ô vuông to cỡ nửa chiều cao chữ
            block = max(8, h // 2)
            small = crop.resize((max(1, crop.width // block), max(1, crop.height // block)), Image.BILINEAR)
            crop = small.resize(crop.size, Image.NEAREST)
        out.paste(crop, (x0, y0))
    return out


# =====================================================================
# 5. Pipeline end-to-end
# =====================================================================

def token_label(w: dict) -> str:
    """Nhãn hiển thị cho 1 token bị che: ưu tiên nhãn CRF, rồi regex, rồi nhãn có xác suất biên lớn nhất."""
    if w["crf_label"] != "O":
        return w["crf_label"][2:]
    if w["rx"] != "O":
        return w["rx"]
    best = max((l for l in w["p_marginal"] if l != "O"), key=w["p_marginal"].get)
    return best[2:]


class SafeScreen:
    """Nạp mô hình 1 lần, gọi nhiều lần: masked, regions = SafeScreen()(img)."""

    def __init__(self, crf_path=DEFAULT_CRF, risk_path=DEFAULT_RISK):
        self.crf = load_crf(crf_path)
        self.risk = RiskModel.load(risk_path)

    def __call__(self, img: Image.Image, mode: str = "black_box", pad: int = 4):
        words = run_ocr(img)                                  # 1. OCR
        if not words:
            return img.convert("RGB").copy(), []
        predict_tokens(self.crf, words)                       # 2+3. regex (bên trong) + CRF
        F = self.risk.score(risk_features(words))             # 4. hàm rủi ro
        flagged = [{"bbox": w["bbox"], "label": token_label(w), "F": float(f), "text": w["text"]}
                   for w, f in zip(words, F) if f >= self.risk.tau]
        regions = merge_regions(flagged)
        return mask(img, regions, mode, pad), regions         # 5. che


_default: SafeScreen | None = None


def pipeline(img: Image.Image, mode: str = "black_box"):
    """Ảnh → (ảnh đã che, [{bbox, labels, F, text}, ...]). Mô hình được nạp ở lần gọi đầu."""
    global _default
    if _default is None:
        _default = SafeScreen()
    return _default(img, mode)


# =====================================================================
# Demo trước/sau
# =====================================================================

def before_after(before: Image.Image, after: Image.Image, title: str, width: int = 540) -> Image.Image:
    """Ghép 2 ảnh cạnh nhau (đã thu nhỏ về `width`), có tiêu đề phía trên."""
    s = width / before.width
    size = (width, int(before.height * s))
    a, b = before.convert("RGB").resize(size, Image.LANCZOS), after.resize(size, Image.LANCZOS)
    head, gap = 44, 16
    canvas = Image.new("RGB", (2 * width + gap, size[1] + head), (252, 252, 251))
    canvas.paste(a, (0, head))
    canvas.paste(b, (width + gap, head))
    d = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except OSError:
        font = ImageFont.load_default()
    d.text((8, 10), "Trước", fill=(11, 11, 11), font=font)
    d.text((width + gap + 8, 10), title, fill=(11, 11, 11), font=font)
    return canvas


def main():
    ap = argparse.ArgumentParser(description="Hàm rủi ro F + che ảnh")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit", help="học λ, β trên val, chọn τ, đánh giá trên test")
    f.add_argument("--val", default="data/val")
    f.add_argument("--test", default="data/test")
    f.add_argument("--crf", default=str(DEFAULT_CRF))
    f.add_argument("--out", default=str(DEFAULT_RISK))
    f.add_argument("--min-recall", type=float, default=0.95)
    d = sub.add_parser("demo", help="lưu ảnh trước/sau cho N ảnh test")
    d.add_argument("--test", default="data/test")
    d.add_argument("--n", type=int, default=5)
    d.add_argument("--out", default="data/demo")
    args = ap.parse_args()

    if args.cmd == "fit":
        crf = load_crf(args.crf)
        model, info = fit_risk(crf, load_docs(args.val), args.min_recall)
        model.save(args.out)
        print(f"Tập val: {info['n_tokens']} token, {info['n_sensitive']} nhạy cảm")
        print("Hàm rủi ro đã học:  F = σ(" + " + ".join(f"{l:.3f}·{n}" for l, n in zip(model.lambdas, FEATURES))
              + f" + ({model.beta:.3f}))")
        print(f"Ngưỡng τ = {model.tau:.4f}  (F2 lớn nhất trong các ngưỡng giữ recall ≥ {args.min_recall} trên val)")
        plot_path = Path(args.out).with_name("risk_pr_curve.png")
        plot_pr_vs_tau(info["thresholds"], info["precision"], info["recall"], model.tau, args.min_recall, plot_path)
        print(f"Đã lưu {Path(args.out).resolve()} và biểu đồ {plot_path.resolve()}")
        evaluate_test(crf, model, load_docs(args.test))

    else:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        labels = sorted((Path(args.test) / "labels").glob("*.json"))
        # Lấy ảnh đầu tiên của mỗi loại màn hình, rồi thêm cho đủ N
        by_type: dict[str, Path] = {}
        for p in labels:
            by_type.setdefault(json.loads(p.read_text(encoding="utf-8"))["screen_type"], p)
        picks = list(by_type.values()) + [p for p in labels if p not in by_type.values()]
        for i, p in enumerate(picks[:args.n]):
            meta = json.loads(p.read_text(encoding="utf-8"))
            img = Image.open(Path(args.test) / meta["image"])
            mode = MASK_MODES[[2, 0, 1, 2, 0][i % 5]]
            masked, regions = pipeline(img, mode)
            before_after(img, masked, f"Sau ({mode})").save(out / f"{i + 1}_{meta['screen_type']}_{p.stem}.png")
            print(f"\n[{i + 1}] {p.stem} ({meta['screen_type']}, {mode}): che {len(regions)} vùng")
            for r in regions:
                print(f"     F={r['F']:.3f}  {'/'.join(r['labels']):<14} {r['text']}")
        print(f"\nĐã lưu ảnh trước/sau vào {out.resolve()}")


if __name__ == "__main__":
    main()
