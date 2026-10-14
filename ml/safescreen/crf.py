"""
safescreen/crf.py — Bước 4: CRF gắn nhãn chuỗi token OCR theo BIO.

CRF (Conditional Random Field) nhìn CẢ CHUỖI từ cùng lúc: nhãn của một từ phụ thuộc vào
  - đặc trưng của chính nó và các từ lân cận (state features), và
  - nhãn của từ ngay trước (transition), vd "B-NAME → I-NAME" rất hay gặp, "O → I-NAME" thì không.
Nhờ vậy CRF bắt được thứ regex không bắt được (họ tên, địa chỉ) và dùng ngữ cảnh để xác nhận / bác bỏ
kết quả regex (vd ngày giao dịch không phải ngày sinh).

Mỗi ảnh = 1 chuỗi (sequence), các từ theo thứ tự đọc từng hàng (ocr.sort_reading_order).

Chạy (đứng ở thư mục ml/):
    python -m safescreen.crf --data data/synth --test data/test
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections import Counter
from pathlib import Path

import joblib
import pandas as pd
import scipy.stats
import sklearn_crfsuite
from seqeval.metrics import classification_report, f1_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV

from safescreen.regex_rules import LOOSE, TIGHT, tag_words
from safescreen.text_utils import bo_dau, fold

ENTITY_TYPES = ["CCCD", "NAME", "DOB", "ADDRESS", "PHONE", "EMAIL", "MONEY", "ACCOUNT"]

# Từ khóa ngữ cảnh (đã bỏ dấu, chữ thường). Chỉ dùng từ khóa chung của giao diện tiếng Việt,
# KHÔNG chép câu mẫu của synth.py, để CRF không "học thuộc" dữ liệu tổng hợp.
KEYWORDS = {
    "NAME": ["ho ten", "ho va ten", "full name", "ten khach hang", "nguoi nhan", "chu tk",
             "chu tai khoan", "nhan tien tu", "xin chao", "noi dung"],
    "DOB": ["ngay sinh", "date of birth", "sinh nhat"],
    "ADDRESS": ["dia chi", "que quan", "thuong tru", "noi o", "place of"],
    "ACCOUNT": ["stk", "so tai khoan", "tai khoan"],
    "MONEY": ["so du", "so tien", "phi giao dich"],
    "CCCD": ["cccd", "can cuoc", "cmnd", "no."],
    "PHONE": ["sdt", "dien thoai"],
    "EMAIL": ["email", "mail"],
}
KW_WINDOW = 4  # xét 4 từ ngay trước


# =====================================================================
# 1. Đặc trưng (features)
#    Tất cả chỉ là thao tác chuỗi đơn giản → sau này viết lại được bằng Kotlin trên Android.
# =====================================================================

def word_shape(w: str) -> str:
    """Hình dạng từ: chữ hoa → X, chữ thường → x, số → d.  'Nguyễn' → 'Xxxxxx', '0912' → 'dddd'."""
    return "".join("X" if c.isupper() else "x" if c.islower() else "d" if c.isdigit() else c for c in w)


def short_shape(w: str) -> str:
    """Hình dạng rút gọn (gộp ký tự lặp): 'Xxxxxx' → 'Xx', '0912.345.678' → 'd.d.d'."""
    return re.sub(r"(.)\1+", r"\1", word_shape(w))


def annotate(sent: list[dict]) -> list[dict]:
    """
    Tính trước cho cả chuỗi những thứ cần nhìn ngữ cảnh rộng:
      rx, rx_score, rx_ctx : kết quả regex (bước 3) phủ lên từ này
      kw                   : các loại từ khóa xuất hiện trong KW_WINDOW từ ngay trước
    """
    for w, m in zip(sent, tag_words(sent)):
        w["rx"] = m.label if m else "O"
        w["rx_score"] = m.score if m else 0.0
        w["rx_ctx"] = bool(m and m.ctx)
    folded = [fold(w["text"]) for w in sent]
    for i, w in enumerate(sent):
        before = " ".join(folded[max(0, i - KW_WINDOW):i])
        before = re.sub(r"[^\w\s.]", " ", before)  # bỏ dấu câu: "họ và tên / full name:" → "ho va ten   full name "
        before = re.sub(r"\s+", " ", before)
        w["kw"] = [ent for ent, kws in KEYWORDS.items() if any(k in before for k in kws)]
    return sent


def _token_feats(w: dict, p: str) -> dict:
    """Đặc trưng cơ bản của 1 từ, dùng cho cả từ hiện tại (p='') và từ lân cận (p='-1:', '+2:', ...)."""
    t = w["text"]
    return {
        p + "lower": t.lower(),
        p + "fold": fold(t),           # dạng không dấu: "Nguyễn" và "NGUYEN" (OCR mất dấu) trùng nhau
        p + "istitle": t.istitle(),
        p + "isupper": t.isupper(),
        p + "isdigit": t.isdigit(),
        p + "shape": short_shape(t),
        p + "rx": w["rx"],
    }


def word2features(sent: list[dict], i: int) -> dict:
    """Đặc trưng của từ thứ i trong chuỗi (chuỗi phải được annotate() trước)."""
    w, t = sent[i], sent[i]["text"]
    low = t.lower()
    f = {
        "bias": 1.0,
        **_token_feats(w, ""),
        "len": min(len(t), 15),
        "prefix2": low[:2], "prefix3": low[:3],
        "suffix2": low[-2:], "suffix3": low[-3:],
        "shape_full": word_shape(t)[:12],
        "has_diacritic": bo_dau(t) != t,      # có dấu tiếng Việt (hoặc chữ đ)
        "has_digit": any(c.isdigit() for c in t),
        "has_at": "@" in t,
        "bol": i == 0 or sent[i - 1]["line"] != w["line"],               # đầu hàng
        "eol": i == len(sent) - 1 or sent[i + 1]["line"] != w["line"],   # cuối hàng
        # Kết quả regex: nhãn đã có trong _token_feats; thêm độ chặt và có từ khóa hay không
        "rx_tight": w["rx_score"] == TIGHT,
        "rx_loose": w["rx_score"] == LOOSE,
        "rx_ctx": w["rx_ctx"],
    }
    for ent in w["kw"]:
        f["kw:" + ent] = True
    # Từ lân cận ±1, ±2
    for off in (-2, -1, 1, 2):
        j = i + off
        p = f"{off:+d}:"
        if 0 <= j < len(sent):
            f.update(_token_feats(sent[j], p))
            f[p + "same_line"] = sent[j]["line"] == w["line"]
        else:
            f[p + ("BOS" if off < 0 else "EOS")] = True  # đầu / cuối chuỗi
    return f


def sent2features(sent: list[dict]) -> list[dict]:
    annotate(sent)
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent: list[dict]) -> list[str]:
    return [w["label"] for w in sent]


# =====================================================================
# 2. Dữ liệu
# =====================================================================

def load_docs(data_dir: str | Path) -> list[list[dict]]:
    """Đọc toàn bộ <data>/ocr/*.json (đầu ra ocr.py) → danh sách chuỗi từ."""
    files = sorted((Path(data_dir) / "ocr").glob("*.json"))
    return [json.loads(f.read_text(encoding="utf-8"))["words"] for f in files]


# =====================================================================
# 3. Train + tìm c1, c2
# =====================================================================

def seqeval_f1(y_true, y_pred) -> float:
    return f1_score(y_true, y_pred)  # F1 mức thực thể (micro), chuẩn của bài toán NER


def train(X_train, y_train, X_val, y_val, n_iter: int, seed: int, n_jobs: int):
    """
    lbfgs + RandomizedSearchCV tìm c1 (phạt L1, làm thưa: loại bớt đặc trưng vô ích) và
    c2 (phạt L2, giữ trọng số nhỏ, chống overfit).
    PredefinedSplit: chỉ dùng đúng 1 lần chia train/val cố định (thay vì k-fold) cho nhanh.
    refit=True: sau khi chọn xong, train lại trên train+val với c1, c2 tốt nhất.
    """
    crf = sklearn_crfsuite.CRF(algorithm="lbfgs", max_iterations=100, all_possible_transitions=True)
    params = {"c1": scipy.stats.expon(scale=0.5), "c2": scipy.stats.expon(scale=0.05)}
    split = PredefinedSplit([-1] * len(X_train) + [0] * len(X_val))  # -1 = chỉ để train, 0 = fold val
    search = RandomizedSearchCV(crf, params, cv=split, n_iter=n_iter, scoring=make_scorer(seqeval_f1),
                                n_jobs=n_jobs, random_state=seed, verbose=1, refit=True)
    search.fit(X_train + X_val, y_train + y_val)
    return search


# =====================================================================
# 4. Đánh giá + giải thích mô hình
# =====================================================================

def entity_of(label: str) -> str:
    return label[2:] if label != "O" else "O"


def evaluate(crf, X_test, y_test):
    y_pred = crf.predict(X_test)
    print("\n=== seqeval trên tập test (mức thực thể: đúng loại + đúng ranh giới mới tính) ===")
    print(classification_report(y_test, y_pred, digits=3))

    gold = [entity_of(l) for s in y_test for l in s]
    pred = [entity_of(l) for s in y_pred for l in s]
    order = ENTITY_TYPES + ["O"]
    cm = pd.crosstab(pd.Categorical(gold, order), pd.Categorical(pred, order),
                     rownames=["thật \\ dự đoán"], colnames=[""], dropna=False)
    print("=== Ma trận nhầm lẫn mức token (bỏ B-/I-) ===")
    print(cm.to_string())

    # Góc nhìn "che hay không": nhầm loại nhãn vẫn được che
    sens = sum(g != "O" for g in gold)
    hit = sum(g != "O" and p != "O" for g, p in zip(gold, pred))
    o_total = sum(g == "O" for g in gold)
    over = sum(g == "O" and p != "O" for g, p in zip(gold, pred))
    print(f"\nChe được token nhạy cảm: {hit / sens:.2%} ({hit}/{sens})"
          f"   |   Over-masking: {over / o_total:.2%} ({over}/{o_total})")
    return y_pred


def print_top_features(crf, n_state: int = 20, n_trans: int = 15):
    state = Counter(crf.state_features_)
    print(f"\n=== Top {n_state} đặc trưng trọng số DƯƠNG lớn nhất (đẩy mạnh về nhãn đó) ===")
    for (attr, label), wt in state.most_common(n_state):
        print(f"  {wt:8.3f}  {label:<10} {attr}")
    print("\n=== Top 10 đặc trưng trọng số ÂM (bác bỏ nhãn đó) ===")
    for (attr, label), wt in sorted(state.items(), key=lambda kv: kv[1])[:10]:
        print(f"  {wt:8.3f}  {label:<10} {attr}")

    trans = Counter(crf.transition_features_)
    print(f"\n=== Top {n_trans} transition hay gặp nhất (trọng số dương) ===")
    for (a, b), wt in trans.most_common(n_trans):
        print(f"  {wt:8.3f}  {a:<10} → {b}")
    print("\n=== Top 10 transition bị cấm (trọng số âm) ===")
    for (a, b), wt in sorted(trans.items(), key=lambda kv: kv[1])[:10]:
        print(f"  {wt:8.3f}  {a:<10} → {b}")


# =====================================================================
# 5. Lưu / nạp / dự đoán kèm xác suất
# =====================================================================

def save(crf, path: str | Path, meta: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(crf, path)
    path.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def load(path: str | Path = "models/crf.joblib"):
    return joblib.load(path)


def predict_tokens(crf, words: list[dict]) -> list[dict]:
    """
    Dự đoán cho 1 ảnh (danh sách từ từ ocr.run_ocr). Mỗi từ được thêm:
      crf_label   : nhãn BIO dự đoán (chuỗi tốt nhất theo Viterbi)
      p_marginal  : {nhãn: xác suất} — xác suất BIÊN của từng nhãn tại từ này (predict_marginals)
      p_sensitive : 1 - P(O) = xác suất từ này thuộc BẤT KỲ thực thể nhạy cảm nào → p_crf trong hàm rủi ro
    """
    X = [sent2features(words)]
    labels, marginals = crf.predict(X)[0], crf.predict_marginals(X)[0]
    for w, lab, m in zip(words, labels, marginals):
        w["crf_label"] = lab
        w["p_marginal"] = m
        w["p_sensitive"] = 1.0 - m.get("O", 0.0)
    return words


# =====================================================================
# 6. Chương trình chính
# =====================================================================

def main():
    ap = argparse.ArgumentParser(description="Train CRF gắn nhãn BIO cho token OCR")
    ap.add_argument("--data", default="data/synth", help="dữ liệu train (đã chạy ocr.py)")
    ap.add_argument("--test", default="data/test", help="dữ liệu test (đã chạy ocr.py)")
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--n-iter", type=int, default=20, help="số bộ (c1, c2) thử ngẫu nhiên")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="models/crf.joblib")
    args = ap.parse_args()

    docs = load_docs(args.data)
    random.Random(args.seed).shuffle(docs)
    n_val = int(len(docs) * args.val_ratio)
    val_docs, train_docs, test_docs = docs[:n_val], docs[n_val:], load_docs(args.test)
    print(f"Train {len(train_docs)} ảnh | Val {len(val_docs)} ảnh | Test {len(test_docs)} ảnh")

    t0 = time.perf_counter()
    X_train, y_train = [sent2features(s) for s in train_docs], [sent2labels(s) for s in train_docs]
    X_val, y_val = [sent2features(s) for s in val_docs], [sent2labels(s) for s in val_docs]
    X_test, y_test = [sent2features(s) for s in test_docs], [sent2labels(s) for s in test_docs]
    print(f"Tính đặc trưng: {time.perf_counter() - t0:.0f}s")

    t0 = time.perf_counter()
    search = train(X_train, y_train, X_val, y_val, args.n_iter, args.seed, args.n_jobs)
    print(f"Tìm tham số + train: {time.perf_counter() - t0:.0f}s")
    print(f"c1, c2 tốt nhất: {search.best_params_}  |  F1 trên val: {search.best_score_:.4f}")
    results = pd.DataFrame(search.cv_results_)[["param_c1", "param_c2", "mean_test_score"]]
    print(results.sort_values("mean_test_score", ascending=False).head(5).to_string(index=False))

    crf = search.best_estimator_
    evaluate(crf, X_test, y_test)
    print_top_features(crf)

    save(crf, args.out, {"best_params": {k: float(v) for k, v in search.best_params_.items()},
                         "val_f1": float(search.best_score_),
                         "test_f1": float(f1_score(y_test, crf.predict(X_test))),
                         "n_train": len(train_docs), "n_val": len(val_docs), "n_test": len(test_docs),
                         "n_features": len(crf.state_features_)})
    print(f"\nĐã lưu mô hình → {Path(args.out).resolve()}")

    # Minh hoạ predict_marginals trên 1 ảnh test
    demo = predict_tokens(load(args.out), test_docs[0])
    print("\nVí dụ xác suất biên (ảnh test đầu tiên, các từ CRF cho là nhạy cảm):")
    for w in demo:
        if w["p_sensitive"] > 0.5:
            top = max(w["p_marginal"], key=w["p_marginal"].get)
            print(f"  {w['text']:<22} {w['crf_label']:<10} p_sensitive={w['p_sensitive']:.3f}"
                  f"  (P({top})={w['p_marginal'][top]:.3f})")


if __name__ == "__main__":
    main()
