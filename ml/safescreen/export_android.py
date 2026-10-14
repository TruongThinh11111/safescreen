"""
safescreen/export_android.py — Xuất mô hình cho Android + kiểm chứng bản Python thuần + golden test.

Các bước (đứng ở thư mục ml/):  python -m safescreen.export_android
    1. Đọc trọng số CRF CHÍNH XÁC từ file nhị phân crfsuite → models/crf_model.json (bỏ |w| < 1e-6)
    2. Xuất regex (cú pháp Java), từ khóa, hằng số, λ/β/τ          → models/config.json
    3. Chép vie/eng.traineddata đã dùng khi OCR                      → models/tessdata/
    4. So crf_pure.py với crfsuite + module gốc trên TOÀN BỘ tập test (nhãn, marginal, F, regex)
    5. Chạy mọi regex trên JVM thật (java.util.regex) và so với Python
    6. Xuất 20 ảnh mẫu                                                → models/golden_test.json

Vì sao phải đọc file nhị phân? crf.state_features_ của sklearn-crfsuite được lấy từ bản dump DẠNG CHỮ
của crfsuite, in bằng "%f" → chỉ còn 6 chữ số thập phân. Sai số ±5e-7 mỗi trọng số cộng dồn qua ~60
thuộc tính/token có thể làm marginal lệch quá 1e-6. File nhị phân lưu float64 đầy đủ.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import random
import re
import shutil
import struct
import subprocess
import tempfile
import time
from datetime import date
from pathlib import Path

from safescreen import regex_rules as rr
from safescreen.crf import KEYWORDS, KW_WINDOW, load as load_crf, load_docs, sent2features
from safescreen.crf_pure import PureModel, fold, java_to_python_regex
from safescreen.ocr import ML_DIR, WINDOWS_TESSERACT
from safescreen.risk_mask import (ALL_KEYWORDS, CTX_WINDOW, FEATURES, RiskModel, merge_regions, risk_features,
                                  token_label)

MODELS = ML_DIR / "models"
JAVA_CANDIDATES = [Path(r"C:\Program Files\Android\Android Studio\jbr\bin\java.exe"), Path("java")]


# =====================================================================
# 1. Đọc file mô hình nhị phân của crfsuite (định dạng CRF1d: "lCRF" / "FOMC")
# =====================================================================

def read_crfsuite_model(path: Path):
    """
    Cấu trúc (little-endian, theo crfsuite/lib/crf/src/crf1d_model.c):
      header 48 byte: magic "lCRF", size, type "FOMC", version, num_features (=0, không dùng), num_labels, num_attrs,
                      off_features, off_labels, off_attrs, off_labelrefs, off_attrrefs
      FEAT chunk   : "FEAT", size, num, rồi num × (type u32, src u32, dst u32, weight f64) = 20 byte/feature
                     type 0 = state (src = id thuộc tính, dst = id nhãn), type 1 = transition (nhãn → nhãn)
      CQDB chunk   : bảng chuỗi; mảng "backward" ở bwd_offset cho offset của bản ghi (id i32, ksize u32, key\\0)
    """
    buf = Path(path).read_bytes()
    (magic, _, mtype, version, _, n_lab, n_attr,
     off_feat, off_lab, off_attr, _, _) = struct.unpack_from("<4sI4sIIIIIIIII", buf, 0)
    assert magic == b"lCRF" and mtype == b"FOMC", f"không phải mô hình CRF1d: {magic} {mtype}"

    def strings(off: int, n: int) -> list[str]:
        chunk, _, _, _, bwd_size, bwd_offset = struct.unpack_from("<4sIIIII", buf, off)
        assert chunk == b"CQDB" and bwd_size >= n
        out = []
        for i in range(n):
            rec = struct.unpack_from("<I", buf, off + bwd_offset + 4 * i)[0]
            rid, ksize = struct.unpack_from("<iI", buf, off + rec)
            assert rid == i
            out.append(buf[off + rec + 8: off + rec + 8 + ksize].rstrip(b"\x00").decode("utf-8"))
        return out

    labels, attrs = strings(off_lab, n_lab), strings(off_attr, n_attr)
    # Lưu ý: crfsuite để num_features = 0 trong header; số feature thật nằm trong chunk FEAT
    chunk, size, num = struct.unpack_from("<4sII", buf, off_feat)
    assert chunk == b"FEAT" and size == 12 + 20 * num
    state, trans = {}, {}
    for k in range(num):
        ftype, src, dst, w = struct.unpack_from("<IIId", buf, off_feat + 12 + 20 * k)
        if ftype == 0:
            state[(attrs[src], labels[dst])] = w
        elif ftype == 1:
            trans[(labels[src], labels[dst])] = w
        else:
            raise ValueError(f"loại feature lạ: {ftype}")
    return {"version": version, "labels": labels, "state": state, "trans": trans}


def check_against_dump(model, crf):
    """Trọng số đọc từ file nhị phân, làm tròn 6 chữ số, phải trùng bản dump dạng chữ của sklearn-crfsuite."""
    assert set(model["labels"]) == set(crf.classes_)
    for name, ours, theirs in [("state", model["state"], crf.state_features_),
                               ("transition", model["trans"], crf.transition_features_)]:
        missing = set(theirs) - set(ours)
        worst = max(abs(ours[k] - v) for k, v in theirs.items() if k in ours)
        assert not missing and worst <= 5.0001e-7, f"{name}: thiếu {len(missing)}, lệch {worst}"
        print(f"   {name:<10}: {len(ours):>6} trọng số, khớp bản dump (lệch lớn nhất {worst:.1e} do làm tròn %f)")


def crf_json(model, prune: float) -> dict:
    state, trans = {}, {}
    for (attr, lab), w in model["state"].items():
        if abs(w) >= prune:
            state.setdefault(attr, {})[lab] = w
    for (a, b), w in model["trans"].items():
        if abs(w) >= prune:
            trans.setdefault(a, {})[b] = w
    return {
        "format": "safescreen-crf/1",
        "crfsuite_model_version": model["version"],
        "prune_abs_weight_below": prune,
        "labels": model["labels"],
        "attribute_encoding": "chuỗi → 'khóa:giá_trị' = 1.0; bool/số → 'khóa' = float(giá trị). "
                              "Điểm nhãn y tại vị trí t = Σ giá_trị × state_features[thuộc_tính][y]; "
                              "chuỗi: + transition_features[y_(t-1)][y_t]; không có trọng số đầu/cuối chuỗi.",
        "state_features": state,
        "transition_features": trans,
    }


def dump_compact(obj) -> str:
    # float của Python được in dạng ngắn nhất mà đọc lại vẫn đúng y hệt (round-trip) → không mất độ chính xác
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


# =====================================================================
# 2. config.json
# =====================================================================

def py_to_java_regex(p: re.Pattern) -> dict:
    src = p.pattern
    java = src.replace("(?P<", "(?<")
    assert "(?P=" not in src and "\\Z" not in src, "cú pháp chỉ Python có"
    assert java_to_python_regex(java) == src, "đổi qua đổi lại phải ra đúng pattern gốc"
    return {"pattern": java, "flags": ["CASE_INSENSITIVE"] if p.flags & re.IGNORECASE else []}


def build_config(risk: RiskModel, tessdata: dict, today: date) -> dict:
    return {
        "format": "safescreen-config/1",
        "reference_date": today.isoformat(),
        "unicode": "Mọi chuỗi OCR phải chuẩn hoá NFC trước khi xử lý (xem crf_pure.py [P1]).",
        "regex": {
            "syntax": "java.util.regex. JVM: compile với Pattern.UNICODE_CHARACTER_CLASS (+ CASE_INSENSITIVE nếu "
                      "có trong flags). Android (ICU): Unicode là mặc định, KHÔNG chèn (?U) vào pattern.",
            "patterns": {name: py_to_java_regex(p) for name, p in rr.PATTERNS.items()},
            "rule_order": ["CCCD", "DOB", "PHONE", "EMAIL", "MONEY", "ACCOUNT"],
            "tight": rr.TIGHT, "loose": rr.LOOSE, "window": rr.WINDOW,
            "keywords": rr.KEYWORDS, "dob_negative": rr.DOB_NEGATIVE,
            "province_codes": sorted(rr.PROVINCE_CODES), "mobile_prefixes": sorted(rr.MOBILE_PREFIXES),
            "hotline_prefixes": list(rr.HOTLINE_PREFIXES), "priority": rr.PRIORITY,
            "digit_lookalikes": {"O": "0", "o": "0", "l": "1", "I": "1", "|": "1"},
            "dashes": {"–": "-", "—": "-", "−": "-"},
        },
        "crf_features": {
            "keywords": KEYWORDS, "kw_window": KW_WINDOW,
            "patterns": {"punct_to_space": r"[^\w\s.]", "multi_space": r"\s+", "repeated_char": r"(.)\1+"},
            "len_max": 15, "shape_full_max": 12, "neighbor_offsets": [-2, -1, 1, 2],
        },
        "risk": {"features": FEATURES, "lambdas": risk.lambdas, "beta": risk.beta, "tau": risk.tau,
                 "ctx_window": CTX_WINDOW, "keywords": ALL_KEYWORDS},
        "mask": {"default_mode": "black_box", "pad_px": 4, "merge_min_v_overlap": 0.5, "merge_max_h_gap": 0.8},
        "ocr": {"lang": "vie+eng", "psm": 3, "target_width": 1440,
                "normalize_background": {"block": 8, "median_window": 7},
                "reading_order": "gom theo hàng ngang: tâm dọc của từ nằm trong [y0, y1] của hàng hiện tại",
                **tessdata},
    }


# =====================================================================
# 3. Tessdata
# =====================================================================

def export_tessdata(out_dir: Path) -> dict:
    """Ghi lại CHÍNH XÁC file đã dùng khi OCR (ocr.setup_tesseract chọn ml/tessdata vì có vie.traineddata)."""
    src_dir = ML_DIR / "tessdata"
    out_dir.mkdir(parents=True, exist_ok=True)
    tess = str(WINDOWS_TESSERACT) if WINDOWS_TESSERACT.exists() else "tesseract"
    version = subprocess.run([tess, "--version"], capture_output=True, text=True).stdout.splitlines()
    combine = Path(tess).with_name("combine_tessdata.exe")
    files = {}
    for lang in ("vie", "eng"):
        src = src_dir / f"{lang}.traineddata"
        shutil.copy2(src, out_dir / src.name)
        info = {"source_path": str(src), "size_bytes": src.stat().st_size,
                "sha256": hashlib.sha256(src.read_bytes()).hexdigest()}
        if combine.exists():
            d = subprocess.run([str(combine), "-d", str(src)], capture_output=True, text=True)
            text = d.stdout + d.stderr
            info["version_string"] = next((l.split("Version:", 1)[1].strip() for l in text.splitlines()
                                           if "Version:" in l), None)
            lstm = re.search(r"\d+:lstm:size=(\d+)", text)
            info["lstm_size_bytes"] = int(lstm.group(1)) if lstm else None
        files[lang] = info
    files["vie"]["origin"] = ("tessdata_best — https://github.com/tesseract-ocr/tessdata_best/raw/main/"
                              "vie.traineddata (tải ngày 2026-10-14)")
    files["eng"]["origin"] = (r"đi kèm bộ cài Tesseract (C:\Program Files\Tesseract-OCR\tessdata), mạng LSTM nhỏ "
                              "~0,4 MB → nhiều khả năng là bản tessdata_fast")
    return {"tesseract_version": version[0] if version else None,
            "leptonica_version": version[1].strip() if len(version) > 1 else None,
            "tessdata": files}


# =====================================================================
# 4. Kiểm chứng crf_pure trên toàn bộ tập test
# =====================================================================

def verify(crf, risk: RiskModel, pure: PureModel, docs, today: date) -> dict:
    stats = {"docs": 0, "tokens": 0, "label_mismatch": 0, "feature_mismatch": 0, "regex_mismatch": 0,
             "max_marginal_diff": 0.0, "max_F_diff": 0.0, "mask_mismatch": 0, "region_mismatch": 0,
             "order_mismatch": 0}
    for words in docs:
        # --- bản gốc: crf.py + regex_rules + crfsuite + risk_mask
        ref = [dict(w) for w in words]
        X = sent2features(ref)                         # annotate() bên trong dùng tag_words(…, today mặc định)
        y_ref = crf.predict_single(X)
        m_ref = crf.predict_marginals_single(X)
        for w, m in zip(ref, m_ref):
            w["p_sensitive"], w["p_marginal"], w["crf_label"] = 1.0 - m["O"], m, None
        for w, lab in zip(ref, y_ref):
            w["crf_label"] = lab
        F_ref = risk.score(risk_features(ref))
        regions_ref = merge_regions([{"bbox": w["bbox"], "label": token_label(w), "F": float(f), "text": w["text"]}
                                     for w, f in zip(ref, F_ref) if f >= risk.tau])
        # --- bản thuần
        out = pure.process([dict(w) for w in words], today)
        stats["docs"] += 1
        stats["tokens"] += len(words)
        for i, t in enumerate(out["tokens"]):
            stats["label_mismatch"] += t["label"] != y_ref[i]
            stats["feature_mismatch"] += t["features"] != X[i]
            stats["regex_mismatch"] += (t["rx"], t["rx_score"], t["rx_ctx"]) != (
                ref[i]["rx"], ref[i]["rx_score"], ref[i]["rx_ctx"])
            stats["max_marginal_diff"] = max(stats["max_marginal_diff"],
                                             max(abs(t["marginals"][l] - m_ref[i][l]) for l in pure.labels))
            stats["max_F_diff"] = max(stats["max_F_diff"], abs(t["F"] - F_ref[i]))
            stats["mask_mismatch"] += t["mask"] != bool(F_ref[i] >= risk.tau)
        stats["region_mismatch"] += [(r["bbox"], r["labels"]) for r in out["regions"]] != \
                                    [(r["bbox"], r["labels"]) for r in regions_ref]
        # sort_reading_order: xáo trộn đầu vào rồi xếp lại phải ra đúng thứ tự/số dòng của ocr.py
        shuffled = [{k: w[k] for k in ("text", "bbox", "conf")} for w in words]
        random.Random(len(words)).shuffle(shuffled)
        resorted = pure.sort_reading_order(shuffled)
        stats["order_mismatch"] += [(w["text"], w["bbox"], w["line"]) for w in resorted] != \
                                   [(w["text"], w["bbox"], w["line"]) for w in words]
    return stats


# =====================================================================
# 5. Chạy regex trên JVM thật
# =====================================================================

JAVA_CHECK = r"""
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;
import java.util.regex.*;

public class RegexCheck {
    public static void main(String[] a) throws Exception {
        Base64.Decoder dec = Base64.getDecoder();
        List<String> pats = Files.readAllLines(Path.of(a[0]));
        List<String> texts = Files.readAllLines(Path.of(a[1]));
        StringBuilder out = new StringBuilder();
        for (String line : pats) {
            String[] p = line.split("\t", -1);
            int flags = Pattern.UNICODE_CHARACTER_CLASS;
            if (p[2].contains("CASE_INSENSITIVE")) flags |= Pattern.CASE_INSENSITIVE;
            Pattern pat = Pattern.compile(new String(dec.decode(p[1]), StandardCharsets.UTF_8), flags);
            for (int i = 0; i < texts.size(); i++) {
                Matcher m = pat.matcher(new String(dec.decode(texts.get(i)), StandardCharsets.UTF_8));
                while (m.find()) out.append(p[0]).append('\t').append(i).append('\t')
                                    .append(m.start()).append('\t').append(m.end()).append('\n');
            }
        }
        Files.writeString(Path.of(a[2]), out.toString(), StandardCharsets.UTF_8);
    }
}
"""


def check_regex_on_jvm(config: dict, pure: PureModel, docs) -> dict | None:
    java = next((str(j) for j in JAVA_CANDIDATES if j.name == "java" or j.exists()), None)
    texts = []
    for words in docs:
        text, _ = pure.words_to_text(words)
        texts += [text, pure.normalize_ocr(text), fold(text)]
    patterns = dict(config["regex"]["patterns"])
    patterns.update({f"feature_{k}": {"pattern": v, "flags": []}
                     for k, v in config["crf_features"]["patterns"].items()})
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "RegexCheck.java").write_text(JAVA_CHECK, encoding="utf-8")
        b64 = lambda s: base64.b64encode(s.encode("utf-8")).decode("ascii")
        (tmp / "pats.tsv").write_text("".join(f"{n}\t{b64(d['pattern'])}\t{','.join(d['flags'])}\n"
                                              for n, d in patterns.items()), encoding="utf-8")
        (tmp / "texts.tsv").write_text("".join(b64(t) + "\n" for t in texts), encoding="utf-8")
        try:
            r = subprocess.run([java, str(tmp / "RegexCheck.java"), str(tmp / "pats.tsv"), str(tmp / "texts.tsv"),
                                str(tmp / "out.tsv")], capture_output=True, text=True, timeout=600)
        except (OSError, TypeError):
            return None
        if r.returncode != 0:
            raise RuntimeError("Java báo lỗi:\n" + r.stderr)
        jvm = set(tuple(l.split("\t")) for l in (tmp / "out.tsv").read_text(encoding="utf-8").splitlines())
    py = set()
    for name, d in patterns.items():
        p = re.compile(java_to_python_regex(d["pattern"]), re.IGNORECASE if "CASE_INSENSITIVE" in d["flags"] else 0)
        for i, t in enumerate(texts):
            py |= {(name, str(i), str(m.start()), str(m.end())) for m in p.finditer(t)}
    ver = subprocess.run([java, "-version"], capture_output=True, text=True).stderr.splitlines()[0]
    return {"java": ver, "patterns": len(patterns), "texts": len(texts), "matches_python": len(py),
            "matches_jvm": len(jvm), "only_python": sorted(py - jvm)[:5], "only_jvm": sorted(jvm - py)[:5]}


# =====================================================================
# 6. Golden test
# =====================================================================

def golden(pure: PureModel, test_dir: Path, today: date, n: int) -> dict:
    """n ảnh, lấy đều theo loại màn hình. Giá trị kỳ vọng = đầu ra crf_pure (đã kiểm chứng khớp crfsuite)."""
    files = sorted((test_dir / "ocr").glob("*.json"))
    by_type: dict[str, list] = {}
    for f in files:
        by_type.setdefault(json.loads(f.read_text(encoding="utf-8"))["screen_type"], []).append(f)
    picks, k = [], 0
    while len(picks) < n:  # xoay vòng các loại màn hình
        for files_t in by_type.values():
            if k < len(files_t) and len(picks) < n:
                picks.append(files_t[k])
        k += 1
    samples = []
    for f in picks:
        doc = json.loads(f.read_text(encoding="utf-8"))
        words = [{"text": w["text"], "bbox": w["bbox"], "conf": w["conf"], "line": w["line"]} for w in doc["words"]]
        out = pure.process([dict(w) for w in words], today)
        text, spans = pure.words_to_text(out["tokens"])
        samples.append({
            "id": f.stem, "screen_type": doc["screen_type"], "image": doc["image"],
            "text": text,
            "regex_matches": pure.find_entities(text, today),
            "tokens": [{
                "text": t["text"], "bbox": t["bbox"], "conf": t["conf"], "line": t["line"], "span": spans[i],
                "regex": {"label": t["rx"], "score": t["rx_score"], "ctx": t["rx_ctx"]}, "kw": t["kw"],
                "features": t["features"], "attributes": t["attributes"],
                "label": t["label"], "marginals": t["marginals"], "p_sensitive": t["p_sensitive"],
                "risk_x": t["risk_x"], "F": t["F"], "mask": t["mask"],
            } for i, t in enumerate(out["tokens"])],
            "regions": out["regions"],
        })
    return {
        "format": "safescreen-golden/1",
        "reference_date": today.isoformat(),
        "how_to_use": "Đầu vào mỗi mẫu: tokens[].{text,bbox,conf,line} (đã theo thứ tự đọc). Chạy bản Kotlin với "
                      "reference_date làm 'hôm nay' rồi so: text, regex_matches, features/attributes (bằng tuyệt "
                      "đối), label (bằng tuyệt đối), marginals/p_sensitive/F (sai khác < 1e-6), mask, regions.",
        "tolerance": 1e-6,
        "samples": samples,
    }


# =====================================================================
# Chương trình chính
# =====================================================================

def kb(n: int) -> str:
    return f"{n:,} byte ({n / 1024:,.1f} KB)"


def main():
    ap = argparse.ArgumentParser(description="Xuất mô hình sang JSON cho Android và kiểm chứng")
    ap.add_argument("--crf", default=str(MODELS / "crf.joblib"))
    ap.add_argument("--risk", default=str(MODELS / "risk.json"))
    ap.add_argument("--test", default=str(ML_DIR / "data" / "test"))
    ap.add_argument("--prune", type=float, default=1e-6)
    ap.add_argument("--golden", type=int, default=20)
    args = ap.parse_args()
    today = date.today()  # PORT [P9]: ghi vào config/golden làm mốc cố định

    print("1. Đọc trọng số chính xác từ file nhị phân crfsuite")
    crf = load_crf(args.crf)
    model = read_crfsuite_model(Path(crf.modelfile.name))
    check_against_dump(model, crf)
    full, pruned = crf_json(model, 0.0), crf_json(model, args.prune)
    full_bytes, pruned_text = len(dump_compact(full).encode("utf-8")), dump_compact(pruned)
    (MODELS / "crf_model.json").write_text(pruned_text, encoding="utf-8")
    n_full = sum(len(v) for v in full["state_features"].values()) + sum(len(v) for v in full["transition_features"].values())
    n_pruned = sum(len(v) for v in pruned["state_features"].values()) + sum(len(v) for v in pruned["transition_features"].values())
    print(f"   crf_model.json TRƯỚC khi bỏ |w| < {args.prune:g}: {kb(full_bytes)}, {n_full} trọng số")
    print(f"   crf_model.json SAU   khi bỏ |w| < {args.prune:g}: {kb(len(pruned_text.encode('utf-8')))}, "
          f"{n_pruned} trọng số (bỏ {n_full - n_pruned})")
    print(f"   (so sánh: file nhị phân crfsuite {kb(Path(crf.modelfile.name).stat().st_size)}, "
          f"crf.joblib {kb(Path(args.crf).stat().st_size)})")

    print("\n2–3. config.json + tessdata")
    tess = export_tessdata(MODELS / "tessdata")
    config = build_config(RiskModel.load(args.risk), tess, today)
    (MODELS / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"   config.json: {kb((MODELS / 'config.json').stat().st_size)}, {len(config['regex']['patterns'])} regex")
    print(f"   {tess['tesseract_version']} / {tess['leptonica_version']}")
    for lang, info in tess["tessdata"].items():
        print(f"   {lang}: {info['source_path']}  {kb(info['size_bytes'])}  sha256={info['sha256'][:16]}…")
        print(f"        version={info.get('version_string')}  lstm={info.get('lstm_size_bytes'):,} byte")

    print("\n4. Kiểm chứng crf_pure (chỉ đọc 2 file JSON) với crfsuite trên toàn bộ tập test")
    pure = PureModel(MODELS / "crf_model.json", MODELS / "config.json")
    docs = load_docs(args.test)
    t0 = time.perf_counter()
    stats = verify(crf, RiskModel.load(args.risk), pure, docs, today)
    print(f"   {stats['docs']} ảnh, {stats['tokens']} token, {time.perf_counter() - t0:.0f}s")
    for k in ("label_mismatch", "feature_mismatch", "regex_mismatch", "mask_mismatch", "region_mismatch",
              "order_mismatch"):
        print(f"   {k:<18}: {stats[k]}")
    print(f"   max |Δ marginal|  : {stats['max_marginal_diff']:.3e}")
    print(f"   max |Δ F|         : {stats['max_F_diff']:.3e}")
    ok = (all(stats[k] == 0 for k in ("label_mismatch", "feature_mismatch", "regex_mismatch", "mask_mismatch",
                                      "region_mismatch", "order_mismatch"))
          and stats["max_marginal_diff"] < 1e-6 and stats["max_F_diff"] < 1e-6)
    print(f"   ⇒ {'ĐẠT' if ok else 'KHÔNG ĐẠT'}")

    print("\n5. Chạy regex trên JVM")
    jvm = check_regex_on_jvm(config, pure, docs)
    if jvm is None:
        print("   (không tìm thấy java — bỏ qua)")
    else:
        same = jvm["matches_python"] == jvm["matches_jvm"] and not jvm["only_python"] and not jvm["only_jvm"]
        print(f"   {jvm['java']}: {jvm['patterns']} regex × {jvm['texts']} văn bản → Python {jvm['matches_python']} "
              f"khớp, JVM {jvm['matches_jvm']} khớp ⇒ {'GIỐNG HỆT' if same else 'KHÁC NHAU'}")
        if not same:
            print("   chỉ Python:", jvm["only_python"], "\n   chỉ JVM:", jvm["only_jvm"])

    print(f"\n6. Golden test ({args.golden} ảnh)")
    g = golden(pure, Path(args.test), today, args.golden)
    (MODELS / "golden_test.json").write_text(json.dumps(g, ensure_ascii=False), encoding="utf-8")
    n_tok = sum(len(s["tokens"]) for s in g["samples"])
    types = sorted({s["screen_type"] for s in g["samples"]})
    print(f"   golden_test.json: {kb((MODELS / 'golden_test.json').stat().st_size)}, {n_tok} token, loại: {types}")


if __name__ == "__main__":
    main()
