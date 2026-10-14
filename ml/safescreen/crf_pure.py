"""
safescreen/crf_pure.py — Toàn bộ phần suy luận SAU OCR viết bằng Python thuần.
Đây là BẢN MẪU để port sang Kotlin: mỗi hàm ở đây nên có 1 hàm tương ứng 1–1 bên Kotlin.

Chỉ dùng thư viện chuẩn (json, math, re, unicodedata, datetime) và CHỈ đọc 2 file:
    models/crf_model.json — trọng số CRF: state {thuộc tính: {nhãn: w}}, transition {nhãn trước: {nhãn sau: w}}
    models/config.json    — regex (cú pháp Java), từ khóa, hằng số, λ/β/τ của hàm rủi ro
Không dùng sklearn-crfsuite, không import module nào khác của safescreen.

Luồng xử lý 1 ảnh (danh sách từ OCR {text, bbox, conf}):
    0. sort_reading_order   — xếp từ theo hàng ngang, gán "line"
    1. words_to_text        — ghép thành văn bản, nhớ vị trí từng từ
    2. regex                — normalize_ocr → 6 luật → resolve → kết quả regex cho từng từ
    3. đặc trưng CRF        — word2features → to_attributes (thuộc tính kiểu crfsuite)
    4. viterbi + marginals  — nhãn tốt nhất + xác suất biên (forward–backward)
    5. hàm rủi ro F, quyết định che, gộp vùng

CÁC CHỖ DỄ SAI KHI PORT SANG KOTLIN — tìm chữ "PORT:" trong file. Tóm tắt:
  [P1] Unicode: chuẩn hoá NFC chữ OCR TRƯỚC mọi bước. Đặc trưng được train trên chuỗi NFC
       ("lower:nguyễn"); nếu Tesseract trả về NFD thì tra trọng số trượt hết mà KHÔNG báo lỗi.
       Kotlin: java.text.Normalizer.normalize(s, Normalizer.Form.NFC)
  [P2] fold() bỏ dấu: 'đ'/'Đ' KHÔNG tự tách dấu khi NFD (là chữ cái riêng, không phải d + dấu)
       → phải thay tay TRƯỚC khi NFD. Sau NFD bỏ mọi ký tự loại Mn (\\p{Mn}), RỒI mới viết thường.
  [P3] Viết thường/hoa không phụ thuộc locale: Kotlin dùng lowercase()/uppercase() (Locale.ROOT).
       KHÔNG dùng toLowerCase() mặc định — máy cài tiếng Thổ Nhĩ Kỳ biến "I" thành "ı".
  [P4] Độ dài & cắt chuỗi: Python đếm theo code point, Kotlin String theo UTF-16. Chữ Việt đều nằm
       trong BMP nên giống nhau; emoji (ngoài BMP) sẽ lệch → nếu cần, làm việc trên codePoints().
  [P5] isupper/istitle/isdigit theo đúng định nghĩa của Python (viết tay bên dưới), không dùng all{}.
  [P6] Mã hoá thuộc tính crfsuite: chuỗi → "khóa:giá_trị" = 1.0; bool/số → "khóa" = float(giá trị).
       Bản thân khóa có thể chứa ':' ("-1:fold"), nên KHÔNG tách ngược tên thuộc tính theo ':'.
  [P7] Regex: config lưu cú pháp Java. Trên JVM (unit test) phải compile với
       Pattern.UNICODE_CHARACTER_CLASS để \\w \\d \\s hiểu Unicode như Python. Android dùng ICU, mặc định
       đã Unicode; đừng chèn "(?U)" vào chuỗi pattern (ICU có thể báo lỗi cú pháp) — truyền cờ riêng.
  [P8] resolve(): sort phải ỔN ĐỊNH (Kotlin sortedWith là ổn định) và giữ đúng thứ tự ứng viên
       (theo thứ tự luật, trong mỗi luật theo thứ tự tìm thấy).
  [P9] Ngày "hôm nay" cho luật DOB (loại ngày tương lai) — truyền vào, đừng gọi now() rải rác.
  [P10] Forward–backward dùng log-sum-exp với Double; sigmoid viết dạng ổn định số.
"""
from __future__ import annotations

import json
import math
import re
import unicodedata
from datetime import date
from pathlib import Path


# =====================================================================
# Xử lý chuỗi
# =====================================================================

def nfc(s: str) -> str:
    """PORT [P1]: gọi cho MỌI chuỗi từ OCR trước khi xử lý."""
    return unicodedata.normalize("NFC", s)


def bo_dau(s: str) -> str:
    """
    Bỏ dấu tiếng Việt: 'Nguyễn Văn Đức' → 'Nguyen Van Duc'.
    PORT [P2]: thay đ/Đ TRƯỚC (NFD không tách chúng), rồi NFD, rồi bỏ ký tự loại Mn.
    """
    s = s.replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def fold(s: str) -> str:
    """Bỏ dấu rồi viết thường. PORT [P3]: Kotlin bo_dau(s).lowercase() — KHÔNG toLowerCase()."""
    return bo_dau(s).lower()


def is_titlecase_char(c: str) -> bool:
    """
    Ký tự loại Unicode Lt ('ǅ', 'ǈ'...). Kotlin: Char.isTitleCase().
    PORT [P5]: KHÔNG viết c.istitle() trong Python — với chuỗi 1 ký tự, "L".istitle() là True
    (chuỗi "L" đúng là "viết hoa chữ đầu"), nên mọi chữ hoa sẽ bị coi là Lt.
    """
    return unicodedata.category(c) == "Lt"


def py_isupper(s: str) -> bool:
    """= str.isupper(): có ≥1 chữ có hoa/thường, và không có chữ thường/chữ dạng tiêu đề nào. PORT [P5]"""
    has_cased = False
    for c in s:
        if c.islower() or is_titlecase_char(c):
            return False
        if c.isupper():
            has_cased = True
    return has_cased


def py_istitle(s: str) -> bool:
    """
    = str.istitle(): chữ HOA chỉ được đứng sau ký tự không có hoa/thường, chữ thường chỉ đứng sau chữ có
    hoa/thường; phải có ≥1 chữ. Vd 'Nguyễn' True, 'Hà-Nội' True, 'No.:075' True, 'NGUYỄN' False. PORT [P5]
    """
    cased = prev_cased = False
    for c in s:
        if c.isupper() or is_titlecase_char(c):
            if prev_cased:
                return False
            prev_cased = cased = True
        elif c.islower():
            if not prev_cased:
                return False
            prev_cased = cased = True
        else:
            prev_cased = False
    return cased


def py_isdigit(s: str) -> bool:
    """= str.isdigit(): chuỗi khác rỗng, mọi ký tự là chữ số.
    PORT [P5]: Python coi cả chữ số mũ '²' là digit, Kotlin Char.isDigit() thì không — khác nhau chỉ với ký tự lạ."""
    return len(s) > 0 and all(c.isdigit() for c in s)


def word_shape(w: str) -> str:
    """'Nguyễn' → 'Xxxxxx', '0912' → 'dddd'. Ký tự khác giữ nguyên."""
    return "".join("X" if c.isupper() else "x" if c.islower() else "d" if c.isdigit() else c for c in w)


def sigmoid(z: float) -> float:
    """PORT [P10]: dạng ổn định — không tính exp() của số dương lớn (tránh tràn số)."""
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def java_to_python_regex(p: str) -> str:
    """Config lưu cú pháp Java: nhóm có tên (?<name>...) → Python (?P<name>...). Không đụng (?<= / (?<!."""
    return re.sub(r"\(\?<(?=[A-Za-z])", "(?P<", p)


# =====================================================================
# Mô hình
# =====================================================================

class PureModel:
    def __init__(self, model_path="models/crf_model.json", config_path="models/config.json"):
        m = json.loads(Path(model_path).read_text(encoding="utf-8"))
        c = json.loads(Path(config_path).read_text(encoding="utf-8"))
        self.labels: list[str] = m["labels"]
        self.state: dict[str, dict[str, float]] = m["state_features"]
        # Ma trận chuyển trans[i][j] = trọng số (nhãn i → nhãn j); cặp không có trong file = 0
        L = len(self.labels)
        self.trans = [[m["transition_features"].get(a, {}).get(b, 0.0) for b in self.labels] for a in self.labels]
        self.o_index = self.labels.index("O")

        rc = c["regex"]
        self.rx = {name: re.compile(java_to_python_regex(d["pattern"]),
                                    re.IGNORECASE if "CASE_INSENSITIVE" in d["flags"] else 0)
                   for name, d in rc["patterns"].items()}
        self.TIGHT, self.LOOSE, self.WINDOW = rc["tight"], rc["loose"], rc["window"]
        self.rx_keywords: dict[str, list[str]] = rc["keywords"]
        self.dob_negative: list[str] = rc["dob_negative"]
        self.province_codes = set(rc["province_codes"])
        self.mobile_prefixes = set(rc["mobile_prefixes"])
        self.hotline_prefixes = tuple(rc["hotline_prefixes"])
        self.priority: dict[str, int] = rc["priority"]
        self.rule_order: list[str] = rc["rule_order"]
        self.lookalikes = str.maketrans(rc["digit_lookalikes"])
        self.dashes = str.maketrans(rc["dashes"])

        fc = c["crf_features"]
        self.crf_keywords: dict[str, list[str]] = fc["keywords"]   # thứ tự khóa quan trọng cho "kw"
        self.kw_window = fc["kw_window"]
        self.punct = re.compile(fc["patterns"]["punct_to_space"])
        self.spaces = re.compile(fc["patterns"]["multi_space"])
        self.repeat = re.compile(fc["patterns"]["repeated_char"])
        self.len_max, self.shape_full_max = fc["len_max"], fc["shape_full_max"]
        self.offsets: list[int] = fc["neighbor_offsets"]

        r = c["risk"]
        self.lambdas, self.beta, self.tau = r["lambdas"], r["beta"], r["tau"]
        self.ctx_window = r["ctx_window"]
        self.risk_keywords: list[str] = r["keywords"]
        mc = c["mask"]
        self.merge_v_overlap, self.merge_h_gap = mc["merge_min_v_overlap"], mc["merge_max_h_gap"]

    # =================================================================
    # 0–1. Thứ tự đọc và ghép văn bản
    # =================================================================
    @staticmethod
    def sort_reading_order(words: list[dict]) -> list[dict]:
        """
        Gom từ theo hàng ngang (tâm dọc nằm trong dải [y0, y1] của hàng đang xét), trong hàng xếp trái → phải.
        PORT: dùng ĐÚNG khóa sắp xếp đầy đủ dưới đây (tâm dọc là số thực, chia 2 kiểu Double) — chỉ sort theo
        tâm dọc thì 2 từ cùng tâm sẽ xếp theo thứ tự Tesseract trả về, và cách chia hàng có thể khác.
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
            for w in sorted(row["words"], key=lambda w: (w["bbox"][0], w["bbox"][1], w["bbox"][2], w["bbox"][3],
                                                         w["text"])):
                out.append({**w, "line": i})
        return out

    @staticmethod
    def words_to_text(words: list[dict]) -> tuple[str, list[tuple[int, int]]]:
        """Cùng dòng nối bằng ' ', khác dòng nối bằng '\\n'. PORT [P4]: vị trí tính theo ký tự."""
        parts, spans, pos, prev_line = [], [], 0, None
        for w in words:
            sep = "" if prev_line is None else (" " if w["line"] == prev_line else "\n")
            pos += len(sep)
            spans.append((pos, pos + len(w["text"])))
            parts.append(sep + w["text"])
            pos += len(w["text"])
            prev_line = w["line"]
        return "".join(parts), spans

    # =================================================================
    # 2. Regex (port 1–1 từ regex_rules.py)
    # =================================================================
    def normalize_ocr(self, text: str) -> str:
        """Gạch dài → '-'; trong cụm số: O/o → 0, l/I/| → 1. Độ dài chuỗi KHÔNG đổi."""
        text = text.translate(self.dashes)

        def fix(m):
            s = m.group()
            return s.translate(self.lookalikes) if any(c.isdigit() for c in s) else s

        return self.rx["number_run"].sub(fix, text)

    @staticmethod
    def _last_pos(window: str, keywords: list[str]) -> int:
        """Vị trí xuất hiện cuối cùng của bất kỳ từ khóa nào (Kotlin: lastIndexOf), -1 nếu không có."""
        return max((window.rfind(k) for k in keywords), default=-1)

    def _window(self, t: str, start: int) -> str:
        return fold(t[max(0, start - self.WINDOW):start])

    def has_keyword(self, t: str, start: int, label: str) -> bool:
        return self._last_pos(self._window(t, start), self.rx_keywords[label]) >= 0

    def _match(self, start, end, label, score, ctx):
        return {"start": start, "end": end, "label": label, "score": score, "ctx": bool(ctx)}

    def match_cccd(self, t):
        out = []
        for m in self.rx["cccd"].finditer(t):
            digits = self.rx["non_digit"].sub("", m.group())
            structured = digits[:3] in self.province_codes and digits[3] in "0123"
            ctx = self.has_keyword(t, m.start(), "CCCD")
            if not structured and not ctx:
                continue
            score = self.TIGHT if structured and m.group() == digits else self.LOOSE
            out.append(self._match(m.start(), m.end(), "CCCD", score, ctx))
        return out

    def match_dob(self, t, today: date):
        """PORT [P9]: `today` truyền vào."""
        out = []
        for m in self.rx["date"].finditer(t):
            d, mo, y = int(m.group(1)), int(m.group(3)), int(m.group(4))
            try:
                dob = date(y, mo, d)  # Kotlin: LocalDate.of trong try/catch DateTimeException
            except ValueError:
                continue
            if not (1900 <= y and dob <= today):
                continue
            window = self._window(t, m.start())
            pos, neg = self._last_pos(window, self.rx_keywords["DOB"]), self._last_pos(window, self.dob_negative)
            if neg > pos:
                continue
            ctx = pos >= 0
            score = self.TIGHT if ctx and " " not in m.group() else self.LOOSE
            out.append(self._match(m.start(), m.end(), "DOB", score, ctx))
        return out

    def match_phone(self, t):
        out = []
        for m in self.rx["phone"].finditer(t):
            digits = self.rx["non_digit"].sub("", m.group())
            local = "0" + digits[2:] if digits.startswith("84") else digits
            tight = (self.rx["phone_canonical"].fullmatch(m.group().replace("\n", " ")) is not None
                     and local[:3] in self.mobile_prefixes)  # Kotlin: Regex.matches() = fullmatch
            out.append(self._match(m.start(), m.end(), "PHONE", self.TIGHT if tight else self.LOOSE,
                                   self.has_keyword(t, m.start(), "PHONE")))
        return out

    def match_email(self, t):
        out = [self._match(m.start(), m.end(), "EMAIL", self.TIGHT, self.has_keyword(t, m.start(), "EMAIL"))
               for m in self.rx["email_tight"].finditer(t)]
        for m in self.rx["email_loose"].finditer(t):
            if not any(m.start() < o["end"] and o["start"] < m.end() for o in out):
                out.append(self._match(m.start(), m.end(), "EMAIL", self.LOOSE,
                                       self.has_keyword(t, m.start(), "EMAIL")))
        return out

    def match_money(self, t):
        out = []
        tg = self.rx["thousand_group"]
        for m in self.rx["money"].finditer(t):
            num = m.group("num")
            if m.group("cur"):  # Kotlin: m.groups["cur"] != null
                score = self.TIGHT
            elif tg.search(num) and (num[0] in "+-" or len(tg.findall(num)) >= 2):
                score = self.LOOSE
            else:
                continue
            out.append(self._match(m.start(), m.end(), "MONEY", score, self.has_keyword(t, m.start(), "MONEY")))
        for m in self.rx["money_prefix"].finditer(t):
            out.append(self._match(m.start(), m.end(), "MONEY", self.TIGHT, self.has_keyword(t, m.start(), "MONEY")))
        for m in self.rx["money_short"].finditer(t):
            score = self.LOOSE if m.group("unit").lower() in ("tr", "k") else self.TIGHT
            out.append(self._match(m.start(), m.end(), "MONEY", score, self.has_keyword(t, m.start(), "MONEY")))
        return out

    def match_account(self, t):
        out = []
        for m in self.rx["account"].finditer(t):
            groups = list(self.rx["digit_run"].finditer(m.group()))
            k, digits = len(groups), ""
            while k > 0:  # số nhóm lớn nhất: các nhóm trước nhóm cuối đều 4 số, tổng 8–16 số
                digits = "".join(g.group() for g in groups[:k])
                if all(len(g.group()) == 4 for g in groups[:k - 1]) and 8 <= len(digits) <= 16:
                    break
                k -= 1
            if k == 0 or digits.startswith(self.hotline_prefixes):
                continue
            ctx = self.has_keyword(t, m.start(), "ACCOUNT")
            out.append(self._match(m.start(), m.start() + groups[k - 1].end(), "ACCOUNT",
                                   self.TIGHT if ctx else self.LOOSE, ctx))
        return out

    def resolve(self, matches):
        """PORT [P8]: sort ỔN ĐỊNH theo (−score, −ctx, ưu tiên nhãn, −độ dài); giữ cái không chồng lấn."""
        ranked = sorted(matches, key=lambda m: (-m["score"], -int(m["ctx"]), self.priority[m["label"]],
                                                -(m["end"] - m["start"])))
        kept = []
        for m in ranked:
            if all(m["end"] <= k["start"] or k["end"] <= m["start"] for k in kept):
                kept.append(m)
        return sorted(kept, key=lambda m: m["start"])

    def find_entities(self, text: str, today: date):
        norm = self.normalize_ocr(text)
        rules = {"CCCD": self.match_cccd, "DOB": lambda t: self.match_dob(t, today), "PHONE": self.match_phone,
                 "EMAIL": self.match_email, "MONEY": self.match_money, "ACCOUNT": self.match_account}
        cands = [m for name in self.rule_order for m in rules[name](norm)]
        for m in cands:  # phải sửa lỗi OCR mới khớp được → chỉ tính là khớp lỏng
            if norm[m["start"]:m["end"]] != text[m["start"]:m["end"]]:
                m["score"] = self.LOOSE
        return self.resolve(cands)

    def tag_words(self, words, today: date):
        """Mỗi từ → match regex phủ lên nó (match sau ghi đè match trước), hoặc None."""
        text, spans = self.words_to_text(words)
        tags = [None] * len(words)
        for m in self.find_entities(text, today):
            for i, (s, e) in enumerate(spans):
                if s < m["end"] and m["start"] < e:
                    tags[i] = m
        return tags

    # =================================================================
    # 3. Đặc trưng CRF (port 1–1 từ crf.py)
    # =================================================================
    def annotate(self, words, today: date) -> list[dict]:
        """Thông tin cần ngữ cảnh rộng: kết quả regex và loại từ khóa trong kw_window từ phía trước."""
        ann = []
        folded = [fold(w["text"]) for w in words]
        for i, m in enumerate(self.tag_words(words, today)):
            before = " ".join(folded[max(0, i - self.kw_window):i])
            before = self.spaces.sub(" ", self.punct.sub(" ", before))
            ann.append({
                "rx": m["label"] if m else "O",
                "rx_score": m["score"] if m else 0.0,
                "rx_ctx": bool(m and m["ctx"]),
                "kw": [ent for ent, kws in self.crf_keywords.items() if any(k in before for k in kws)],
            })
        return ann

    def short_shape(self, w: str) -> str:
        return self.repeat.sub(r"\1", word_shape(w))  # Kotlin: replace(Regex("(.)\\1+"), "$1")

    def _token_feats(self, w, a, p):
        t = w["text"]
        return {p + "lower": t.lower(), p + "fold": fold(t), p + "istitle": py_istitle(t),
                p + "isupper": py_isupper(t), p + "isdigit": py_isdigit(t), p + "shape": self.short_shape(t),
                p + "rx": a["rx"]}

    def word2features(self, words, ann, i) -> dict:
        w, a, t = words[i], ann[i], words[i]["text"]
        low = t.lower()
        f = {"bias": 1.0, **self._token_feats(w, a, ""),
             "len": min(len(t), self.len_max),                # PORT [P4]: số ký tự
             "prefix2": low[:2], "prefix3": low[:3], "suffix2": low[-2:], "suffix3": low[-3:],
             "shape_full": word_shape(t)[:self.shape_full_max],
             "has_diacritic": bo_dau(t) != t, "has_digit": any(c.isdigit() for c in t), "has_at": "@" in t,
             "bol": i == 0 or words[i - 1]["line"] != w["line"],
             "eol": i == len(words) - 1 or words[i + 1]["line"] != w["line"],
             "rx_tight": a["rx_score"] == self.TIGHT, "rx_loose": a["rx_score"] == self.LOOSE,
             "rx_ctx": a["rx_ctx"]}
        for ent in a["kw"]:
            f["kw:" + ent] = True
        for off in self.offsets:
            j, p = i + off, f"{off:+d}:"   # tiền tố "-2:", "-1:", "+1:", "+2:"
            if 0 <= j < len(words):
                f.update(self._token_feats(words[j], ann[j], p))
                f[p + "same_line"] = words[j]["line"] == w["line"]
            else:
                f[p + ("BOS" if off < 0 else "EOS")] = True
        return f

    @staticmethod
    def to_attributes(features: dict) -> dict[str, float]:
        """
        Đổi dict đặc trưng sang thuộc tính crfsuite. PORT [P6]:
          chuỗi       → tên "khóa:giá_trị", giá trị 1.0
          bool / số   → tên "khóa", giá trị float (True=1.0, False=0.0 — góp 0 điểm, có thể bỏ qua)
        """
        attrs = {}
        for k, v in features.items():
            if isinstance(v, str):
                attrs[f"{k}:{v}"] = 1.0
            else:
                attrs[k] = float(v)
        return attrs

    # =================================================================
    # 4. Viterbi + forward–backward
    # =================================================================
    def state_scores(self, attrs: dict[str, float]) -> list[float]:
        """Điểm từng nhãn tại 1 vị trí = Σ giá_trị_thuộc_tính × w(thuộc_tính, nhãn). Thuộc tính lạ → bỏ qua."""
        idx = {l: k for k, l in enumerate(self.labels)}
        s = [0.0] * len(self.labels)
        for a, v in attrs.items():
            if v == 0.0:
                continue
            for lab, w in self.state.get(a, {}).items():
                s[idx[lab]] += v * w
        return s

    def viterbi(self, S: list[list[float]]) -> list[int]:
        """CRF bậc 1 của crfsuite: KHÔNG có trọng số bắt đầu/kết thúc chuỗi."""
        n, L, T = len(S), len(self.labels), self.trans
        if n == 0:
            return []
        delta, back = [S[0][:]], []
        for t in range(1, n):
            row, bp = [], []
            for j in range(L):
                best_i = max(range(L), key=lambda i: delta[-1][i] + T[i][j])  # hoà → chỉ số nhỏ nhất
                bp.append(best_i)
                row.append(delta[-1][best_i] + T[best_i][j] + S[t][j])
            delta.append(row)
            back.append(bp)
        path = [max(range(L), key=lambda j: delta[-1][j])]
        for bp in reversed(back):
            path.append(bp[path[-1]])
        return path[::-1]

    @staticmethod
    def _logsumexp(xs: list[float]) -> float:
        m = max(xs)
        return m + math.log(sum(math.exp(x - m) for x in xs))

    def marginals(self, S: list[list[float]]) -> list[list[float]]:
        """P(nhãn_t = j | cả chuỗi) = exp(α_t(j) + β_t(j) − log Z). PORT [P10]: làm trong không gian log."""
        n, L, T = len(S), len(self.labels), self.trans
        if n == 0:
            return []
        alpha = [S[0][:]]
        for t in range(1, n):
            alpha.append([S[t][j] + self._logsumexp([alpha[-1][i] + T[i][j] for i in range(L)]) for j in range(L)])
        beta = [[0.0] * L for _ in range(n)]
        for t in range(n - 2, -1, -1):
            beta[t] = [self._logsumexp([T[i][j] + S[t + 1][j] + beta[t + 1][j] for j in range(L)]) for i in range(L)]
        log_z = self._logsumexp(alpha[-1])
        return [[math.exp(alpha[t][j] + beta[t][j] - log_z) for j in range(L)] for t in range(n)]

    # =================================================================
    # 5. Hàm rủi ro, quyết định che, gộp vùng
    # =================================================================
    def risk_inputs(self, words, ann, p_sensitive) -> list[list[float]]:
        folded = [self.punct.sub(" ", fold(w["text"])) for w in words]
        X = []
        for i, w in enumerate(words):
            window = self.spaces.sub(" ", " ".join(folded[max(0, i - self.ctx_window):i + self.ctx_window + 1]))
            ctx = any(k in window for k in self.risk_keywords)
            X.append([p_sensitive[i], ann[i]["rx_score"], min(max(w["conf"], 0.0), 1.0), 1.0 if ctx else 0.0])
        return X

    def risk_score(self, x: list[float]) -> float:
        return sigmoid(sum(l * v for l, v in zip(self.lambdas, x)) + self.beta)

    def _same_line_adjacent(self, a, b) -> bool:
        ha, hb = a[3] - a[1], b[3] - b[1]
        if min(a[3], b[3]) - max(a[1], b[1]) < self.merge_v_overlap * min(ha, hb):
            return False
        return max(a[0], b[0]) - min(a[2], b[2]) <= self.merge_h_gap * max(ha, hb)

    def merge_regions(self, items: list[dict]) -> list[dict]:
        regions = [{"bbox": list(it["bbox"]), "labels": [it["label"]], "F": it["F"], "text": it["text"]}
                   for it in items]
        merged = True
        while merged:
            merged = False
            for i in range(len(regions)):
                for j in range(i + 1, len(regions)):
                    a, b = regions[i], regions[j]
                    if self._same_line_adjacent(a["bbox"], b["bbox"]):
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

    # =================================================================
    # Toàn bộ pipeline cho 1 ảnh
    # =================================================================
    def process(self, words: list[dict], today: date | None = None, already_sorted: bool = True) -> dict:
        """
        words: [{text, bbox, conf[, line]}] từ OCR. already_sorted=False → tự sort_reading_order.
        Trả về mọi kết quả trung gian (dùng cho golden test).
        """
        today = today or date.today()
        words = [{**w, "text": nfc(w["text"])} for w in words]       # PORT [P1]
        if not already_sorted:
            words = self.sort_reading_order(words)
        ann = self.annotate(words, today)
        feats = [self.word2features(words, ann, i) for i in range(len(words))]
        attrs = [self.to_attributes(f) for f in feats]
        S = [self.state_scores(a) for a in attrs]
        path = self.viterbi(S)
        marg = self.marginals(S)
        p_sens = [1.0 - row[self.o_index] for row in marg]
        X = self.risk_inputs(words, ann, p_sens)
        F = [self.risk_score(x) for x in X]
        tokens = []
        for i, w in enumerate(words):
            lab = self.labels[path[i]]
            tokens.append({**w, "rx": ann[i]["rx"], "rx_score": ann[i]["rx_score"], "rx_ctx": ann[i]["rx_ctx"],
                           "kw": ann[i]["kw"], "features": feats[i], "attributes": attrs[i],
                           "label": lab, "marginals": dict(zip(self.labels, marg[i])), "p_sensitive": p_sens[i],
                           "risk_x": X[i], "F": F[i], "mask": F[i] >= self.tau})
        flagged = [{"bbox": t["bbox"], "label": self._display_label(t), "F": t["F"], "text": t["text"]}
                   for t in tokens if t["mask"]]
        return {"tokens": tokens, "regions": self.merge_regions(flagged)}

    def _display_label(self, t) -> str:
        """Nhãn hiển thị cho vùng che: nhãn CRF → nhãn regex → nhãn có xác suất biên cao nhất (khác O)."""
        if t["label"] != "O":
            return t["label"][2:]
        if t["rx"] != "O":
            return t["rx"]
        best = max((l for l in self.labels if l != "O"), key=lambda l: t["marginals"][l])
        return best[2:]
