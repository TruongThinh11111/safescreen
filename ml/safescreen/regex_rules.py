"""
safescreen/regex_rules.py — Bước 3: bắt các thực thể CÓ CẤU TRÚC bằng regex.

Thực thể: CCCD, DOB, PHONE, EMAIL, MONEY, ACCOUNT (NAME/ADDRESS để CRF lo).

Mỗi kết quả là RegexMatch(start, end, label, score, ctx):
    start, end : vị trí ký tự trong văn bản đầu vào (text[start:end] là chuỗi khớp)
    score      : 1.0 = khớp CHẶT (đúng định dạng chuẩn), 0.6 = khớp LỎNG (sai định dạng nhẹ,
                 phải sửa lỗi OCR mới khớp, hoặc thiếu từ khóa ngữ cảnh) → r_regex trong hàm rủi ro
    ctx        : có từ khóa ngữ cảnh ngay phía trước không (vd "STK", "Ngày sinh") → ctx trong hàm rủi ro

Văn bản đầu vào nên ở dạng Unicode NFC (Tesseract và Python mặc định đã là NFC).

Đánh giá precision/recall trên tập test (đứng ở thư mục ml/):
    python -m safescreen.regex_rules --data data/test
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import date
from pathlib import Path
from typing import NamedTuple

from safescreen.text_utils import fold

REGEX_LABELS = ["CCCD", "DOB", "PHONE", "EMAIL", "MONEY", "ACCOUNT"]
TIGHT, LOOSE = 1.0, 0.6


class RegexMatch(NamedTuple):
    start: int
    end: int
    label: str
    score: float
    ctx: bool = False


# =====================================================================
# 1. Chuẩn hoá lỗi OCR (giữ nguyên độ dài chuỗi → vị trí start/end vẫn đúng với văn bản gốc)
# =====================================================================

DIGIT_LOOKALIKES = str.maketrans({"O": "0", "o": "0", "l": "1", "I": "1", "|": "1"})
DASHES = str.maketrans({"–": "-", "—": "-", "−": "-"})

# Một "cụm số": chuỗi chữ số + ký tự dễ nhầm với số, KHÔNG dính liền chữ cái khác ở hai đầu.
# (?<![^\W\d]) nghĩa là "ký tự trước không phải chữ cái" (\w trừ chữ số).
_NUMBER_RUN = re.compile(r"(?<![^\W\d])[0-9OolI|]+(?![^\W\d])")


def normalize_ocr(text: str) -> str:
    """
    Sửa lỗi OCR phổ biến, thay từng ký tự một (độ dài chuỗi không đổi):
      - Gạch ngang dài (–, —) → '-'
      - Trong một cụm số: O/o → 0, l/I/| → 1     ("O79203O12345" → "079203012345")
    Chỉ sửa trong cụm có ít nhất 1 chữ số thật và không dính liền chữ cái, để không phá
    các từ thường như "Idviet1570@yahoo.com" hay "No.:".
    """
    text = text.translate(DASHES)

    def fix(m: re.Match) -> str:
        s = m.group()
        return s.translate(DIGIT_LOOKALIKES) if any(c.isdigit() for c in s) else s

    return _NUMBER_RUN.sub(fix, text)


# =====================================================================
# 2. Từ khóa ngữ cảnh (đã bỏ dấu, chữ thường — so khớp trên fold(text))
# =====================================================================

KEYWORDS = {
    "CCCD": ["cccd", "cmnd", "can cuoc", "dinh danh", "no.", "citizen", "identity"],
    "DOB": ["ngay sinh", "sinh ngay", "nam sinh", "sinh nhat", "date of birth", "dob", "birth"],
    "PHONE": ["sdt", "dien thoai", "phone", "goi", "zalo", "lien he", "so may"],
    "EMAIL": ["email", "mail", "gmail"],
    "MONEY": ["so tien", "so du", "phi", "luong", "gia", "chuyen", "nhan", "thanh toan", "tong"],
    "ACCOUNT": ["stk", "tk", "tai khoan", "account", "chuyen tien den", "chuyen den", "chuyen khoan vao"],
}
# Ngày đi sau các từ này thì KHÔNG phải ngày sinh
DOB_NEGATIVE = ["gia tri den", "het han", "ngay cap", "thoi gian", "giao dich", "expiry", "valid"]

WINDOW = 40  # chỉ nhìn 40 ký tự ngay trước chỗ khớp


def _last_pos(window: str, keywords: list[str]) -> int:
    """Vị trí xuất hiện cuối cùng (gần chỗ khớp nhất) của bất kỳ từ khóa nào; -1 nếu không có."""
    return max((window.rfind(k) for k in keywords), default=-1)


def has_keyword(text: str, start: int, label: str) -> bool:
    return _last_pos(fold(text[max(0, start - WINDOW):start]), KEYWORDS[label]) >= 0


# =====================================================================
# 3. Các luật (mỗi hàm nhận văn bản ĐÃ chuẩn hoá, trả về list[RegexMatch])
# =====================================================================

# 63 mã tỉnh/thành trong 3 số đầu của CCCD
PROVINCE_CODES = set(
    "001 002 004 006 008 010 011 012 014 015 017 019 020 022 024 025 026 027 030 031 033 034 035 036 "
    "037 038 040 042 044 045 046 048 049 051 052 054 056 058 060 062 064 066 067 068 070 072 074 075 "
    "077 079 080 082 083 084 086 087 089 091 092 093 094 095 096".split())
MOBILE_PREFIXES = set(
    "032 033 034 035 036 037 038 039 052 055 056 058 059 070 076 077 078 079 "
    "081 082 083 084 085 086 087 088 089 090 091 092 093 094 096 097 098 099".split())

# Mẫu dùng chung: không dính chữ số / dấu + phía trước, và không phải phần đuôi của dãy "1234 5678 ..."
_NO_DIGIT_BEFORE = r"(?<![\d+])(?<!\d[ .])"
_NO_DIGIT_AFTER = r"(?![ .-]?\d)"


_CCCD = re.compile(_NO_DIGIT_BEFORE + r"\d(?:[ .\n]?\d){11}" + _NO_DIGIT_AFTER)
_NON_DIGIT = re.compile(r"\D")


def match_cccd(t: str) -> list[RegexMatch]:
    """
    12 chữ số; giữa các số có thể có 1 dấu cách/chấm do OCR tách ("079 203 012 345").
    Chặt: 12 số liền + đúng cấu trúc (3 số đầu là mã tỉnh, số thứ 4 là giới tính/thế kỷ 0–3).
    Lỏng: có dấu cách xen giữa, hoặc sai cấu trúc nhưng có từ khóa "CCCD" phía trước.
    Sai cấu trúc + không từ khóa → bỏ qua (nhiều khả năng là số tài khoản).
    """
    out = []
    for m in _CCCD.finditer(t):
        digits = _NON_DIGIT.sub("", m.group())
        structured = digits[:3] in PROVINCE_CODES and digits[3] in "0123"
        ctx = has_keyword(t, m.start(), "CCCD")
        if not structured and not ctx:
            continue
        score = TIGHT if structured and m.group() == digits else LOOSE
        out.append(RegexMatch(m.start(), m.end(), "CCCD", score, ctx))
    return out


_DATE = re.compile(r"(?<![\d/.-])(\d{1,2})\s?([/.-])\s?(\d{1,2})\s?\2\s?(\d{4})(?![\d/.-]?\d)")


def match_dob(t: str, today: date | None = None) -> list[RegexMatch]:
    """
    dd/mm/yyyy, dd-mm-yyyy, dd.mm.yyyy (ngày/tháng có thể 1 chữ số).
    - Ngày không hợp lệ (32/01, 05/13) hoặc ở tương lai → bỏ (không thể là ngày sinh).
    - Ngay trước có từ khóa phủ định ("Có giá trị đến", "Thời gian") gần hơn từ khóa ngày sinh → bỏ.
    - Có "Ngày sinh"/"DOB"... → chặt. Không có từ khóa → lỏng (có thể chỉ là ngày giao dịch).
    today: mốc "hôm nay" để loại ngày tương lai (mặc định ngày chạy; truyền cố định để test tái lập được).
    """
    today = today or date.today()
    out = []
    for m in _DATE.finditer(t):
        d, mo, y = int(m.group(1)), int(m.group(3)), int(m.group(4))
        try:
            dob = date(y, mo, d)
        except ValueError:
            continue
        if not (1900 <= y and dob <= today):
            continue
        window = fold(t[max(0, m.start() - WINDOW):m.start()])
        pos, neg = _last_pos(window, KEYWORDS["DOB"]), _last_pos(window, DOB_NEGATIVE)
        if neg > pos:
            continue
        ctx = pos >= 0
        score = TIGHT if ctx and " " not in m.group() else LOOSE
        out.append(RegexMatch(m.start(), m.end(), "DOB", score, ctx))
    return out


# Dấu phân cách có thể là xuống dòng: số bị ngắt dòng trong bong bóng chat ("0902⏎821 625")
_PHONE = re.compile(_NO_DIGIT_BEFORE + r"(?:\(\+84\)|\+\s?84|84|0)[\s.-]?[35789](?:[\s.-]?\d){8}" + _NO_DIGIT_AFTER)
_PHONE_CANONICAL = re.compile(r"0\d{9}|0\d{3}([ .-])\d{3}\1\d{3}|\+84 ?\d{9}|(?:\+84|\(\+84\)) \d{3} \d{3} \d{3}")


def match_phone(t: str) -> list[RegexMatch]:
    """
    SĐT di động VN: 0[35789]xxxxxxxx hoặc +84 / (+84) / 84 thay cho số 0 đầu.
    Chặt: viết đúng kiểu phổ biến (0912345678, 0912 345 678, 0912.345.678, +84 912 345 678, (+84) 912 345 678)
          và đầu số có thật (091, 038, ...). Còn lại (vd "84912345678", "091 23 45 678") → lỏng.
    """
    out = []
    for m in _PHONE.finditer(t):
        digits = _NON_DIGIT.sub("", m.group())
        local = "0" + digits[2:] if digits.startswith("84") else digits
        tight = _PHONE_CANONICAL.fullmatch(m.group().replace("\n", " ")) and local[:3] in MOBILE_PREFIXES
        out.append(RegexMatch(m.start(), m.end(), "PHONE", TIGHT if tight else LOOSE,
                              has_keyword(t, m.start(), "PHONE")))
    return out


_EMAIL_TIGHT = re.compile(r"(?<![\w.+-])[A-Za-z0-9][A-Za-z0-9._%+-]*@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}(?!\w)")
# Lỏng: chấp nhận lỗi OCR hay gặp — "abc(@gmail.com", "abc @ gmail . com", chữ có dấu trong tên
_EMAIL_LOOSE = re.compile(r"(?<![\w.+-])\w[\w.%+-]*\s?[(\[{]?@+\s?[\w-]+(?:\s?\.\s?[\w-]+)*\s?\.\s?[A-Za-z]{2,}(?!\w)")


def match_email(t: str) -> list[RegexMatch]:
    out = [RegexMatch(m.start(), m.end(), "EMAIL", TIGHT, has_keyword(t, m.start(), "EMAIL"))
           for m in _EMAIL_TIGHT.finditer(t)]
    for m in _EMAIL_LOOSE.finditer(t):
        if not any(m.start() < o.end and o.start < m.end() for o in out):  # chỗ nào đã khớp chặt thì thôi
            out.append(RegexMatch(m.start(), m.end(), "EMAIL", LOOSE, has_keyword(t, m.start(), "EMAIL")))
    return out


# Đơn vị tiền: "VND", "VNĐ", "đồng" (OCR hay đọc thành "đông"), "đ"/"₫" (có thể cách 1 dấu cách),
# hoặc "d" dính liền số (OCR đọc đ → d; "d" đứng riêng thì không tính vì quá dễ nhầm)
_CURRENCY = r"(?:\s?(?i:vn[dđ]|đồng|đông|dong)|\s?[đ₫]|d)(?!\w)"
_MONEY = re.compile(r"(?<![\w.,+-])(?P<num>(?:[+-]\s?)?(?:\d{1,3}(?:[.,]\d{3})+|\d+))(?P<cur>" + _CURRENCY + ")?")
# Đơn vị đứng TRƯỚC số: "VND 1.500.000", "VNĐ -200,000"
_MONEY_PREFIX = re.compile(r"(?<!\w)(?i:vn[dđ])\s?(?:[+-]\s?)?\d{1,3}(?:[.,]\d{3})+(?![\d.,]?\d)")
# Cách viết tắt trong tin nhắn: 500k, 2tr, 1tr5, 15 triệu, 300 nghìn
_MONEY_SHORT = re.compile(r"(?<![\w.,])\d+(?:[.,]\d+)?\s?(?P<unit>triệu|trieu|nghìn|ngàn|nghin|ngan|tr|k)\d?(?!\w)",
                          re.IGNORECASE)
_THOUSAND_GROUP = re.compile(r"[.,]\d{3}")  # 1 nhóm hàng nghìn ".000"


def match_money(t: str) -> list[RegexMatch]:
    """
    Chặt: số + đơn vị tiền ("1.500.000đ", "2,000,000 VND", "945.000 đồng", "15 triệu").
    Lỏng: số có phân cách hàng nghìn nhưng thiếu đơn vị, chỉ khi có dấu +/- ("+2,000,000")
          hoặc từ hàng triệu trở lên ("1.500.000"); viết tắt "500k", "2tr".
    """
    out = []
    for m in _MONEY.finditer(t):
        num = m.group("num")
        if m.group("cur"):
            score = TIGHT
        elif _THOUSAND_GROUP.search(num) and (num[0] in "+-" or len(_THOUSAND_GROUP.findall(num)) >= 2):
            score = LOOSE
        else:
            continue
        out.append(RegexMatch(m.start(), m.end(), "MONEY", score, has_keyword(t, m.start(), "MONEY")))
    for m in _MONEY_PREFIX.finditer(t):
        out.append(RegexMatch(m.start(), m.end(), "MONEY", TIGHT, has_keyword(t, m.start(), "MONEY")))
    for m in _MONEY_SHORT.finditer(t):
        score = LOOSE if m.group("unit").lower() in ("tr", "k") else TIGHT
        out.append(RegexMatch(m.start(), m.end(), "MONEY", score, has_keyword(t, m.start(), "MONEY")))
    return out


# Ứng viên: 1 dãy số + tối đa 3 nhóm số phía sau (cách nhau bởi dấu cách hoặc xuống dòng do OCR)
# Phía sau: không dính chữ/số; dấu "." "," cuối câu thì được, miễn sau đó không phải chữ số (1.500.000)
_ACCOUNT = re.compile(r"(?<![\w#+.,])(?<!\d[ .])\d{4,16}(?:\s\d{1,4}){0,3}(?!\w|[.,]\d|[ ]\d)")
_DIGIT_RUN = re.compile(r"\d+")
HOTLINE_PREFIXES = ("1800", "1900")


def match_account(t: str) -> list[RegexMatch]:
    """
    8–16 chữ số liền, hoặc chia nhóm 4 ("1025 4551 61").
    Chặt: có từ khóa "STK"/"tài khoản"/"chuyển đến"... ngay trước.
    Lỏng: không có từ khóa (dễ trùng CCCD, SĐT → để bước gộp xử lý).
    Bỏ qua tổng đài 1800/1900.
    """
    out = []
    for m in _ACCOUNT.finditer(t):
        groups = list(_DIGIT_RUN.finditer(m.group()))
        # Ứng viên có thể nuốt nhầm số ở dòng sau ("112804822⏎20.05.2026" → nhóm "20").
        # Lấy số nhóm k lớn nhất thoả: các nhóm trước nhóm cuối đều đúng 4 số, tổng 8–16 số.
        for k in range(len(groups), 0, -1):
            digits = "".join(g.group() for g in groups[:k])
            if all(len(g.group()) == 4 for g in groups[:k - 1]) and 8 <= len(digits) <= 16:
                break
        else:
            continue
        if digits.startswith(HOTLINE_PREFIXES):
            continue
        ctx = has_keyword(t, m.start(), "ACCOUNT")
        out.append(RegexMatch(m.start(), m.start() + groups[k - 1].end(), "ACCOUNT",
                              TIGHT if ctx else LOOSE, ctx))
    return out


RULES = [match_cccd, match_dob, match_phone, match_email, match_money, match_account]

# Toàn bộ regex của module (để xuất sang Kotlin trong export_android.py)
PATTERNS = {
    "number_run": _NUMBER_RUN, "cccd": _CCCD, "non_digit": _NON_DIGIT, "date": _DATE,
    "phone": _PHONE, "phone_canonical": _PHONE_CANONICAL,
    "email_tight": _EMAIL_TIGHT, "email_loose": _EMAIL_LOOSE,
    "money": _MONEY, "money_prefix": _MONEY_PREFIX, "money_short": _MONEY_SHORT,
    "thousand_group": _THOUSAND_GROUP, "account": _ACCOUNT, "digit_run": _DIGIT_RUN,
}


# =====================================================================
# 4. Gộp kết quả các luật
# =====================================================================

# Khi 2 luật khớp cùng 1 chỗ với điểm và ngữ cảnh ngang nhau: ưu tiên theo thứ tự này
# (vd 0912345678 vừa là SĐT vừa là "số tài khoản lỏng" → chọn PHONE)
PRIORITY = {"EMAIL": 0, "CCCD": 1, "PHONE": 2, "DOB": 3, "MONEY": 4, "ACCOUNT": 5}


def resolve(matches: list[RegexMatch]) -> list[RegexMatch]:
    """Bỏ các kết quả chồng lấn: giữ lần lượt cái điểm cao hơn → có từ khóa → ưu tiên nhãn → dài hơn."""
    ranked = sorted(matches, key=lambda m: (-m.score, -m.ctx, PRIORITY[m.label], -(m.end - m.start)))
    kept = []
    for m in ranked:
        if all(m.end <= k.start or k.end <= m.start for k in kept):
            kept.append(m)
    return sorted(kept, key=lambda m: m.start)


def find_entities(text: str, min_score: float = 0.0, today: date | None = None) -> list[RegexMatch]:
    """Hàm chính: chuẩn hoá → chạy mọi luật → gộp. Vị trí trả về khớp với `text` gốc."""
    norm = normalize_ocr(text)
    # Giữ đúng thứ tự RULES: khi resolve() gặp 2 ứng viên ngang nhau hoàn toàn, sort ổn định giữ cái đến trước
    cands = [m for rule in RULES for m in (rule(norm, today) if rule is match_dob else rule(norm))]
    # Phải sửa lỗi OCR mới khớp được → chỉ tính là khớp lỏng
    cands = [m._replace(score=LOOSE) if norm[m.start:m.end] != text[m.start:m.end] else m for m in cands]
    return [m for m in resolve(cands) if m.score >= min_score]


# =====================================================================
# 5. Gắn nhãn cho danh sách từ (đầu ra OCR hoặc token gốc) + đánh giá
# =====================================================================

def words_to_text(words: list[dict]) -> tuple[str, list[tuple[int, int]]]:
    """Ghép từ thành văn bản (cùng dòng: dấu cách, khác dòng: xuống dòng), nhớ vị trí từng từ."""
    parts, spans, pos, prev_line = [], [], 0, None
    for w in words:
        sep = "" if prev_line is None else (" " if w["line"] == prev_line else "\n")
        pos += len(sep)
        spans.append((pos, pos + len(w["text"])))
        parts.append(sep + w["text"])
        pos += len(w["text"])
        prev_line = w["line"]
    return "".join(parts), spans


def tag_words(words: list[dict], min_score: float = 0.0, today: date | None = None) -> list[RegexMatch | None]:
    """Trả về, cho từng từ, RegexMatch phủ lên nó (hoặc None)."""
    text, spans = words_to_text(words)
    tags: list[RegexMatch | None] = [None] * len(words)
    for m in find_entities(text, min_score, today):
        for i, (s, e) in enumerate(spans):
            if s < m.end and m.start < e:
                tags[i] = m
    return tags


def evaluate(docs: list[list[dict]], min_score: float) -> tuple[dict, Counter]:
    """
    Đánh giá ở mức TỪ (token) — đúng đơn vị sẽ bị che trên ảnh.
      per_label[X]: tp/fp/fn của nhãn X (đòi hỏi đúng loại nhãn)
      mask        : góc nhìn "che hay không che" (nhầm CCCD thành ACCOUNT vẫn được che):
                    sens/sens_hit = token nhạy cảm thuộc 6 nhãn regex / số được che
                    o/o_hit       = token không nhạy cảm / số bị che nhầm (over-masking)
    """
    per_label = {lab: Counter() for lab in REGEX_LABELS}
    mask = Counter()
    for words in docs:
        for w, m in zip(words, tag_words(words, min_score)):
            gold = w["label"][2:] if w["label"] != "O" else "O"
            pred = m.label if m else "O"
            for lab in REGEX_LABELS:
                if pred == lab and gold == lab:
                    per_label[lab]["tp"] += 1
                elif pred == lab:
                    per_label[lab]["fp"] += 1
                elif gold == lab:
                    per_label[lab]["fn"] += 1
            if gold in REGEX_LABELS:
                mask["sens"] += 1
                mask["sens_hit"] += pred != "O"
            elif gold == "O":
                mask["o"] += 1
                mask["o_hit"] += pred != "O"
    return per_label, mask


def prf(c: Counter) -> tuple[float, float, float]:
    p = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 0.0
    r = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def print_report(title: str, docs: list[list[dict]]):
    print(f"\n=== {title} ({len(docs)} ảnh) ===")
    for min_score, name in [(LOOSE, "mọi kết quả (score ≥ 0.6)"), (TIGHT, "chỉ khớp chặt (score = 1.0)")]:
        per_label, mask = evaluate(docs, min_score)
        print(f"\n  -- {name}")
        print(f"  {'Nhãn':<9}{'P':>7}{'R':>7}{'F1':>7}{'TP':>6}{'FP':>6}{'FN':>6}")
        total = Counter()
        for lab in REGEX_LABELS:
            p, r, f = prf(per_label[lab])
            total.update(per_label[lab])
            c = per_label[lab]
            print(f"  {lab:<9}{p:>7.1%}{r:>7.1%}{f:>7.1%}{c['tp']:>6}{c['fp']:>6}{c['fn']:>6}")
        p, r, f = prf(total)
        print(f"  {'micro':<9}{p:>7.1%}{r:>7.1%}{f:>7.1%}")
        print(f"  Che được token nhạy cảm (6 nhãn regex): {mask['sens_hit'] / max(mask['sens'], 1):.1%}"
              f"   |   Over-masking (token O bị che): {mask['o_hit'] / max(mask['o'], 1):.2%}"
              f" ({mask['o_hit']}/{mask['o']})")


def main():
    ap = argparse.ArgumentParser(description="Đánh giá luật regex trên tập dữ liệu tổng hợp")
    ap.add_argument("--data", default="data/test", help="thư mục có labels/ (và ocr/ nếu đã chạy ocr.py)")
    args = ap.parse_args()
    data = Path(args.data)

    def load(sub, key):
        return [json.loads(f.read_text(encoding="utf-8"))[key] for f in sorted((data / sub).glob("*.json"))]

    print_report("Văn bản gốc (OCR hoàn hảo)", load("labels", "tokens"))
    if (data / "ocr").exists():
        print_report("Đầu ra OCR thật", load("ocr", "words"))


if __name__ == "__main__":
    main()
