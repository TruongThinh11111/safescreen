"""
safescreen/synth.py — Sinh ảnh chụp màn hình TỔNG HỢP (synthetic) có gán nhãn.

Mọi thông tin cá nhân (tên, CCCD, SĐT, địa chỉ, ...) đều do Faker / hàm ngẫu nhiên tạo ra,
KHÔNG dùng dữ liệu của người thật.

Có 7 loại màn hình được trộn ngẫu nhiên:
    id_card  — thẻ CCCD giả, mẫu cũ và mẫu "Căn cước" 2024 (chỉ có chữ, không có ảnh/quốc huy)
    bank     — biên lai chuyển tiền / màn hình số dư + lịch sử giao dịch
    chat     — đoạn tin nhắn (có cả tin không dấu)
    form     — form đăng ký / cập nhật thông tin (có cả ô trống chỉ có chữ gợi ý)
    sms      — tin nhắn SMS ngân hàng / OTP / nhà mạng (không dấu, số tài khoản bị che 1 phần)
    order    — trang xác nhận đơn hàng của app mua sắm
    profile  — trang hồ sơ cá nhân

TÁCH MẪU CÂU THEO TẬP (--template-split):
    Trong mỗi danh sách biến thể (mẫu câu chat/SMS, các cách viết nhãn trường "Số điện thoại" / "SĐT"
    / ..., tiêu đề), ~1/4 được dành RIÊNG cho tập test, ~3/4 cho train (xem split_variants). Sinh tập test với
    --template-split test thì mô hình bị đánh giá trên những CÁCH DIỄN ĐẠT CHƯA TỪNG THẤY khi train,
    thay vì chính các câu đã học thuộc.

Mỗi ảnh đi kèm 1 file JSON, chứa danh sách token (từ) theo thứ tự đọc:
    {"text": "Nguyễn", "bbox": [x0, y0, x1, y1], "label": "B-NAME", "line": 3}
Nhãn theo chuẩn BIO: B-X = từ đầu của thực thể X, I-X = các từ tiếp theo, O = không nhạy cảm.

Chạy thử (đứng ở thư mục ml/):
    python -m safescreen.synth --n 20 --out data/synth --preview 5 --template-split train
"""
from __future__ import annotations

import argparse
import io
import json
import random
import re
import zlib
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

from faker import Faker
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from safescreen.text_utils import bo_dau

ENTITY_TYPES = ["CCCD", "NAME", "DOB", "ADDRESS", "PHONE", "EMAIL", "MONEY", "ACCOUNT"]
O = "O"  # nhãn "không nhạy cảm"


# =====================================================================
# 1. Danh sách giá trị
# =====================================================================

PROVINCE_CODES = ["001", "031", "036", "038", "040", "046", "048", "052",
                  "056", "066", "075", "079", "082", "092", "096"]
PROVINCES = ["Hà Nội", "TP. Hồ Chí Minh", "Hải Phòng", "Đà Nẵng", "Cần Thơ", "Nghệ An", "Thanh Hóa",
             "Nam Định", "Bắc Ninh", "Quảng Ninh", "Lâm Đồng", "Khánh Hòa", "An Giang", "Huế"]
PHONE_PREFIXES = ["032", "033", "034", "035", "036", "037", "038", "039", "056", "058",
                  "070", "076", "077", "078", "079", "081", "082", "083", "084", "085",
                  "086", "088", "089", "090", "091", "093", "094", "096", "097", "098", "099"]
BANKS = ["Vietcombank", "VietinBank", "BIDV", "Agribank", "Techcombank",
         "MB Bank", "ACB", "VPBank", "TPBank", "Sacombank"]
EMAIL_DOMAINS = ["gmail.com", "gmail.com", "yahoo.com", "outlook.com", "fpt.edu.vn"]
# Địa danh / nơi công cộng — nhãn O (không phải địa chỉ cá nhân) → dạy mô hình không che nhầm
PLACES = ["Đà Lạt", "Phú Quốc", "Sa Pa", "Hội An", "Vũng Tàu", "Hạ Long", "Nha Trang", "Hồ Gươm",
          "Vincom Bà Triệu", "phố đi bộ Nguyễn Huệ", "Landmark 81", "AEON Mall", "chợ Bến Thành"]
SHOPS = ["Shopee", "Lazada", "Tiki", "TikTok Shop", "Sendo"]
CARRIERS = ["Viettel", "Vinaphone", "Mobifone"]
PRODUCTS = ["Áo thun nam", "Tai nghe bluetooth", "Sạc dự phòng", "Ốp lưng điện thoại", "Bình giữ nhiệt",
            "Sách lập trình Python", "Giày thể thao", "Kem chống nắng", "Chuột không dây", "Balo laptop"]

# Danh sách tên tự soạn (Faker vi_VN chỉ có ~35 tên và lẫn cả tên tiếng Anh như "John", "Jane")
LAST_NAMES = ["Nguyễn"] * 8 + ["Trần"] * 3 + ["Lê"] * 3 + ["Phạm"] * 2 + [
    "Hoàng", "Huỳnh", "Phan", "Vũ", "Võ", "Đặng", "Bùi", "Đỗ", "Hồ", "Ngô", "Dương", "Lý",
    "Đinh", "Trịnh", "Mai", "Lâm", "Tô", "Cao", "Hà", "Đoàn", "Lương", "Tạ", "Châu", "Quách"]
MIDDLE_MALE = ["Văn", "Văn", "Đức", "Minh", "Quang", "Hữu", "Thành", "Công", "Xuân", "Hoàng",
               "Tuấn", "Gia", "Bảo", "Trọng", "Thế", "Ngọc", "Đình", "Anh", "Duy", "Tiến"]
MIDDLE_FEMALE = ["Thị", "Thị", "Ngọc", "Thu", "Thanh", "Minh", "Phương", "Hồng", "Kim", "Mỹ",
                 "Bảo", "Thùy", "Khánh", "Hoài", "Diệu", "Như", "Quỳnh", "Lan", "Hải", "Tuyết"]
FIRST_MALE = ["An", "Anh", "Bảo", "Bình", "Cường", "Dũng", "Duy", "Đạt", "Đức", "Giang", "Hải",
              "Hiếu", "Hoàng", "Hùng", "Huy", "Khang", "Khoa", "Khôi", "Kiên", "Long", "Lâm", "Minh",
              "Nam", "Nghĩa", "Nhân", "Phát", "Phong", "Phúc", "Quân", "Quang", "Sơn", "Tài", "Thắng",
              "Thành", "Thịnh", "Toàn", "Trí", "Trung", "Tuấn", "Tùng", "Việt", "Vinh", "Vũ"]
FIRST_FEMALE = ["An", "Anh", "Châu", "Chi", "Diệp", "Dung", "Duyên", "Giang", "Hà", "Hạnh", "Hằng",
                "Hiền", "Hoa", "Hương", "Huyền", "Lan", "Linh", "Loan", "Ly", "Mai", "My", "Nga",
                "Ngân", "Ngọc", "Nhi", "Nhung", "Oanh", "Phương", "Quyên", "Quỳnh", "Tâm", "Thảo",
                "Thư", "Thủy", "Trang", "Trâm", "Uyên", "Vân", "Vy", "Xuân", "Yến"]


def split_variants(options: list) -> tuple[list, list]:
    """
    Chia 1 danh sách biến thể thành (train, test) một cách CỐ ĐỊNH (không phụ thuộc seed):
    sắp theo mã băm crc32 rồi lấy 1/4 đầu (ít nhất 1) làm nhóm test.
    Danh sách ít hơn 4 lựa chọn thì không tách (dùng chung), vì tách ra sẽ quá nghèo.
    """
    if len(options) < 4:
        return options, options
    ranked = sorted(options, key=lambda o: zlib.crc32(str(o).encode("utf-8")))
    k = max(1, len(options) // 4)
    return ranked[k:], ranked[:k]


# =====================================================================
# 2. Sinh giá trị cho từng loại thực thể
# =====================================================================

class FakeVN:
    """Sinh dữ liệu giả kiểu Việt Nam. Cùng seed → cùng kết quả (tái lập được)."""

    def __init__(self, seed: int, split: str = "all"):
        self.rng = random.Random(seed)
        self.fk = Faker("vi_VN")
        self.fk.seed_instance(seed)
        self.split = split  # "train" | "test" | "all"

    def pick(self, options: list):
        """Chọn ngẫu nhiên 1 biến thể, chỉ trong nhóm của tập hiện tại (xem --template-split)."""
        if self.split != "all":
            train, test = split_variants(options)
            options = test if self.split == "test" else train
        return self.rng.choice(options)

    def digits(self, n: int) -> str:
        return "".join(str(self.rng.randint(0, 9)) for _ in range(n))

    # ---- thực thể nhạy cảm ----
    def name(self, male: bool | None = None) -> str:
        # Tự ghép Họ + Đệm + Tên; tên đệm và tên chọn theo cùng giới tính (Văn ↔ nam, Thị ↔ nữ)
        male = self.rng.random() < 0.5 if male is None else male
        first = self.rng.choice(FIRST_MALE if male else FIRST_FEMALE)
        r = self.rng.random()
        if r < 0.1:  # tên 2 chữ: "Lê Hùng"
            return f"{self.rng.choice(LAST_NAMES)} {first}"
        middle = self.rng.choice(MIDDLE_MALE if male else MIDDLE_FEMALE)
        if r > 0.85:  # 2 tên đệm: "Nguyễn Thị Ngọc Anh"
            middle += " " + self.rng.choice(MIDDLE_MALE[2:] if male else MIDDLE_FEMALE[2:])
        return f"{self.rng.choice(LAST_NAMES)} {middle} {first}"

    def first_name(self) -> str:
        return self.rng.choice(FIRST_MALE + FIRST_FEMALE)

    def styled_name(self, name: str | None = None, styles=("title", "title", "upper", "plain_upper")) -> str:
        """Cùng 1 tên nhưng viết theo kiểu khác: 'Nguyễn Văn An' / 'NGUYỄN VĂN AN' / 'NGUYEN VAN AN'."""
        name = name or self.name()
        style = self.rng.choice(styles)
        if style == "upper":
            return name.upper()
        if style == "plain_upper":
            return bo_dau(name).upper()
        if style == "plain":
            return bo_dau(name)
        return name

    def birth_date(self) -> date:
        # Khoảng năm TRÙNG với ngày "không nhạy cảm" (other_date) → không thể đoán DOB chỉ từ năm
        start = date(1950, 1, 1)
        return start + timedelta(days=self.rng.randint(0, (date(2020, 12, 31) - start).days))

    def fmt_date(self, d: date) -> str:
        sep = self.rng.choice(["/", "/", "/", "-", "."])
        if self.rng.random() < 0.2:  # không thêm số 0: 5/3/1998
            return f"{d.day}{sep}{d.month}{sep}{d.year}"
        return f"{d.day:02d}{sep}{d.month:02d}{sep}{d.year}"

    def cccd(self, dob: date | None = None, male: bool | None = None) -> str:
        # Cấu trúc 12 số: 3 số mã tỉnh + 1 số (giới tính & thế kỷ) + 2 số cuối năm sinh + 6 số ngẫu nhiên
        dob = dob or self.birth_date()
        male = self.rng.random() < 0.5 if male is None else male
        gender_digit = (0 if dob.year < 2000 else 2) + (0 if male else 1)
        return f"{self.rng.choice(PROVINCE_CODES)}{gender_digit}{dob.year % 100:02d}{self.digits(6)}"

    def phone(self) -> str:
        d = self.rng.choice(PHONE_PREFIXES) + self.digits(7)  # SĐT di động 10 số
        style = self.rng.choice(["plain", "plain", "space", "dot", "dash", "intl", "intl_compact", "intl_paren"])
        return {"plain": d,
                "space": f"{d[:4]} {d[4:7]} {d[7:]}",
                "dot": f"{d[:4]}.{d[4:7]}.{d[7:]}",
                "dash": f"{d[:4]}-{d[4:7]}-{d[7:]}",
                "intl": f"+84 {d[1:4]} {d[4:7]} {d[7:]}",
                "intl_compact": f"+84{d[1:]}",
                "intl_paren": f"(+84) {d[1:4]} {d[4:7]} {d[7:]}"}[style]

    def email(self, name: str | None = None) -> str:
        parts = bo_dau(name or self.name()).lower().split()
        user = self.rng.choice([parts[-1] + parts[0], parts[-1] + "." + parts[0], parts[0] + "." + parts[-1],
                                "".join(p[0] for p in parts[:-1]) + parts[-1], parts[-1] + "_" + parts[0]])
        if self.rng.random() < 0.7:
            user += str(self.rng.randint(1, 2005))
        return f"{user}@{self.rng.choice(EMAIL_DOMAINS)}"

    def amount(self) -> int:
        return self.rng.choice([self.rng.randint(1, 999) * 1000,
                                self.rng.randint(1, 500) * 100_000,
                                self.rng.randint(1, 99) * 1_000_000])

    def money(self, chat: bool = False, sign: str = "", compact: bool = False) -> str:
        """
        chat=True   : có thể viết tắt kiểu tin nhắn (500k, 2tr, 1,5 triệu)
        sign        : "+" tiền vào, "-" tiền ra, "" không dấu
        compact=True: kiểu SMS ngân hàng, đơn vị dính liền ("+500,000VND")
        """
        amount = self.amount()
        if chat and self.rng.random() < 0.4:
            if amount < 1_000_000:
                return f"{amount // 1000}k"
            if amount % 1_000_000 and amount < 10_000_000:
                return f"{amount / 1_000_000:g} triệu".replace(".", ",")
            return f"{amount // 1_000_000}{self.rng.choice(['tr', ' triệu'])}"
        s = f"{amount:,}".replace(",", self.rng.choice([".", ","]))  # 1.500.000 hoặc 1,500,000
        if compact:
            return sign + s + self.rng.choice(["VND", "VNĐ", "đ"])
        if self.rng.random() < 0.1:  # đơn vị đứng trước: "VND 1.500.000"
            return self.rng.choice(["VND ", "VNĐ "]) + sign + s
        return sign + s + self.rng.choice([" VND", " VNĐ", "đ", " đ", " đồng", " ₫", "₫"])

    def account(self) -> str:
        n = self.rng.choice([9, 10, 12, 13, 14])  # 12 số → dễ nhầm với CCCD, CRF phải dựa ngữ cảnh
        d = str(self.rng.randint(1, 9)) + self.digits(n - 1)
        if self.rng.random() < 0.2:
            return " ".join(d[i:i + 4] for i in range(0, n, 4))
        return d

    def account_masked(self) -> str:
        """Số tài khoản bị che 1 phần như trong SMS ngân hàng: 1234xxx678 / xxxx5678."""
        d = self.account().replace(" ", "")
        return self.rng.choice([d[:4] + "xxx" + d[-3:], "xxxx" + d[-4:], d[:3] + "***" + d[-4:]])

    def address(self) -> str:
        addr = self.fk.address().replace("\n", ", ")
        if self.rng.random() < 0.2:  # địa chỉ rút gọn: bỏ phần tỉnh/thành
            addr = addr.rsplit(",", 1)[0]
        return addr

    # ---- "mồi nhử" KHÔNG nhạy cảm (nhãn O): số/ngày trông giống thực thể ----
    def other_date(self) -> str:
        """Ngày KHÔNG phải ngày sinh (giao dịch, cấp thẻ, sự kiện...). Khoảng năm trùng với birth_date."""
        start = date(2000, 1, 1)
        return self.fmt_date(start + timedelta(days=self.rng.randint(0, (date(2026, 9, 30) - start).days)))

    def time(self) -> str:
        return f"{self.rng.randint(0, 23):02d}:{self.rng.randint(0, 59):02d}"

    def txn_code(self) -> str:
        alnum = "ABCDEFGHJKLMNPQRSTUVWXYZ0123456789"
        return self.rng.choice(["FT" + self.digits(5) + "".join(self.rng.choice(alnum) for _ in range(6)),
                                self.digits(9),                       # số tham chiếu thuần số → mồi cho ACCOUNT
                                "MBVCB." + self.digits(10)])

    def order_code(self) -> str:
        return self.rng.choice(["#DH" + self.digits(8), "SPX" + self.digits(10) + "VN", "DH-" + self.digits(6)])

    def hotline(self) -> str:
        return f"{self.rng.choice(['1900', '1800'])} {self.digits(4)}"

    def otp(self) -> str:
        return self.digits(6)


# =====================================================================
# 3. Bộ vẽ màn hình: vẽ chữ từng từ một và ghi lại bbox + nhãn
# =====================================================================

FONT_CANDIDATES = [  # (thường, đậm) — đều hỗ trợ tiếng Việt
    ("arial.ttf", "arialbd.ttf"), ("segoeui.ttf", "segoeuib.ttf"),
    ("tahoma.ttf", "tahomabd.ttf"), ("verdana.ttf", "verdanab.ttf"),
    ("calibri.ttf", "calibrib.ttf"),
    ("Roboto-Regular.ttf", "Roboto-Bold.ttf"), ("NotoSans-Regular.ttf", "NotoSans-Bold.ttf"),
    ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf"),                    # có sẵn trên Colab/Kaggle
    ("LiberationSans-Regular.ttf", "LiberationSans-Bold.ttf"),
]
FONT_DIRS = [Path(__file__).resolve().parent.parent / "fonts",   # ml/fonts/ — chép thêm font vào đây
             Path("C:/Windows/Fonts"), Path("/usr/share/fonts")]


def find_fonts() -> list[tuple[str, str]]:
    def locate(name):
        for d in FONT_DIRS:
            if (d / name).exists():
                return str(d / name)
            if d.exists() and d.name != "Fonts":  # /usr/share/fonts có thư mục con → tìm đệ quy
                hits = list(d.rglob(name))
                if hits:
                    return str(hits[0])
        return None

    found = []
    for reg, bold in FONT_CANDIDATES:
        r = locate(reg)
        if r:
            found.append((r, locate(bold) or r))
    if not found:
        raise RuntimeError("Không tìm thấy font tiếng Việt. Hãy chép file .ttf (vd Roboto) vào ml/fonts/")
    return found


def blend(c1, c2, t):
    """Pha màu: t=0 → c1, t=1 → c2."""
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))


ACCENTS = [(0, 104, 255), (0, 140, 80), (200, 30, 40), (120, 60, 200), (0, 120, 140), (230, 110, 0)]


class Screen:
    """
    Một "màn hình" để vẽ. Vẽ trên 2 lớp:
      - base: nền, khung, bong bóng chat (vẽ sau chữ cũng được)
      - ink : lớp trong suốt chỉ chứa chữ
    Cuối cùng ghép ink lên base. Nhờ vậy có thể vẽ chữ trước rồi mới vẽ khung bao quanh.
    Toạ độ/cỡ chữ trong code viết theo chiều rộng chuẩn 720px, nhân với self.s khi vẽ.

    Mỗi màn hình còn ngẫu nhiên hoá bố cục (cỡ chữ, lề, giãn dòng, kiểu dòng nhãn–giá trị,
    có dấu ":" sau nhãn hay không) để mô hình không học thuộc vị trí.
    """

    def __init__(self, rng: random.Random, font_pair: tuple[str, str], width: int, dark: bool):
        self.rng, self.font_pair, self.W = rng, font_pair, width
        self.s = width / 720
        self.fs = rng.uniform(0.9, 1.15)          # hệ số cỡ chữ
        self.lh_factor = rng.uniform(1.3, 1.55)   # giãn dòng
        self.kv_stacked = rng.random() < 0.25     # nhãn nằm TRÊN giá trị thay vì cùng hàng
        self.colon = rng.random() < 0.5           # "Số tài khoản:" hay "Số tài khoản"
        accent = rng.choice(ACCENTS)
        if dark:
            self.c_bg, self.c_text, self.c_muted = (18, 18, 18), (235, 235, 235), (160, 160, 160)
            self.c_border, self.c_other = (70, 70, 70), (45, 45, 45)
            self.c_me = blend(accent, self.c_bg, 0.55)
        else:
            self.c_bg = rng.choice([(255, 255, 255), (245, 246, 248)])
            self.c_text, self.c_muted = (20, 20, 20), (110, 110, 110)
            self.c_border, self.c_other = (205, 205, 205), (233, 233, 235)
            self.c_me = blend(accent, (255, 255, 255), 0.78)
        self.accent = accent

        tall = int(4500 * self.s)  # canvas cao dư, cuối cùng sẽ cắt bớt
        self.base = Image.new("RGB", (width, tall), self.c_bg)
        self.ink = Image.new("RGBA", (width, tall), (0, 0, 0, 0))
        self.d_base, self.d_ink = ImageDraw.Draw(self.base), ImageDraw.Draw(self.ink)

        self.margin = int(rng.randint(24, 40) * self.s)
        self.y = int(12 * self.s)
        self.tokens: list[dict] = []
        self.line_id = 0
        self._fonts = {}

    # ---------- tiện ích ----------
    def font(self, size: int, bold: bool = False):
        key = (size, bold)
        if key not in self._fonts:
            self._fonts[key] = ImageFont.truetype(self.font_pair[bold], int(size * self.s * self.fs))
        return self._fonts[key]

    def lh(self, font) -> int:
        return int(font.size * self.lh_factor)  # chiều cao 1 dòng

    @staticmethod
    def words(segs) -> list[tuple[str, str]]:
        """[(đoạn chữ, thực thể)] → [(từ, nhãn BIO)]. Đây là chỗ gán nhãn B-/I- cho từng từ."""
        out = []
        for text, ent in segs:
            for k, w in enumerate(text.split()):
                out.append((w, O if ent == O else ("B-" if k == 0 else "I-") + ent))
        return out

    def wrap(self, words, font, max_w):
        """Xếp từ vào các dòng, xuống dòng khi vượt max_w. Mỗi dòng: [(từ, nhãn, độ rộng)]."""
        space = font.getlength(" ")
        lines, cur = [[]], 0.0
        for w, lab in words:
            ww = font.getlength(w)
            if lines[-1] and cur + space + ww > max_w:
                lines.append([])
                cur = 0.0
            cur += (space if lines[-1] else 0) + ww
            lines[-1].append((w, lab, ww))
        return [ln for ln in lines if ln]

    def line_width(self, line, font) -> float:
        return sum(ww for _, _, ww in line) + font.getlength(" ") * (len(line) - 1)

    def draw_line(self, line, font, x, y, color):
        """Vẽ 1 dòng và ghi lại bbox thật của từng từ (textbbox = khung bao sát nét chữ)."""
        space = font.getlength(" ")
        for w, lab, ww in line:
            self.d_ink.text((x, y), w, font=font, fill=color + (255,))
            x0, y0, x1, y1 = self.d_ink.textbbox((x, y), w, font=font)
            self.tokens.append({"text": w, "bbox": [int(x0), int(y0), int(x1), int(y1)],
                                "label": lab, "line": self.line_id})
            x += ww + space

    # ---------- các thành phần giao diện ----------
    def space(self, h: int):
        self.y += int(h * self.s)

    def text(self, segs, size=26, bold=False, color=None, align="left", gap=10):
        font = self.font(size, bold)
        x0, x1 = self.margin, self.W - self.margin
        for line in self.wrap(self.words(segs), font, x1 - x0):
            lw = self.line_width(line, font)
            x = {"left": x0, "center": (x0 + x1 - lw) / 2, "right": x1 - lw}[align]
            self.draw_line(line, font, x, self.y, color or self.c_text)
            self.line_id += 1
            self.y += self.lh(font)
        self.y += int(gap * self.s)

    def _key(self, key):
        """Nhãn của dòng nhãn–giá trị: chuỗi → segs, thêm ':' nếu màn hình này dùng dấu hai chấm."""
        if isinstance(key, str):
            return [(key + (":" if self.colon else ""), O)]
        return key

    def kv(self, key, value, size=24):
        """Dòng 'nhãn ........ giá trị' kiểu app ngân hàng (giá trị căn phải, tự xuống dòng)."""
        key = self._key(key)
        if self.kv_stacked:  # kiểu xếp chồng: nhãn nhỏ ở trên, giá trị đậm ở dưới
            self.text(key, size=size - 4, color=self.c_muted, gap=2)
            self.text(value, size=size, bold=True, gap=12)
            return
        fk, fv = self.font(size), self.font(size, bold=True)
        x0, x1 = self.margin, self.W - self.margin
        k_lines = self.wrap(self.words(key), fk, (x1 - x0) * 0.5)
        k_w = max(self.line_width(ln, fk) for ln in k_lines)
        v_lines = self.wrap(self.words(value), fv, x1 - (x0 + k_w + 24 * self.s))
        # Vẽ hết cột nhãn rồi mới tới cột giá trị, để token của 1 thực thể luôn liền nhau
        # (nếu vẽ xen kẽ theo hàng, tên ở cột trái có thể bị số tiền ở cột phải chen vào giữa)
        lh, y0, line0 = self.lh(fv), self.y, self.line_id
        for i, line in enumerate(k_lines):
            self.line_id = line0 + i  # nhãn và giá trị cùng hàng → cùng line id
            self.draw_line(line, fk, x0, y0 + i * lh, self.c_muted)
        for i, line in enumerate(v_lines):
            self.line_id = line0 + i
            self.draw_line(line, fv, x1 - self.line_width(line, fv), y0 + i * lh, self.c_text)
        rows = max(len(k_lines), len(v_lines))
        self.line_id = line0 + rows
        self.y = y0 + rows * lh + int(12 * self.s)

    def field(self, label, value, size=26, placeholder=False):
        """Ô nhập liệu của form: nhãn nhỏ phía trên, giá trị nằm trong khung viền.
        placeholder=True: ô trống chỉ có chữ gợi ý màu xám ("Nhập số điện thoại")."""
        self.text([(label, O)], size=20, color=self.c_muted, gap=4)
        font, pad = self.font(size), int(16 * self.s)
        x0, x1 = self.margin, self.W - self.margin
        lines = self.wrap(self.words(value), font, x1 - x0 - 2 * pad)
        h = len(lines) * self.lh(font) + 2 * pad
        self.d_base.rounded_rectangle([x0, self.y, x1, self.y + h], radius=int(12 * self.s),
                                      outline=self.c_border, width=max(1, int(2 * self.s)))
        y = self.y + pad
        for line in lines:
            self.draw_line(line, font, x0 + pad, y, self.c_muted if placeholder else self.c_text)
            self.line_id += 1
            y += self.lh(font)
        self.y += h + int(18 * self.s)

    def bubble(self, segs, me: bool, size=26):
        """Bong bóng chat: tin của mình (me=True) nằm bên phải, của người kia bên trái."""
        font, pad = self.font(size), int(18 * self.s)
        lines = self.wrap(self.words(segs), font, self.W * 0.68 - 2 * pad)
        w = max(self.line_width(ln, font) for ln in lines) + 2 * pad
        h = len(lines) * self.lh(font) + 2 * pad
        x = self.W - self.margin - w if me else self.margin
        self.d_base.rounded_rectangle([x, self.y, x + w, self.y + h], radius=int(22 * self.s),
                                      fill=self.c_me if me else self.c_other)
        y = self.y + pad
        for line in lines:
            self.draw_line(line, font, x + pad, y, self.c_text)
            self.line_id += 1
            y += self.lh(font)
        self.y += h + int(12 * self.s)

    def status_bar(self):
        font = self.font(20, bold=True)
        left = self.wrap(self.words([(f"{self.rng.randint(6, 23)}:{self.rng.randint(0, 59):02d}", O)]), font, self.W)[0]
        right = self.wrap(self.words([(f"{self.rng.choice(['4G', '5G', 'LTE', 'Wi-Fi'])} {self.rng.randint(5, 100)}%", O)]), font, self.W)[0]
        self.draw_line(left, font, self.margin, self.y, self.c_text)
        self.draw_line(right, font, self.W - self.margin - self.line_width(right, font), self.y, self.c_text)
        self.line_id += 1
        self.y += self.lh(font) + int(8 * self.s)

    def app_bar(self, segs):
        """Thanh tiêu đề màu nhấn, chữ trắng (tên app / tên người đang chat)."""
        h, font = int(96 * self.s), self.font(28, bold=True)
        self.d_base.rectangle([0, self.y, self.W, self.y + h], fill=self.accent)
        line = self.wrap(self.words(segs), font, self.W - 2 * self.margin)[0]
        self.draw_line(line, font, self.margin, self.y + (h - font.size) / 2, (255, 255, 255))
        self.line_id += 1
        self.y += h + int(20 * self.s)

    def button(self, label):
        pad = int(18 * self.s)
        y0 = self.y
        self.y += pad
        saved, self.c_text = self.c_text, (255, 255, 255)
        self.text([(label, O)], size=26, bold=True, align="center", gap=0)
        self.c_text = saved
        self.d_base.rounded_rectangle([self.margin, y0, self.W - self.margin, self.y + pad],
                                      radius=int(14 * self.s), fill=self.accent)
        self.y += pad + int(20 * self.s)

    def divider(self):
        self.d_base.line([self.margin, self.y, self.W - self.margin, self.y], fill=self.c_border,
                         width=max(1, int(2 * self.s)))
        self.y += int(20 * self.s)

    def card_begin(self, fill):
        """Bắt đầu 1 thẻ (vd CCCD): thu hẹp lề, chữ trong thẻ luôn màu tối."""
        self._card = (self.y, self.margin, self.c_text, self.c_muted, fill)
        self.margin += int(28 * self.s)
        self.c_text, self.c_muted = (25, 25, 25), (90, 90, 90)
        self.y += int(24 * self.s)

    def card_end(self):
        y0, margin, c_text, c_muted, fill = self._card
        self.d_base.rounded_rectangle([margin, y0, self.W - margin, self.y + int(12 * self.s)],
                                      radius=int(20 * self.s), fill=fill, outline=(150, 150, 150))
        self.margin, self.c_text, self.c_muted = margin, c_text, c_muted
        self.y += int(36 * self.s)

    def finish(self):
        """Cắt canvas theo chiều cao nội dung (tối thiểu tỉ lệ màn hình điện thoại 1:2) và ghép 2 lớp."""
        H = min(max(int(self.W * 2.0), self.y + int(40 * self.s)), self.base.height)
        img = Image.alpha_composite(self.base.crop((0, 0, self.W, H)).convert("RGBA"),
                                    self.ink.crop((0, 0, self.W, H))).convert("RGB")
        return img, [t for t in self.tokens if t["bbox"][3] <= H]


# =====================================================================
# 4. Mẫu câu (chat / SMS) — {X}: X thuộc ENTITY_TYPES → nhãn thực thể, còn lại → nhãn O
# =====================================================================

CHAT_SENSITIVE = [
    "Số điện thoại của mình là {PHONE}",
    "Có gì gọi em số {PHONE}",
    "SĐT mới của anh: {PHONE}",
    "Lưu số này giúp chị {PHONE}",
    "Zalo mình {PHONE} nha",
    "Gửi hàng về địa chỉ {ADDRESS} giúp em",
    "Địa chỉ nhà mình: {ADDRESS}",
    "Mình chuyển nhà rồi, giờ ở {ADDRESS}",
    "Ship tới {ADDRESS}, người nhận {NAME}, sđt {PHONE}",
    "Email của mình là {EMAIL}",
    "Gửi file qua mail {EMAIL}",
    "Mail công ty: {EMAIL}",
    "Chuyển khoản vào STK {ACCOUNT} {BANK} chủ TK {NAME}",
    "{BANK} {ACCOUNT} - {NAME}",
    "Số tài khoản của em là {ACCOUNT} ngân hàng {BANK}",
    "Em chuyển {MONEY} rồi, anh kiểm tra giúp",
    "Tháng này lương được {MONEY} thôi",
    "Còn nợ mình {MONEY} đó",
    "Tiền nhà tháng này {MONEY}",
    "Số CCCD của tôi là {CCCD}",
    "CCCD: {CCCD}, ngày sinh {DOB}",
    "Căn cước của bố: {CCCD}",
    "Sinh nhật {NAME} là ngày {DOB}",
    "Em sinh ngày {DOB}",
    "Ngày sinh của con: {DOB}",
    "Bạn {NAME} bên phòng kế toán nhờ chuyển lời",
    "Họ tên: {NAME} - SĐT: {PHONE}",
    "Anh {FIRST} ơi, mai họp lúc mấy giờ?",
    "Chị {FIRST} nhắn là chiều nay qua lấy đồ",
    "Cho mình xin info: {NAME}, {DOB}, {ADDRESS}",
    "Liên hệ {NAME} theo số {PHONE}",
    "{FIRST} ơi chuyển giúp chị {MONEY} vào {ACCOUNT}",
]
CHAT_NEUTRAL = [
    "Ok anh", "Cảm ơn bạn nhiều!", "Ừ, để tối mình xem", "Mai mấy giờ họp nhỉ?",
    "Họp lúc {TIME} sáng mai nhé", "Đơn hàng {ORDER} đã giao thành công",
    "Gọi tổng đài {HOTLINE} hỏi thử xem", "Hôm {ODATE} trời mưa to quá",
    "Đang kẹt xe, chờ mình {N} phút", "Đang giảm giá {PCT} đó", "Phòng {N} tầng 3 nha",
    "Bạn ơi mai đi học không?", "Bạn xem giúp mình file này với",
    "Tuần sau mình đi {PLACE} chơi", "Hẹn ở {PLACE} lúc {TIME} nhé",
    "Mã OTP là {OTP}, đừng gửi cho ai nhé", "Hạn nộp báo cáo là {ODATE}",
    "Sự kiện diễn ra ngày {ODATE}", "Lớp mình có {N} bạn", "Mua {N} cái thôi",
    "Đã nhận được rồi nhé", "Haha", "Để mình hỏi lại sếp", "Mai mình gọi lại sau",
]
CHAT_PREFIXES = ["À", "Ừ", "Nhớ nhé,", "Anh ơi,", "Ê,", "Dạ"]
CHAT_SUFFIXES = ["nhé", "nha", "ạ", "đó", "nhé!", "luôn"]

SMS_TEMPLATES = [  # SMS ngân hàng/nhà mạng thường KHÔNG DẤU
    "{BANK}: TK {ACCMASK} {SMONEY} luc {TIME} {ODATE}. SD: {CMONEY}. ND: {NAMEU} chuyen tien",
    "{BANK}: So du TK {ACCMASK} thay doi {SMONEY} luc {TIME}. SD {CMONEY}.",
    "TK {ACCMASK} {SMONEY} {TIME} {ODATE} SD {CMONEY} ND: {NAMEU} CK",
    "Ma OTP cua Quy khach la {OTP}. Tuyet doi khong chia se ma nay cho bat ky ai.",
    "{OTP} la ma xac thuc giao dich cua ban, hieu luc 5 phut.",
    "[{SHOP}] Don hang {ORDER} dang duoc giao den ban. LH {HOTLINE}.",
    "{CARRIER}: Nap thanh cong {CMONEY} cho thue bao {PHONE} luc {TIME} {ODATE}.",
    "{CARRIER}: Thue bao {PHONE} con {CMONEY} trong tai khoan chinh.",
    "Quy khach {NAMEU} vui long thanh toan {CMONEY} truoc ngay {ODATE}. LH {HOTLINE}.",
    "Chuc mung sinh nhat Quy khach {NAMEU}! Tang ban voucher {PCT}.",
]


def fill_template(tpl: str, fk: FakeVN, chat: bool = True) -> list[tuple[str, str]]:
    """'Số của mình là {PHONE} nhé' → [('Số của mình là', O), ('0912345678', 'PHONE'), ('nhé', O)]"""
    fillers = {  # tên chỗ trống → (hàm sinh giá trị, nhãn)
        "CCCD": (fk.cccd, "CCCD"), "NAME": (lambda: fk.styled_name(), "NAME"),
        "FIRST": (fk.first_name, "NAME"),
        "NAMEU": (lambda: fk.styled_name(styles=("plain_upper",)), "NAME"),
        "DOB": (lambda: fk.fmt_date(fk.birth_date()), "DOB"), "ADDRESS": (fk.address, "ADDRESS"),
        "PHONE": (fk.phone, "PHONE"), "EMAIL": (fk.email, "EMAIL"),
        "MONEY": (lambda: fk.money(chat=chat), "MONEY"),
        "SMONEY": (lambda: fk.money(sign=fk.rng.choice("+-"), compact=True), "MONEY"),
        "CMONEY": (lambda: fk.money(compact=True), "MONEY"),
        "ACCOUNT": (fk.account, "ACCOUNT"), "ACCMASK": (fk.account_masked, "ACCOUNT"),
        "BANK": (lambda: fk.rng.choice(BANKS), O), "TIME": (fk.time, O), "ORDER": (fk.order_code, O),
        "HOTLINE": (fk.hotline, O), "ODATE": (fk.other_date, O), "OTP": (fk.otp, O),
        "N": (lambda: str(fk.rng.randint(2, 50)), O), "PCT": (lambda: f"{fk.rng.randint(5, 70)}%", O),
        "PLACE": (lambda: fk.rng.choice(PLACES), O), "SHOP": (lambda: fk.rng.choice(SHOPS), O),
        "CARRIER": (lambda: fk.rng.choice(CARRIERS), O),
    }
    # re.split với nhóm (...) giữ lại tên chỗ trống: phần tử lẻ là tên, phần tử chẵn là chữ thường
    parts = re.split(r"\{(\w+)\}", tpl)
    segs = [[p, O] if i % 2 == 0 else [fillers[p][0](), fillers[p][1]] for i, p in enumerate(parts)]
    # Dấu câu viết LIỀN với chỗ trống ("[{SHOP}]", "{ODATE}.", "{CCCD},") phải dính vào giá trị,
    # nếu không sẽ thành từ riêng "[ Sendo ]" — điều không bao giờ có trên màn hình thật.
    for i in range(1, len(segs), 2):
        before, after = segs[i - 1], segs[i + 1]
        if m := re.search(r"\S+$", before[0]):
            segs[i][0] = m.group() + segs[i][0]
            before[0] = before[0][:m.start()]
        if m := re.match(r"\S+", after[0]):
            segs[i][0] += m.group()
            after[0] = after[0][m.end():]
    return [(t.strip(), lab) for t, lab in segs if t.strip()]


# =====================================================================
# 5. Các mẫu màn hình
# =====================================================================

def screen_id_card(sc: Screen, fk: FakeVN):
    rng = fk.rng
    male, dob = rng.random() < 0.5, fk.birth_date()
    sc.status_bar()
    sc.app_bar([(fk.pick(["Ảnh", "Thư viện", "Ảnh đã chụp", "Tài liệu", "Album", "Ảnh gần đây"]), O)])
    sc.space(30)
    sc.card_begin(fill=rng.choice([(232, 242, 228), (230, 240, 250), (245, 240, 225)]))
    sc.text([("CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM", O)], size=19, bold=True, align="center", gap=0)
    sc.text([("Độc lập - Tự do - Hạnh phúc", O)], size=18, align="center", gap=8)
    if rng.random() < 0.5:  # mẫu CCCD cũ
        sc.text([("CĂN CƯỚC CÔNG DÂN", O)], size=30, bold=True, align="center", color=(200, 30, 40), gap=12)
        sc.text([("Số / No.:", O), (fk.cccd(dob, male), "CCCD")], size=26, bold=True)
        sc.text([("Họ và tên / Full name:", O)], size=20, gap=0)
        sc.text([(fk.name(male).upper(), "NAME")], size=26, bold=True)
        sc.text([("Ngày sinh / Date of birth:", O), (fk.fmt_date(dob), "DOB")], size=22)
        sc.text([("Giới tính / Sex:", O), ("Nam" if male else "Nữ", O),
                 ("Quốc tịch / Nationality:", O), ("Việt Nam", O)], size=22)
        sc.text([("Quê quán / Place of origin:", O)], size=20, gap=0)
        sc.text([(fk.address(), "ADDRESS")], size=22)
        sc.text([("Nơi thường trú / Place of residence:", O)], size=20, gap=0)
        sc.text([(fk.address(), "ADDRESS")], size=22)
        expiry = date(dob.year + rng.choice([25, 40, 60]), dob.month, min(dob.day, 28))
        sc.text([("Có giá trị đến:", O), (fk.fmt_date(expiry), O)], size=20)  # ngày nhưng KHÔNG phải DOB
    else:  # mẫu "Căn cước" 2024
        sc.text([("CĂN CƯỚC", O)], size=30, bold=True, align="center", color=(200, 30, 40), gap=0)
        sc.text([("IDENTITY CARD", O)], size=20, align="center", gap=12)
        sc.text([("Số định danh cá nhân / Personal identification number", O)], size=18, gap=0)
        sc.text([(fk.cccd(dob, male), "CCCD")], size=28, bold=True)
        sc.text([("Họ, chữ đệm và tên khai sinh / Full name", O)], size=18, gap=0)
        sc.text([(fk.name(male).upper(), "NAME")], size=26, bold=True)
        sc.text([("Ngày, tháng, năm sinh / Date of birth", O)], size=18, gap=0)
        sc.text([(fk.fmt_date(dob), "DOB")], size=24)
        sc.text([("Giới tính / Sex:", O), ("Nam" if male else "Nữ", O),
                 ("Quốc tịch / Nationality:", O), ("Việt Nam", O)], size=22)
        sc.text([("Nơi đăng ký khai sinh / Place of birth", O)], size=18, gap=0)
        sc.text([(rng.choice(PROVINCES), "ADDRESS")], size=22)
        sc.text([("Nơi cư trú / Place of residence", O)], size=18, gap=0)
        sc.text([(fk.address(), "ADDRESS")], size=22)
        sc.text([("Ngày cấp / Date of issue:", O), (fk.other_date(), O)], size=20)
    sc.card_end()


def screen_bank(sc: Screen, fk: FakeVN):
    rng = fk.rng
    sc.status_bar()
    sc.app_bar([(rng.choice(BANKS), O)])
    if rng.random() < 0.6:  # --- biên lai chuyển tiền ---
        sc.space(20)
        sc.text([(fk.pick(["Chuyển tiền thành công", "Giao dịch thành công", "Chuyển khoản thành công",
                           "Thành công", "Hoàn tất giao dịch"]), O)],
                size=30, bold=True, align="center", color=sc.accent)
        sc.text([(fk.money(), "MONEY")], size=40, bold=True, align="center", gap=24)
        sc.divider()
        rows = [
            (fk.pick(["Người nhận", "Người thụ hưởng", "Tên người nhận", "Đến", "Tên tài khoản nhận"]),
             [(fk.styled_name(), "NAME")]),
            (fk.pick(["Số tài khoản", "Tài khoản nhận", "STK", "Đến tài khoản", "Số TK thụ hưởng"]),
             [(fk.account(), "ACCOUNT")]),
            (fk.pick(["Ngân hàng", "Ngân hàng nhận", "NH thụ hưởng", "Tại ngân hàng"]), [(rng.choice(BANKS), O)]),
        ]
        if rng.random() < 0.4:
            rows.insert(0, (fk.pick(["Tài khoản nguồn", "Từ tài khoản", "TK chuyển", "Tài khoản trích nợ"]),
                            [(fk.account(), "ACCOUNT")]))
        content = rng.random()
        if content < 0.5:
            value = [(fk.styled_name(styles=("plain_upper",)), "NAME"),
                     (fk.pick(["chuyen tien", "CK", "thanh toan tien nha", "tra tien hang", "gui tien"]), O)]
        elif content < 0.75:
            value = [(fk.pick(["chuyen tien cho", "tra no", "gui", "tien hoc cua"]), O),
                     (fk.styled_name(styles=("plain_upper", "plain")), "NAME")]
        else:  # nội dung không có tên
            value = [(fk.pick(["tien nha thang 10", "thanh toan hoa don", "dong hoc phi", "mua hang online"]), O)]
        rows += [
            (fk.pick(["Nội dung", "Nội dung chuyển khoản", "Lời nhắn", "Diễn giải"]), value),
            (fk.pick(["Mã giao dịch", "Số tham chiếu", "Mã GD", "Số lệnh"]), [(fk.txn_code(), O)]),
            (fk.pick(["Thời gian", "Ngày giao dịch", "Thời gian giao dịch", "Ngày thực hiện"]),
             [(f"{fk.time()} {fk.other_date()}", O)]),
            (fk.pick(["Phí giao dịch", "Phí", "Phí chuyển tiền", "Phí dịch vụ"]),
             [("Miễn phí", O)] if rng.random() < 0.6 else [(fk.money(), "MONEY")]),
        ]
        for key, value in rows:
            sc.kv(key, value)
    else:  # --- màn hình tài khoản + lịch sử giao dịch ---
        sc.text([(fk.pick(["Xin chào,", "Chào buổi sáng,", "Chào mừng trở lại,", "Hi,"]), O)],
                size=22, color=sc.c_muted, gap=0)
        sc.text([(fk.styled_name(styles=("upper", "title")), "NAME")], size=30, bold=True)
        sc.kv(fk.pick(["Tài khoản thanh toán", "Tài khoản chính", "Số tài khoản", "TK thanh toán"]),
              [(fk.account(), "ACCOUNT")])
        sc.kv(fk.pick(["Số dư khả dụng", "Số dư", "Số dư tài khoản", "Số dư hiện tại"]), [(fk.money(), "MONEY")])
        sc.divider()
        sc.text([(fk.pick(["Lịch sử giao dịch", "Giao dịch gần đây", "Biến động số dư", "Hoạt động gần đây"]), O)],
                size=26, bold=True)
        for _ in range(rng.randint(3, 6)):
            kind, sign = rng.random(), "-"
            if kind < 0.35:
                key, sign = [(fk.pick(["Nhận tiền từ", "Tiền vào từ", "Nhận từ", "Từ"]), O),
                             (fk.styled_name(styles=("plain_upper",)), "NAME")], "+"
            elif kind < 0.7:
                key = [(fk.pick(["Thanh toán QR", "Nạp tiền điện thoại", "Thanh toán hóa đơn điện",
                                 "Rút tiền ATM", "Thanh toán thẻ", "Hoàn tiền"]), O)]
            else:
                key = [(fk.pick(["Chuyển tiền đến", "CK đến", "Chuyển đến", "Tới"]), O), (fk.account(), "ACCOUNT")]
            sc.kv(key + [(fk.other_date(), O)], [(fk.money(sign=sign), "MONEY")], size=22)
    sc.space(20)
    sc.text([("Hotline:", O), (fk.hotline(), O)], size=20, color=sc.c_muted, align="center")


def screen_chat(sc: Screen, fk: FakeVN):
    rng = fk.rng
    sc.status_bar()
    if rng.random() < 0.75:
        sc.app_bar([(fk.styled_name(styles=("title", "title", "plain")), "NAME")])  # tên người đang chat
    else:
        sc.app_bar([(fk.pick(["Nhóm lớp 12A1", "Team Marketing", "Gia đình", "Hội đồng hương",
                              "Phòng Kế toán", "CLB Bóng đá"]), O)])
    for _ in range(rng.randint(5, 10)):
        if rng.random() < 0.25:
            sc.text([(fk.time(), O)], size=18, color=sc.c_muted, align="center", gap=6)
        segs = fill_template(fk.pick(CHAT_SENSITIVE if rng.random() < 0.45 else CHAT_NEUTRAL), fk)
        # Ghép thêm từ đệm đầu/cuối câu cho câu không lặp lại y hệt
        if rng.random() < 0.2:
            segs.insert(0, (rng.choice(CHAT_PREFIXES), O))
        if rng.random() < 0.4:
            segs.append((rng.choice(CHAT_SUFFIXES), O))
        if rng.random() < 0.25:  # khoảng 1/4 tin nhắn gõ không dấu
            segs = [(bo_dau(t), e) for t, e in segs]
        sc.bubble(segs, me=rng.random() < 0.5)


FORM_FIELDS = [  # (các cách viết nhãn, thực thể, hàm sinh giá trị)
    (["Họ và tên", "Họ tên", "Tên khách hàng", "Họ tên người nhận", "Tên đầy đủ", "Full name"],
     "NAME", lambda fk: fk.styled_name(styles=("title", "title", "upper"))),
    (["Ngày sinh", "Ngày tháng năm sinh", "Sinh ngày", "Date of birth", "DOB"],
     "DOB", lambda fk: fk.fmt_date(fk.birth_date())),
    (["Số CCCD", "Số căn cước công dân", "CCCD/CMND", "Số giấy tờ tùy thân", "Số định danh cá nhân"],
     "CCCD", lambda fk: fk.cccd()),
    (["Số điện thoại", "SĐT", "Điện thoại liên hệ", "Số di động", "Phone"], "PHONE", lambda fk: fk.phone()),
    (["Email", "Địa chỉ email", "E-mail", "Thư điện tử"], "EMAIL", lambda fk: fk.email()),
    (["Địa chỉ", "Địa chỉ thường trú", "Địa chỉ nhận hàng", "Nơi ở hiện tại", "Địa chỉ liên hệ"],
     "ADDRESS", lambda fk: fk.address()),
    (["Số tài khoản ngân hàng", "STK nhận tiền", "Số tài khoản", "Tài khoản thụ hưởng"],
     "ACCOUNT", lambda fk: fk.account()),
]
FORM_NEUTRAL = [
    (["Giới tính"], lambda fk: fk.rng.choice(["Nam", "Nữ"])),
    (["Nghề nghiệp", "Công việc"], lambda fk: fk.rng.choice(["Sinh viên", "Kỹ sư", "Giáo viên", "Kinh doanh"])),
    (["Mã giới thiệu", "Mã khuyến mãi"], lambda fk: "GT" + fk.digits(6)),
    (["Ngày cấp", "Ngày cấp CCCD"], lambda fk: fk.other_date()),
    (["Nơi cấp"], lambda fk: "Cục Cảnh sát QLHC về TTXH"),
]


def screen_form(sc: Screen, fk: FakeVN):
    rng = fk.rng
    sc.status_bar()
    sc.app_bar([(fk.pick(["Đăng ký tài khoản", "Thông tin cá nhân", "Cập nhật hồ sơ", "Xác thực thông tin",
                          "Đăng ký nhận hàng", "Mở tài khoản", "eKYC"]), O)])
    # Luôn có Họ tên, thêm ngẫu nhiên 2–6 trường nhạy cảm khác (giữ thứ tự tự nhiên) + 0–2 trường thường
    chosen = [FORM_FIELDS[0]] + sorted(rng.sample(FORM_FIELDS[1:], rng.randint(2, 6)), key=FORM_FIELDS.index)
    items = []
    for labels, ent, gen in chosen:
        if rng.random() < 0.12:  # ô trống chỉ có chữ gợi ý → nhãn O
            label = fk.pick(labels)
            items.append((label, [(f"Nhập {label.lower()}", O)], True))
        else:
            items.append((fk.pick(labels), [(gen(fk), ent)], False))
    for labels, gen in rng.sample(FORM_NEUTRAL, rng.randint(0, 2)):
        items.insert(rng.randint(1, len(items)), (rng.choice(labels), [(gen(fk), O)], False))
    for label, value, placeholder in items:
        sc.field(label, value, placeholder=placeholder)
    sc.text([(fk.pick(["Tôi đồng ý với Điều khoản sử dụng", "Tôi đã đọc và đồng ý chính sách bảo mật",
                       "Xác nhận thông tin trên là chính xác", "Nhận thông báo khuyến mãi"]), O)],
            size=20, color=sc.c_muted)
    sc.button(fk.pick(["Tiếp tục", "Đăng ký", "Lưu thông tin", "Xác nhận", "Hoàn tất"]))


def screen_sms(sc: Screen, fk: FakeVN):
    rng = fk.rng
    sc.status_bar()
    sc.app_bar([(rng.choice(BANKS + CARRIERS + SHOPS + ["Tin nhắn"]), O)])
    for _ in range(rng.randint(3, 6)):
        sc.text([(fk.other_date(), O), (fk.time(), O)], size=18, color=sc.c_muted, align="center", gap=6)
        sc.bubble(fill_template(fk.pick(SMS_TEMPLATES), fk, chat=False), me=False, size=24)


def screen_order(sc: Screen, fk: FakeVN):
    rng = fk.rng
    sc.status_bar()
    sc.app_bar([(fk.pick(["Thanh toán", "Xác nhận đơn hàng", "Chi tiết đơn hàng", "Đặt hàng"]), O)])
    sc.text([(fk.pick(["Địa chỉ nhận hàng", "Thông tin giao hàng", "Giao đến", "Người nhận hàng"]), O)],
            size=22, bold=True, color=sc.accent, gap=4)
    sc.text([(fk.styled_name(styles=("title",)), "NAME"), ("|", O), (fk.phone(), "PHONE")], size=24, gap=2)
    sc.text([(fk.address(), "ADDRESS")], size=22, color=sc.c_muted)
    sc.divider()
    for _ in range(rng.randint(1, 3)):
        sc.kv([(rng.choice(PRODUCTS), O), (f"x{rng.randint(1, 3)}", O)], [(fk.money(), "MONEY")], size=22)
    sc.divider()
    sc.kv(fk.pick(["Phí vận chuyển", "Phí ship", "Phí giao hàng", "Vận chuyển"]), [(fk.money(), "MONEY")])
    if rng.random() < 0.5:
        sc.kv(fk.pick(["Giảm giá", "Voucher", "Khuyến mãi", "Mã giảm giá"]), [(fk.money(sign="-"), "MONEY")])
    sc.kv(fk.pick(["Tổng thanh toán", "Tổng cộng", "Thành tiền", "Tổng tiền"]), [(fk.money(), "MONEY")])
    sc.kv(fk.pick(["Phương thức thanh toán", "Thanh toán bằng", "Hình thức thanh toán"]),
          [(rng.choice(["Thanh toán khi nhận hàng", "Ví điện tử", "Thẻ ATM nội địa"]), O)])
    sc.kv(fk.pick(["Mã đơn hàng", "Mã vận đơn", "Số đơn hàng", "Order ID"]), [(fk.order_code(), O)])
    sc.kv(fk.pick(["Thời gian đặt hàng", "Ngày đặt", "Đặt lúc", "Ngày tạo đơn"]),
          [(f"{fk.time()} {fk.other_date()}", O)])
    sc.button(fk.pick(["Đặt hàng", "Mua lại", "Xác nhận", "Liên hệ shop"]))


def screen_profile(sc: Screen, fk: FakeVN):
    rng = fk.rng
    sc.status_bar()
    sc.app_bar([(fk.pick(["Tài khoản", "Hồ sơ của tôi", "Thông tin cá nhân", "Cá nhân"]), O)])
    sc.space(10)
    sc.text([(fk.styled_name(styles=("title", "upper")), "NAME")], size=30, bold=True, align="center", gap=2)
    sc.text([(fk.pick(["Thành viên Vàng", "Thành viên Bạc", "Hạng Kim cương", "Thành viên mới"]), O)],
            size=20, color=sc.c_muted, align="center", gap=18)
    sc.divider()
    rows = [
        (["Số điện thoại", "Điện thoại", "SĐT", "Di động"], [(fk.phone(), "PHONE")]),
        (["Email", "Địa chỉ email", "E-mail"], [(fk.email(), "EMAIL")]),
        (["Ngày sinh", "Sinh nhật", "Ngày tháng năm sinh"], [(fk.fmt_date(fk.birth_date()), "DOB")]),
        (["Giới tính"], [(rng.choice(["Nam", "Nữ"]), O)]),
        (["Địa chỉ", "Địa chỉ mặc định", "Nơi ở"], [(fk.address(), "ADDRESS")]),
        (["Ngày tham gia", "Thành viên từ"], [(fk.other_date(), O)]),
    ]
    for labels, value in rng.sample(rows, rng.randint(3, len(rows))):
        sc.kv(fk.pick(labels), value)
    sc.divider()
    for item in rng.sample(["Đổi mật khẩu", "Cài đặt", "Trung tâm trợ giúp", "Đăng xuất", "Ví của tôi"], 3):
        sc.text([(item, O)], size=24, gap=14)


SCREENS = {"id_card": screen_id_card, "bank": screen_bank, "chat": screen_chat, "form": screen_form,
           "sms": screen_sms, "order": screen_order, "profile": screen_profile}


# =====================================================================
# 6. Làm "bẩn" ảnh cho giống ảnh thật (nén JPEG, mờ nhẹ, co giãn)
#    Kích thước ảnh giữ nguyên nên bbox không đổi.
# =====================================================================

def degrade(img: Image.Image, rng: random.Random) -> Image.Image:
    if rng.random() < 0.15:  # thu nhỏ rồi phóng lại → chữ hơi vỡ như ảnh bị gửi qua app chat
        f = rng.uniform(0.6, 0.9)
        small = img.resize((int(img.width * f), int(img.height * f)), Image.BILINEAR)
        img = small.resize(img.size, Image.BILINEAR)
    if rng.random() < 0.15:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 0.8)))
    if rng.random() < 0.5:
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=rng.randint(55, 95))
        img = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    return img


def generate_one(seed: int, fonts, split: str = "all") -> tuple[str, Image.Image, list[dict]]:
    fk = FakeVN(seed, split)
    rng = fk.rng  # dùng chung 1 bộ random → cùng seed sinh ra đúng cùng ảnh
    stype = rng.choice(list(SCREENS))
    sc = Screen(rng, rng.choice(fonts), width=rng.choice([720, 1080]), dark=rng.random() < 0.3)
    SCREENS[stype](sc, fk)
    img, tokens = sc.finish()
    return stype, degrade(img, rng), tokens


# =====================================================================
# 7. Vẽ bbox để kiểm tra bằng mắt
# =====================================================================

PREVIEW_COLORS = {"CCCD": "red", "NAME": "blue", "DOB": "purple", "ADDRESS": "darkgreen",
                  "PHONE": "orange", "EMAIL": "magenta", "MONEY": "brown", "ACCOUNT": "cyan"}


def draw_preview(img: Image.Image, tokens: list[dict]) -> Image.Image:
    img = img.copy()
    d = ImageDraw.Draw(img)
    for t in tokens:
        if t["label"] != O:
            ent = t["label"][2:]
            d.rectangle(t["bbox"], outline=PREVIEW_COLORS[ent], width=2)
            if t["label"].startswith("B-"):
                d.text((t["bbox"][0], t["bbox"][1] - 12), ent, fill=PREVIEW_COLORS[ent])
    return img


def main():
    ap = argparse.ArgumentParser(description="Sinh dữ liệu ảnh chụp màn hình tổng hợp cho Safescreen")
    ap.add_argument("--n", type=int, default=100, help="số ảnh cần sinh")
    ap.add_argument("--out", default="data/synth", help="thư mục đầu ra")
    ap.add_argument("--seed", type=int, default=42, help="ảnh thứ i dùng seed + i")
    ap.add_argument("--template-split", choices=["train", "test", "all"], default="all",
                    help="chỉ dùng mẫu câu/biến thể thuộc nhóm này (train cho train/val, test cho test)")
    ap.add_argument("--preview", type=int, default=0, help="lưu thêm N ảnh có vẽ bbox để kiểm tra")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    if args.preview:
        (out / "preview").mkdir(exist_ok=True)

    fonts = find_fonts()
    print(f"Tìm thấy {len(fonts)} bộ font: {[Path(r).name for r, _ in fonts]}")
    screen_count, entity_count = Counter(), Counter()

    for i in range(args.n):
        seed = args.seed + i
        stype, img, tokens = generate_one(seed, fonts, args.template_split)
        name = f"{i:05d}"
        img.save(out / "images" / f"{name}.png")
        meta = {"image": f"images/{name}.png", "screen_type": stype, "seed": seed,
                "template_split": args.template_split,
                "width": img.width, "height": img.height, "tokens": tokens}
        (out / "labels" / f"{name}.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        if i < args.preview:
            draw_preview(img, tokens).save(out / "preview" / f"{name}.png")

        screen_count[stype] += 1
        entity_count.update(t["label"][2:] for t in tokens if t["label"].startswith("B-"))

    print(f"Đã sinh {args.n} ảnh vào {out.resolve()}")
    print("Loại màn hình:", dict(screen_count))
    print("Số thực thể  :", {e: entity_count[e] for e in ENTITY_TYPES})


if __name__ == "__main__":
    main()
