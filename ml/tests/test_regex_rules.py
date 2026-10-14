"""
Unit test cho safescreen/regex_rules.py. Chạy (đứng ở thư mục ml/):  pytest -v

Mỗi luật có 2 nhóm test:
  - POSITIVE: (văn bản, chuỗi phải được bắt) → find_entities phải trả về đúng nhãn, đúng chuỗi
  - NEGATIVE: văn bản KHÔNG được bắt ra nhãn đó (có thể là nhãn khác, vd STK 12 số → ACCOUNT chứ không phải CCCD)
"""
import pytest

from safescreen.regex_rules import LOOSE, TIGHT, find_entities, normalize_ocr


def found(text):
    """[(nhãn, chuỗi khớp, điểm)] mà find_entities trả về."""
    return [(m.label, text[m.start:m.end], m.score) for m in find_entities(text)]


def assert_found(text, label, expected, score=None):
    hits = [(lab, s, sc) for lab, s, sc in found(text) if lab == label and s == expected]
    assert hits, f"{label} {expected!r} không được bắt trong {text!r}; kết quả: {found(text)}"
    if score is not None:
        assert hits[0][2] == score, f"điểm {hits[0][2]} != {score} cho {expected!r}"


def assert_not_found(text, label):
    assert all(lab != label for lab, _, _ in found(text)), f"{label} bị bắt nhầm trong {text!r}: {found(text)}"


# ---------------------------------------------------------------- chuẩn hoá OCR
@pytest.mark.parametrize("raw, expected", [
    ("CCCD: O79203O12345", "CCCD: 079203012345"),
    ("STK 1O25455l61", "STK 1025455161"),
    ("Idviet1570@yahoo.com", "Idviet1570@yahoo.com"),   # chữ trong email không bị đổi
    ("No.:075151039465", "No.:075151039465"),           # "No" không thành "N0"
    ("Ngày 05–12–1998", "Ngày 05-12-1998"),             # gạch dài → gạch thường
    ("TOI DI HOC", "TOI DI HOC"),                       # không có số → giữ nguyên
])
def test_normalize_ocr(raw, expected):
    out = normalize_ocr(raw)
    assert out == expected
    assert len(out) == len(raw)  # độ dài không đổi → vị trí start/end vẫn đúng


# ---------------------------------------------------------------- CCCD
@pytest.mark.parametrize("text, expected, score", [
    ("Số / No.: 079203012345", "079203012345", TIGHT),
    ("CCCD: 001198001234", "001198001234", TIGHT),
    ("Số CCCD của tôi là 038198890838 nhé", "038198890838", TIGHT),
    ("No.:075151039465", "075151039465", TIGHT),             # OCR dính "No.:" vào số
    ("căn cước 079 203 012 345", "079 203 012 345", LOOSE),  # OCR tách bằng dấu cách
    ("CCCD: O79203O12345", "O79203O12345", LOOSE),           # OCR nhầm 0 → O
    ("CCCD 0792030l2345", "0792030l2345", LOOSE),            # OCR nhầm 1 → l
    ("CCCD: 999123456789", "999123456789", LOOSE),           # sai mã tỉnh nhưng có từ khóa
])
def test_cccd_positive(text, expected, score):
    assert_found(text, "CCCD", expected, score)


@pytest.mark.parametrize("text", [
    "STK 079203012345",            # có từ khóa tài khoản → ACCOUNT
    "CCCD: 07920301234",           # 11 số
    "CCCD: 0792030123456",         # 13 số
    "0912345678",                  # SĐT
    "Mã giao dịch FT24287ABC123",
    "123456789012",                # sai cấu trúc, không từ khóa
])
def test_cccd_negative(text):
    assert_not_found(text, "CCCD")


# ---------------------------------------------------------------- DOB
@pytest.mark.parametrize("text, expected, score", [
    ("Ngày sinh: 05/12/1998", "05/12/1998", TIGHT),
    ("Ngày sinh / Date of birth: 05-12-1998", "05-12-1998", TIGHT),
    ("Ngày sinh\n01.01.2000", "01.01.2000", TIGHT),
    ("Sinh nhật Lan là ngày 20/11/1995 đó", "20/11/1995", TIGHT),
    ("dob 1/2/1990", "1/2/1990", TIGHT),
    ("Ngày sinh: 05 / 12 / 1998", "05 / 12 / 1998", LOOSE),   # OCR chèn dấu cách
    ("15/08/1985", "15/08/1985", LOOSE),                      # không có từ khóa
])
def test_dob_positive(text, expected, score):
    assert_found(text, "DOB", expected, score)


@pytest.mark.parametrize("text", [
    "Có giá trị đến: 05-12-2040",   # tương lai
    "Có giá trị đến: 05-12-2023",   # từ khóa phủ định
    "Thời gian: 14:32 12/10/2025",
    "Ngày cấp: 01/01/2021",
    "Ngày sinh: 32/01/1990",        # ngày không hợp lệ
    "Ngày sinh: 05/13/1990",        # tháng không hợp lệ
    "Ngày sinh: 12/10",             # thiếu năm
    "Số tiền 1.500.000",
])
def test_dob_negative(text):
    assert_not_found(text, "DOB")


# ---------------------------------------------------------------- PHONE
@pytest.mark.parametrize("text, expected, score", [
    ("0912345678", "0912345678", TIGHT),
    ("SĐT: 0912 345 678", "0912 345 678", TIGHT),
    ("+84 912 345 678", "+84 912 345 678", TIGHT),
    ("0912.345.678", "0912.345.678", TIGHT),
    ("gọi em số 0389123456 nhé", "0389123456", TIGHT),
    ("+84912345678", "+84912345678", TIGHT),
    ("84912345678", "84912345678", LOOSE),        # thiếu dấu +
    ("091 23 45 678", "091 23 45 678", LOOSE),    # chia nhóm lạ
    ("Số của mình là 0902\n821 625 nhé", "0902\n821 625", TIGHT),   # bị ngắt dòng trong bong bóng chat
    ("SĐT: +84\n888 062 855\n22:30", "+84\n888 062 855", TIGHT),    # dòng sau là giờ, không được nuốt
    ("SĐT: (+84) 912 345 678", "(+84) 912 345 678", TIGHT),
    ("0912-345-678", "0912-345-678", TIGHT),
])
def test_phone_positive(text, expected, score):
    assert_found(text, "PHONE", expected, score)


@pytest.mark.parametrize("text", [
    "Hotline 1900 6060",
    "0212345678",          # đầu 02 (máy bàn) — chỉ bắt di động
    "0612345678",          # đầu 06 không tồn tại
    "091234567",           # 9 số
    "09123456789",         # 11 số
    "STK 1234567890",
    "Mã OTP 123456",
])
def test_phone_negative(text):
    assert_not_found(text, "PHONE")


# ---------------------------------------------------------------- EMAIL
@pytest.mark.parametrize("text, expected, score", [
    ("abc@gmail.com", "abc@gmail.com", TIGHT),
    ("Email: nguyen.van.an1990@yahoo.com", "nguyen.van.an1990@yahoo.com", TIGHT),
    ("gửi qua mail dat.dinh@fpt.edu.vn nha", "dat.dinh@fpt.edu.vn", TIGHT),
    ("Idviet1570@yahoo.com", "Idviet1570@yahoo.com", TIGHT),
    ("xuannguyen457(@yahoo.com", "xuannguyen457(@yahoo.com", LOOSE),   # OCR thêm "("
    ("an @ gmail.com", "an @ gmail.com", LOOSE),
    ("nghỉĩa.doan1018@outlook.com", "nghỉĩa.doan1018@outlook.com", LOOSE),  # OCR thêm dấu
])
def test_email_positive(text, expected, score):
    assert_found(text, "EMAIL", expected, score)


@pytest.mark.parametrize("text", [
    "@gmail.com",
    "abc@",
    "abc.gmail.com",
    "user@localhost",
    "Hẹn 9h @ quán cà phê",
    "Giá 50@ cửa hàng",
])
def test_email_negative(text):
    assert_not_found(text, "EMAIL")


# ---------------------------------------------------------------- MONEY
@pytest.mark.parametrize("text, expected, score", [
    ("1.500.000đ", "1.500.000đ", TIGHT),
    ("Số tiền: 2.000.000 VND", "2.000.000 VND", TIGHT),
    ("-11,400,000 VNĐ", "-11,400,000 VNĐ", TIGHT),
    ("945,000 đồng", "945,000 đồng", TIGHT),
    ("61.000d", "61.000d", TIGHT),                    # OCR đọc đ → d
    ("lương được 15 triệu", "15 triệu", TIGHT),
    ("+2,000,000", "+2,000,000", LOOSE),              # thiếu đơn vị
    ("Em chuyển 500k rồi", "500k", LOOSE),
    ("Em chuyển 1tr5 rồi", "1tr5", LOOSE),
    ("Số dư khả dụng 70,000,000 đ", "70,000,000 đ", TIGHT),   # "đ" tách bằng dấu cách
    ("Tổng tiền 250.000 ₫", "250.000 ₫", TIGHT),              # ký hiệu ₫
    ("Số dư VND 1.500.000", "VND 1.500.000", TIGHT),          # đơn vị đứng trước
    ("SD: 2,345,000VND", "2,345,000VND", TIGHT),              # kiểu SMS ngân hàng
])
def test_money_positive(text, expected, score):
    assert_found(text, "MONEY", expected, score)


@pytest.mark.parametrize("text", [
    "Đang giảm giá 50% đó",
    "Phòng 35 tầng 3 nha",
    "Họp lúc 14:32",
    "0912.345.678",
    "05.12.1998",
    "Mã 1.500",          # 1 nhóm nghìn, không dấu, không đơn vị
    "Đi bộ 5km",
])
def test_money_negative(text):
    assert_not_found(text, "MONEY")


# ---------------------------------------------------------------- ACCOUNT
@pytest.mark.parametrize("text, expected, score", [
    ("STK 1025455161", "1025455161", TIGHT),
    ("Số tài khoản: 0071000123456", "0071000123456", TIGHT),
    ("Chuyển khoản vào STK 1025 4551 61 MB Bank", "1025 4551 61", TIGHT),
    ("Tài khoản thanh toán\n4225858605", "4225858605", TIGHT),
    ("Chuyển tiền đến 1773593210", "1773593210", TIGHT),
    ("stk: 19036318273013", "19036318273013", TIGHT),
    ("1773593210", "1773593210", LOOSE),        # không có từ khóa
    ("Chuyển tiền đến 112804822\n20.05.2026", "112804822", TIGHT),     # dòng sau là ngày
    ("Chuyển tiền đến 3580 7424 1\n29-09-2026", "3580 7424 1", TIGHT),
    ("STK 1025455161, chủ TK Lan", "1025455161", TIGHT),      # dấu phẩy dính liền cuối số
])
def test_account_positive(text, expected, score):
    assert_found(text, "ACCOUNT", expected, score)


@pytest.mark.parametrize("text", [
    "STK 1234567",              # 7 số
    "STK 12345678901234567",    # 17 số
    "Hotline 1900 3468",
    "Đơn hàng #DH12345678",
    "Mã GD FT24287123456",
    "STK 12345678.9",           # số thập phân, không phải STK
    "SĐT 0912345678",           # → PHONE
    "CCCD: 079203012345",       # → CCCD
])
def test_account_negative(text):
    assert_not_found(text, "ACCOUNT")
