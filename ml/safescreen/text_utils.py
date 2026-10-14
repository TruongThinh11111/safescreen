"""Hàm xử lý chuỗi tiếng Việt dùng chung cho các module."""
import unicodedata


def bo_dau(text: str) -> str:
    """Bỏ dấu tiếng Việt: 'Nguyễn Văn Đức' -> 'Nguyen Van Duc'. Giữ nguyên độ dài nếu đầu vào là NFC."""
    text = text.replace("đ", "d").replace("Đ", "D")
    # NFD tách chữ có dấu thành chữ gốc + dấu; category "Mn" là các ký tự dấu → bỏ đi
    nfd = unicodedata.normalize("NFD", text)
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn")


def fold(text: str) -> str:
    """Bỏ dấu + chữ thường, để so khớp từ khóa không phụ thuộc dấu/hoa thường ('Số Tài Khoản' == 'so tai khoan')."""
    return bo_dau(text).lower()
