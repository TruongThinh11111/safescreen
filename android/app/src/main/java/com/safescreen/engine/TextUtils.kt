package com.safescreen.engine

import java.text.Normalizer
import kotlin.math.exp

/*
 * Hàm xử lý chuỗi — port 1–1 từ ml/safescreen/crf_pure.py.
 * Các ghi chú [P1]..[P10] trùng với danh sách "PORT:" trong crf_pure.py.
 *
 * Mọi hàm duyệt theo CODE POINT (không theo Char), để khớp Python kể cả khi gặp ký tự ngoài BMP (emoji). [P4]
 */

/** [P1] Chuẩn hoá NFC — gọi cho MỌI chuỗi từ OCR trước khi xử lý. Đặc trưng được train trên chuỗi NFC. */
fun nfc(s: String): String = Normalizer.normalize(s, Normalizer.Form.NFC)

/**
 * [P2] Bỏ dấu: 'Nguyễn Văn Đức' → 'Nguyen Van Duc'.
 * 'đ'/'Đ' là chữ cái riêng, NFD KHÔNG tách thành d + dấu → phải thay tay TRƯỚC khi NFD.
 */
fun boDau(s: String): String {
    val nfd = Normalizer.normalize(s.replace('đ', 'd').replace('Đ', 'D'), Normalizer.Form.NFD)
    val sb = StringBuilder(nfd.length)
    nfd.forEachCodePoint { cp ->
        if (Character.getType(cp) != Character.NON_SPACING_MARK.toInt()) sb.appendCodePoint(cp)
    }
    return sb.toString()
}

/** [P3] Bỏ dấu rồi viết thường. lowercase() dùng Locale.ROOT — KHÔNG dùng toLowerCase() theo locale máy. */
fun fold(s: String): String = boDau(s).lowercase()

/** = Python str.isupper(): có ≥1 chữ có hoa/thường, không có chữ thường hay chữ dạng tiêu đề (Lt). [P5] */
fun pyIsUpper(s: String): Boolean {
    var hasCased = false
    s.forEachCodePoint { cp ->
        if (Character.isLowerCase(cp) || Character.isTitleCase(cp)) return false
        if (Character.isUpperCase(cp)) hasCased = true
    }
    return hasCased
}

/**
 * = Python str.istitle(): chữ HOA chỉ đứng sau ký tự không có hoa/thường, chữ thường chỉ đứng sau chữ
 * có hoa/thường; phải có ≥1 chữ. 'Nguyễn' true, 'No.:075' true, 'NGUYỄN' false. [P5]
 */
fun pyIsTitle(s: String): Boolean {
    var cased = false
    var prevCased = false
    s.forEachCodePoint { cp ->
        when {
            Character.isUpperCase(cp) || Character.isTitleCase(cp) -> {
                if (prevCased) return false
                prevCased = true; cased = true
            }
            Character.isLowerCase(cp) -> {
                if (!prevCased) return false
                prevCased = true; cased = true
            }
            else -> prevCased = false
        }
    }
    return cased
}

/**
 * = Python ch.isdigit() cho 1 code point. [P5] Python coi cả '²', '①' là digit; Character.isDigit chỉ nhận
 * nhóm Nd → bổ sung nhóm No có giá trị 0–9.
 */
fun pyIsDigitChar(cp: Int): Boolean =
    Character.isDigit(cp) ||
        (Character.getType(cp) == Character.OTHER_NUMBER.toInt() && Character.getNumericValue(cp) in 0..9)

/** = Python str.isdigit(): khác rỗng và mọi ký tự là chữ số. */
fun pyIsDigit(s: String): Boolean {
    if (s.isEmpty()) return false
    s.forEachCodePoint { if (!pyIsDigitChar(it)) return false }
    return true
}

fun hasDigit(s: String): Boolean {
    s.forEachCodePoint { if (pyIsDigitChar(it)) return true }
    return false
}

/** 'Nguyễn' → 'Xxxxxx', '0912' → 'dddd'. Ký tự khác giữ nguyên. */
fun wordShape(w: String): String {
    val sb = StringBuilder(w.length)
    w.forEachCodePoint { cp ->
        when {
            Character.isUpperCase(cp) -> sb.append('X')
            Character.isLowerCase(cp) -> sb.append('x')
            pyIsDigitChar(cp) -> sb.append('d')
            else -> sb.appendCodePoint(cp)
        }
    }
    return sb.toString()
}

/** [P4] Độ dài và cắt chuỗi theo code point, như Python len(s), s[:n], s[-n:]. */
fun cpLen(s: String): Int = s.codePointCount(0, s.length)

fun cpPrefix(s: String, n: Int): String =
    if (n >= cpLen(s)) s else s.substring(0, s.offsetByCodePoints(0, n))

fun cpSuffix(s: String, n: Int): String {
    val len = cpLen(s)
    return if (n >= len) s else s.substring(s.offsetByCodePoints(0, len - n))
}

/** [P10] Sigmoid ổn định số: không tính exp() của số dương lớn. */
fun sigmoid(z: Double): Double =
    if (z >= 0) 1.0 / (1.0 + exp(-z)) else exp(z).let { it / (1.0 + it) }

internal inline fun String.forEachCodePoint(action: (Int) -> Unit) {
    var i = 0
    while (i < length) {
        val cp = codePointAt(i)
        action(cp)
        i += Character.charCount(cp)
    }
}
