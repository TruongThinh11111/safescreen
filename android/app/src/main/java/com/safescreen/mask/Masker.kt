package com.safescreen.mask

import android.graphics.Bitmap
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sqrt

/** 3 cách che, cùng tên với MASK_MODES trong ml/safescreen/risk_mask.py */
enum class MaskMode(val configName: String, val displayName: String) {
    BLACK_BOX("black_box", "Hộp đen"),
    PIXELATE("pixelate", "Pixelate"),
    BLUR("gaussian_blur", "Blur");

    companion object {
        fun fromConfig(name: String) = entries.firstOrNull { it.configName == name } ?: BLACK_BOX
    }
}

/**
 * Bản Kotlin của mask() trong risk_mask.py. Làm việc trên mảng điểm ảnh ARGB (không dùng API Android)
 * → test được trên máy tính (MaskerTest).
 *
 * Vùng che đã được engine gộp sẵn (box liền nhau cùng dòng → 1 vùng), ở đây chỉ nới rộng `pad` px rồi che.
 * Bảo mật: blur/pixelate với chữ số đôi khi đoán ngược lại được; hộp đen là cách duy nhất xoá hẳn thông tin.
 */
object Masker {

    /** Che trực tiếp trên `px` (ảnh w×h). boxes: mỗi box [x0, y0, x1, y1] theo pixel ảnh. */
    fun maskPixels(px: IntArray, w: Int, h: Int, boxes: List<List<Int>>, mode: MaskMode, pad: Int) {
        for (b in boxes) {
            val x0 = max(0, b[0] - pad)
            val y0 = max(0, b[1] - pad)
            val x1 = min(w, b[2] + pad)
            val y1 = min(h, b[3] + pad)
            if (x1 <= x0 || y1 <= y0) continue
            val bh = y1 - y0
            when (mode) {
                // PIL draw.rectangle tô CẢ cạnh x1, y1 → +1 (vẫn không vượt ra ngoài ảnh)
                MaskMode.BLACK_BOX -> fill(px, w, x0, y0, min(w, x1 + 1), min(h, y1 + 1), 0xFF000000.toInt())
                MaskMode.PIXELATE -> pixelate(px, w, x0, y0, x1, y1, block = max(8, bh / 2))  // ô vuông ≈ nửa chiều cao chữ
                MaskMode.BLUR -> gaussianBlur(px, w, x0, y0, x1, y1, sigma = max(6.0, bh * 0.6))  // bán kính theo cỡ chữ
            }
        }
    }

    private fun fill(px: IntArray, w: Int, x0: Int, y0: Int, x1: Int, y1: Int, color: Int) {
        for (y in y0 until y1) px.fill(color, y * w + x0, y * w + x1)
    }

    /**
     * Chia vùng thành lưới ô ~block×block, mỗi ô tô bằng màu TRUNG BÌNH của ô đó
     * (= thu nhỏ rồi phóng lại kiểu "nearest" như bản Python).
     */
    private fun pixelate(px: IntArray, w: Int, x0: Int, y0: Int, x1: Int, y1: Int, block: Int) {
        val rw = x1 - x0
        val rh = y1 - y0
        val cols = max(1, rw / block)
        val rows = max(1, rh / block)
        for (cy in 0 until rows) {
            val ya = y0 + cy * rh / rows
            val yb = y0 + (cy + 1) * rh / rows
            for (cx in 0 until cols) {
                val xa = x0 + cx * rw / cols
                val xb = x0 + (cx + 1) * rw / cols
                var a = 0L; var r = 0L; var g = 0L; var bl = 0L
                for (y in ya until yb) for (x in xa until xb) {
                    val c = px[y * w + x]
                    a += (c ushr 24); r += (c shr 16) and 0xFF; g += (c shr 8) and 0xFF; bl += c and 0xFF
                }
                val n = ((yb - ya) * (xb - xa)).toLong()
                val avg = ((a / n).toInt() shl 24) or ((r / n).toInt() shl 16) or ((g / n).toInt() shl 8) or (bl / n).toInt()
                fill(px, w, xa, ya, xb, yb, avg)
            }
        }
    }

    /**
     * Làm mờ Gaussian gần đúng bằng 3 lần làm mờ hộp (box blur) theo mỗi chiều — cách PIL cũng dùng.
     * Chỉ làm mờ bên trong vùng; điểm ngoài mép vùng lấy bằng điểm ở mép (giống PIL làm trên ảnh cắt ra).
     */
    private fun gaussianBlur(px: IntArray, w: Int, x0: Int, y0: Int, x1: Int, y1: Int, sigma: Double) {
        val rw = x1 - x0
        val rh = y1 - y0
        // Tách 4 kênh màu của vùng ra mảng riêng
        val ch = Array(4) { IntArray(rw * rh) }
        for (y in 0 until rh) for (x in 0 until rw) {
            val c = px[(y0 + y) * w + x0 + x]
            val i = y * rw + x
            ch[0][i] = c ushr 24; ch[1][i] = (c shr 16) and 0xFF; ch[2][i] = (c shr 8) and 0xFF; ch[3][i] = c and 0xFF
        }
        for (radius in boxRadiiForGauss(sigma, 3)) {
            for (c in ch) {
                boxBlur(c, rw, rh, radius, horizontal = true)
                boxBlur(c, rw, rh, radius, horizontal = false)
            }
        }
        for (y in 0 until rh) for (x in 0 until rw) {
            val i = y * rw + x
            px[(y0 + y) * w + x0 + x] = (ch[0][i] shl 24) or (ch[1][i] shl 16) or (ch[2][i] shl 8) or ch[3][i]
        }
    }

    /** Bán kính n hộp sao cho làm mờ hộp n lần ≈ Gaussian độ lệch chuẩn sigma (công thức chuẩn "boxes for gauss"). */
    private fun boxRadiiForGauss(sigma: Double, n: Int): List<Int> {
        val wIdeal = sqrt(12 * sigma * sigma / n + 1)
        var wl = floor(wIdeal).toInt()
        if (wl % 2 == 0) wl--
        val wu = wl + 2
        val m = ((12 * sigma * sigma - n * wl * wl - 4 * n * wl - 3 * n) / (-4 * wl - 4)).roundToInt()
        return List(n) { (if (it < m) wl else wu) / 2 }  // hộp rộng 2r+1 (số lẻ) → r = rộng / 2
    }

    /** Trung bình trượt cửa sổ 2r+1 theo 1 chiều, điểm ngoài mép lấy bằng điểm ở mép. */
    private fun boxBlur(v: IntArray, w: Int, h: Int, r: Int, horizontal: Boolean) {
        if (r <= 0) return
        val len = if (horizontal) w else h
        val lines = if (horizontal) h else w
        val line = IntArray(len)
        val win = 2 * r + 1
        for (l in 0 until lines) {
            fun idx(k: Int) = if (horizontal) l * w + k else k * w + l
            for (k in 0 until len) line[k] = v[idx(k)]
            var sum = 0L
            for (k in -r..r) sum += line[k.coerceIn(0, len - 1)]
            for (k in 0 until len) {
                v[idx(k)] = ((sum + win / 2) / win).toInt()
                sum += line[(k + r + 1).coerceAtMost(len - 1)] - line[(k - r).coerceAtLeast(0)]
            }
        }
    }
}

/** Trả về ẢNH MỚI đã che (ảnh gốc giữ nguyên). */
fun Bitmap.masked(boxes: List<List<Int>>, mode: MaskMode, pad: Int): Bitmap {
    val px = IntArray(width * height)
    getPixels(px, 0, width, 0, 0, width, height)
    Masker.maskPixels(px, width, height, boxes, mode, pad)
    return Bitmap.createBitmap(px, width, height, Bitmap.Config.ARGB_8888)
}
