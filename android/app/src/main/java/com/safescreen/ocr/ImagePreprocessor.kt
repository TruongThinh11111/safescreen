package com.safescreen.ocr

import kotlin.math.abs
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min

/** Ảnh xám 8-bit: mỗi điểm ảnh 1 byte (0 = đen, 255 = trắng), lưu lần lượt từng hàng. */
class GrayImage(val width: Int, val height: Int, val pixels: ByteArray = ByteArray(width * height)) {
    operator fun get(x: Int, y: Int): Int = pixels[y * width + x].toInt() and 0xFF
}

/**
 * Bản Kotlin của preprocess() trong ml/safescreen/ocr.py.
 *
 * Không dùng API Android → chạy được trong unit test trên máy tính (xem PreprocessTest).
 * Mỗi hàm bắt chước ĐÚNG phép OpenCV tương ứng, kể cả cách làm tròn, để ảnh đưa vào Tesseract
 * giống từng điểm ảnh với bản Python → OCR trên điện thoại cho kết quả như lúc train.
 */
object ImagePreprocessor {

    const val TARGET_WIDTH = 1440   // = target_width bên Python
    private const val BLOCK = 8     // = block trong normalize_background

    /** image: ảnh đưa cho Tesseract; scale: hệ số phóng (chia toạ độ cho scale để về ảnh gốc). */
    class Result(val image: GrayImage, val scale: Double)

    /** = preprocess(img, target_width=1440, invert_dark=True) */
    fun preprocess(gray: GrayImage, targetWidth: Int = TARGET_WIDTH, invertDark: Boolean = true): Result {
        val norm = if (invertDark) normalizeBackground(gray) else gray
        val scale = targetWidth.toDouble() / norm.width
        return if (abs(scale - 1) > 0.05) Result(resizeCubic(norm, scale), scale) else Result(norm, 1.0)
    }

    /**
     * = cv2.cvtColor(RGB2GRAY): hệ số nguyên 15 bit (0.299/0.587/0.114 × 32768).
     * Đã thử với cả 16,7 triệu màu: khớp cv2 5.0 tuyệt đối (bản 14 bit lệch 43.864 màu).
     */
    fun rgbToGray(argb: IntArray, width: Int, height: Int): GrayImage {
        val out = GrayImage(width, height)
        for (i in 0 until width * height) {
            val c = argb[i]
            val r = (c shr 16) and 0xFF
            val g = (c shr 8) and 0xFF
            val b = c and 0xFF
            out.pixels[i] = ((r * 9798 + g * 19235 + b * 3735 + (1 shl 14)) shr 15).toByte()
        }
        return out
    }

    /** = normalize_background(gray, block=8): đưa mọi vùng về "chữ đen trên nền trắng". */
    fun normalizeBackground(gray: GrayImage, block: Int = BLOCK): GrayImage {
        val w = gray.width
        val h = gray.height
        val sw = max(1, w / block)
        val sh = max(1, h / block)

        val bg = medianBlur(resizeArea(gray, sw, sh), 7)                                   // màu nền
        val hi = morph(resizeArea(morph(gray, block, isMax = true), sw, sh), 7, isMax = true)    // sáng nhất quanh đó
        val lo = morph(resizeArea(morph(gray, block, isMax = false), sw, sh), 7, isMax = false)  // tối nhất quanh đó

        // grow() = phóng ảnh nhỏ về kích thước gốc bằng INTER_NEAREST
        val xMap = nearestIndex(sw, w)
        val yMap = nearestIndex(sh, h)
        val out = GrayImage(w, h)
        for (y in 0 until h) {
            val rowSmall = yMap[y] * sw
            val row = y * w
            for (x in 0 until w) {
                val s = rowSmall + xMap[x]
                val b = bg.pixels[s].toInt() and 0xFF
                val lightText = ((hi.pixels[s].toInt() and 0xFF) - b) > (b - (lo.pixels[s].toInt() and 0xFF))
                var g = gray.pixels[row + x].toInt() and 0xFF
                var bgv = b
                if (lightText) {  // chữ sáng hơn nền → đảo màu
                    g = 255 - g
                    bgv = 255 - bgv
                }
                // numpy làm bằng float32 rồi astype(uint8) = cắt phần thập phân
                val v = (g.toFloat() * 255f) / max(bgv.toFloat(), 1f)
                out.pixels[row + x] = v.coerceIn(0f, 255f).toInt().toByte()
            }
        }
        return out
    }

    // =================================================================
    // Các phép OpenCV viết lại
    // =================================================================

    /** cv2.dilate (isMax) / cv2.erode với kernel vuông k×k, tâm ở k/2; điểm ngoài ảnh bị bỏ qua. */
    fun morph(src: GrayImage, k: Int, isMax: Boolean): GrayImage {
        val w = src.width
        val h = src.height
        val from = -(k / 2)
        val to = k - 1 - k / 2
        // Lấy max/min theo hàng ngang trước, rồi theo cột dọc (kết quả y hệt làm cả ô vuông một lúc)
        val tmp = IntArray(w * h)
        for (y in 0 until h) {
            val row = y * w
            for (x in 0 until w) {
                var m = if (isMax) 0 else 255
                for (xx in max(0, x + from)..min(w - 1, x + to)) {
                    val v = src.pixels[row + xx].toInt() and 0xFF
                    m = if (isMax) max(m, v) else min(m, v)
                }
                tmp[row + x] = m
            }
        }
        val out = GrayImage(w, h)
        for (y in 0 until h) {
            for (x in 0 until w) {
                var m = if (isMax) 0 else 255
                for (yy in max(0, y + from)..min(h - 1, y + to)) {
                    val v = tmp[yy * w + x]
                    m = if (isMax) max(m, v) else min(m, v)
                }
                out.pixels[y * w + x] = m.toByte()
            }
        }
        return out
    }

    /** cv2.medianBlur(src, k): trung vị của ô k×k, điểm ngoài ảnh lấy bằng điểm ở mép (replicate). */
    fun medianBlur(src: GrayImage, k: Int): GrayImage {
        val w = src.width
        val h = src.height
        val r = k / 2
        val window = IntArray(k * k)
        val out = GrayImage(w, h)
        for (y in 0 until h) {
            for (x in 0 until w) {
                var n = 0
                for (dy in -r..r) {
                    val yy = (y + dy).coerceIn(0, h - 1)
                    for (dx in -r..r) window[n++] = src[(x + dx).coerceIn(0, w - 1), yy]
                }
                window.sort()
                out.pixels[y * w + x] = window[window.size / 2].toByte()
            }
        }
        return out
    }

    /** Bảng chỉ số của cv2.resize(INTER_NEAREST): điểm đích x lấy điểm nguồn floor(x * src/dst). */
    private fun nearestIndex(srcSize: Int, dstSize: Int): IntArray {
        val inv = 1.0 / (dstSize.toDouble() / srcSize)
        return IntArray(dstSize) { min(floor(it * inv).toInt(), srcSize - 1) }
    }

    /** cv2.resize(INTER_AREA) khi thu nhỏ: mỗi điểm đích = trung bình vùng ảnh nguồn nó phủ lên. */
    fun resizeArea(src: GrayImage, dw: Int, dh: Int): GrayImage {
        val scaleX = 1.0 / (dw.toDouble() / src.width)
        val scaleY = 1.0 / (dh.toDouble() / src.height)
        val ix = Math.rint(scaleX).toInt()
        val iy = Math.rint(scaleY).toInt()
        val eps = Math.ulp(1.0)  // DBL_EPSILON
        return if (abs(scaleX - ix) < eps && abs(scaleY - iy) < eps) resizeAreaFast(src, dw, dh, ix, iy)
        else resizeAreaGeneral(src, dw, dh, scaleX, scaleY)
    }

    /** Hệ số thu nhỏ nguyên (vd 1080 → 135 = đúng 8 lần): cộng ô ix×iy rồi nhân 1/diện tích, làm tròn. */
    private fun resizeAreaFast(src: GrayImage, dw: Int, dh: Int, ix: Int, iy: Int): GrayImage {
        val out = GrayImage(dw, dh)
        val inv = 1f / (ix * iy)
        for (dy in 0 until dh) {
            for (dx in 0 until dw) {
                var sum = 0
                for (yy in dy * iy until dy * iy + iy) {
                    val row = yy * src.width
                    for (xx in dx * ix until dx * ix + ix) sum += src.pixels[row + xx].toInt() and 0xFF
                }
                // Math.rint = làm tròn "về số chẵn" khi đúng .5, giống OpenCV (KHÔNG dùng roundToInt)
                out.pixels[dy * dw + dx] = Math.rint((sum * inv).toDouble()).toInt().coerceIn(0, 255).toByte()
            }
        }
        return out
    }

    /** Bảng trọng số (computeResizeAreaTab của OpenCV): điểm nguồn si góp alpha vào điểm đích di. */
    private class AreaTab(val di: IntArray, val si: IntArray, val alpha: FloatArray, val size: Int)

    private fun areaTab(ssize: Int, dsize: Int, scale: Double): AreaTab {
        val di = IntArray(ssize * 2)
        val si = IntArray(ssize * 2)
        val alpha = FloatArray(ssize * 2)
        var k = 0
        for (dx in 0 until dsize) {
            val fsx1 = dx * scale
            val fsx2 = fsx1 + scale
            val cellWidth = min(scale, ssize - fsx1)
            var sx2 = floor(fsx2).toInt()
            sx2 = min(sx2, ssize - 1)
            val sx1 = min(Math.ceil(fsx1).toInt(), sx2)
            if (sx1 - fsx1 > 1e-3) {
                di[k] = dx; si[k] = sx1 - 1; alpha[k++] = ((sx1 - fsx1) / cellWidth).toFloat()
            }
            for (sx in sx1 until sx2) {
                di[k] = dx; si[k] = sx; alpha[k++] = (1.0 / cellWidth).toFloat()
            }
            if (fsx2 - sx2 > 1e-3) {
                di[k] = dx; si[k] = sx2; alpha[k++] = (min(min(fsx2 - sx2, 1.0), cellWidth) / cellWidth).toFloat()
            }
        }
        return AreaTab(di, si, alpha, k)
    }

    /** Hệ số thu nhỏ lẻ (vd 2340 → 292): cộng có trọng số bằng float, đúng thứ tự cộng của OpenCV. */
    private fun resizeAreaGeneral(src: GrayImage, dw: Int, dh: Int, scaleX: Double, scaleY: Double): GrayImage {
        val xtab = areaTab(src.width, dw, scaleX)
        val ytab = areaTab(src.height, dh, scaleY)
        val out = GrayImage(dw, dh)
        val buf = FloatArray(dw)
        val sum = FloatArray(dw)

        fun writeRow(dy: Int) {
            for (dx in 0 until dw) out.pixels[dy * dw + dx] = Math.rint(sum[dx].toDouble()).toInt().coerceIn(0, 255).toByte()
        }

        var prevDy = ytab.di[0]
        for (j in 0 until ytab.size) {
            val beta = ytab.alpha[j]
            val dy = ytab.di[j]
            val row = ytab.si[j] * src.width
            buf.fill(0f)
            for (k in 0 until xtab.size) {
                buf[xtab.di[k]] += (src.pixels[row + xtab.si[k]].toInt() and 0xFF).toFloat() * xtab.alpha[k]
            }
            if (dy != prevDy) {
                writeRow(prevDy)
                for (dx in 0 until dw) sum[dx] = beta * buf[dx]
                prevDy = dy
            } else {
                for (dx in 0 until dw) sum[dx] += beta * buf[dx]
            }
        }
        writeRow(prevDy)
        return out
    }

    /** 4 hệ số nội suy bicubic của OpenCV (A = -0.75), tính bằng float như bản C++. */
    private fun cubicCoeffs(x: Float, c: FloatArray) {
        val a = -0.75f
        c[0] = ((a * (x + 1) - 5 * a) * (x + 1) + 8 * a) * (x + 1) - 4 * a
        c[1] = ((a + 2) * x - (a + 3)) * x * x + 1
        c[2] = ((a + 2) * (1 - x) - (a + 3)) * (1 - x) * (1 - x) + 1
        c[3] = 1f - c[0] - c[1] - c[2]
    }

    /**
     * cv2.resize(src, None, fx=f, fy=f, interpolation=INTER_CUBIC) cho ảnh 8-bit.
     * Hai lượt: ngang rồi dọc, mỗi điểm đích = tổng 4 điểm nguồn × trọng số. Điểm ngoài ảnh lấy bằng điểm ở mép.
     *
     * Cách tính được dò bằng thực nghiệm với cv2 5.0 (xem PreprocessTest): vị trí điểm nguồn tính bằng
     * double, trọng số và phép cộng bằng float, kết quả giữa 2 lượt KHÔNG làm tròn.
     * Khớp tuyệt đối mọi ảnh 1080/720 px; ảnh kích thước lẻ có thể lệch 1 mức xám ở vài điểm.
     */
    fun resizeCubic(src: GrayImage, f: Double): GrayImage {
        val sw = src.width
        val sh = src.height
        val dw = Math.rint(sw * f).toInt()
        val dh = Math.rint(sh * f).toInt()
        val scale = 1.0 / f
        val c = FloatArray(4)

        // Bảng cho từng cột (hoặc hàng) đích: 4 vị trí điểm nguồn + 4 trọng số
        fun table(size: Int, srcSize: Int): Pair<IntArray, FloatArray> {
            val idx = IntArray(size * 4)
            val coef = FloatArray(size * 4)
            for (d in 0 until size) {
                val pos = (d + 0.5) * scale - 0.5
                val s = floor(pos).toInt()
                cubicCoeffs((pos - s).toFloat(), c)
                for (k in 0 until 4) {
                    idx[d * 4 + k] = (s - 1 + k).coerceIn(0, srcSize - 1)
                    coef[d * 4 + k] = c[k]
                }
            }
            return idx to coef
        }
        val (xi, xc) = table(dw, sw)
        val (yi, yc) = table(dh, sh)

        // Lượt ngang cho mọi hàng nguồn (giữ nguyên float, chưa làm tròn)
        val horiz = FloatArray(sh * dw)
        for (y in 0 until sh) {
            val row = y * sw
            for (dx in 0 until dw) {
                val k = dx * 4
                fun px(j: Int) = (src.pixels[row + xi[k + j]].toInt() and 0xFF).toFloat()
                horiz[y * dw + dx] = px(0) * xc[k] + (px(1) * xc[k + 1] + (px(2) * xc[k + 2] + px(3) * xc[k + 3]))
            }
        }

        // Lượt dọc → làm tròn (về số chẵn khi đúng .5) và cắt về 0..255
        val out = GrayImage(dw, dh)
        for (dy in 0 until dh) {
            val k = dy * 4
            val r0 = yi[k] * dw; val r1 = yi[k + 1] * dw; val r2 = yi[k + 2] * dw; val r3 = yi[k + 3] * dw
            val b0 = yc[k]; val b1 = yc[k + 1]; val b2 = yc[k + 2]; val b3 = yc[k + 3]
            for (dx in 0 until dw) {
                val v = horiz[r0 + dx] * b0 + (horiz[r1 + dx] * b1 + (horiz[r2 + dx] * b2 + horiz[r3 + dx] * b3))
                out.pixels[dy * dw + dx] = Math.rint(v.toDouble()).toInt().coerceIn(0, 255).toByte()
            }
        }
        return out
    }
}
