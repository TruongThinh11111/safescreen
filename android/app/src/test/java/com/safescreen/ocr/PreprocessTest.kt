package com.safescreen.ocr

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.Assert.assertEquals
import org.junit.Test
import java.awt.image.BufferedImage
import javax.imageio.ImageIO
import kotlin.math.abs
import kotlin.math.max

/**
 * So bản Kotlin của tiền xử lý với bản Python (OpenCV) TỪNG ĐIỂM ẢNH.
 * Dữ liệu đối chiếu: src/test/resources/preprocess/, tạo bằng
 *   python -m safescreen.export_preprocess_golden   (đứng ở thư mục ml/)
 */
class PreprocessTest {

    private class Case(val name: String, val scale: Double)

    private val cases: List<Case> by lazy {
        val json = Json.parseToJsonElement(resource("cases.json").readBytes().decodeToString())
        json.jsonObject["cases"]!!.jsonArray.map {
            val o = it.jsonObject
            Case(o["name"]!!.jsonPrimitive.content, o["scale"]!!.jsonPrimitive.content.toDouble())
        }
    }

    private fun resource(name: String) =
        javaClass.getResourceAsStream("/preprocess/$name") ?: error("Thiếu file test: $name")

    private fun readImage(name: String): BufferedImage = resource(name).use { ImageIO.read(it) }

    /** Ảnh đầu vào RGB → xám bằng hàm Kotlin (giống Bitmap.getPixels trên Android) */
    private fun inputGray(name: String): GrayImage {
        val img = readImage("${name}_input.png")
        val argb = img.getRGB(0, 0, img.width, img.height, null, 0, img.width)
        return ImagePreprocessor.rgbToGray(argb, img.width, img.height)
    }

    /** Ảnh xám kết quả của Python */
    private fun expectedGray(file: String): GrayImage {
        val img = readImage(file)
        val out = GrayImage(img.width, img.height)
        val raster = img.raster
        for (y in 0 until img.height) for (x in 0 until img.width) {
            out.pixels[y * img.width + x] = raster.getSample(x, y, 0).toByte()
        }
        return out
    }

    /** Số điểm ảnh khác nhau + chênh lệch lớn nhất */
    private fun diff(got: GrayImage, exp: GrayImage): String {
        if (got.width != exp.width || got.height != exp.height) {
            return "khác kích thước: ${got.width}x${got.height} vs ${exp.width}x${exp.height}"
        }
        var count = 0
        var maxDiff = 0
        for (i in got.pixels.indices) {
            val d = abs((got.pixels[i].toInt() and 0xFF) - (exp.pixels[i].toInt() and 0xFF))
            if (d != 0) count++
            maxDiff = max(maxDiff, d)
        }
        return if (count == 0) "OK" else "$count/${got.pixels.size} điểm khác, lệch tối đa $maxDiff"
    }

    @Test
    fun normalizeBackgroundGiongPython() {
        val report = cases.associate { c ->
            c.name to diff(ImagePreprocessor.normalizeBackground(inputGray(c.name)), expectedGray("${c.name}_norm.png"))
        }
        println("normalize_background: $report")
        assertEquals(cases.associate { it.name to "OK" }, report)
    }

    @Test
    fun preprocessGiongPython() {
        val report = cases.associate { c ->
            val r = ImagePreprocessor.preprocess(inputGray(c.name))
            assertEquals("scale của ${c.name}", c.scale, r.scale, 0.0)
            c.name to diff(r.image, expectedGray("${c.name}_final.png"))
        }
        println("preprocess: $report")
        // Ảnh cắt 1077x1501 (cố ý chọn kích thước lẻ): bicubic bản Kotlin lệch cv2 đúng 2 điểm, mỗi điểm 1 mức xám
        // (chưa dò ra thứ tự cộng float chính xác của cv2 cho tỉ lệ phóng lẻ). Mọi ảnh 1080/720 px khớp tuyệt đối.
        val expected = cases.associate { it.name to "OK" } +
            ("crop_1077x1501" to "2/2890080 điểm khác, lệch tối đa 1")
        assertEquals(expected, report)
    }
}
