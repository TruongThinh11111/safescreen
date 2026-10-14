package com.safescreen.engine

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import java.text.Normalizer
import java.time.LocalDate
import java.util.Locale

/**
 * Golden test chạy trên JVM của máy tính (nhanh, không cần điện thoại):  gradlew :app:testDebugUnitTest
 * Thư mục làm việc của unit test là app/, nên đọc thẳng file trong src/.
 */
class GoldenTest {

    private fun read(path: String) = File(path).readText(Charsets.UTF_8)

    private val engine by lazy {
        SafeScreenEngine(read("src/main/assets/crf_model.json"), read("src/main/assets/config.json"))
    }
    private val golden by lazy { read("src/androidTest/assets/golden_test.json") }

    @Test
    fun golden_khopPython() {
        val report = GoldenVerifier.verify(engine, golden)
        println("[JVM ${System.getProperty("java.version")}] $report")
        assertTrue(report.toString(), report.passed)
    }

    /**
     * Kiểm tra chính bộ so sánh: cố tình sửa 1 feature + nhãn của token #2 trong mẫu đầu tiên của golden.
     * Bộ so sánh phải báo KHÔNG ĐẠT và chỉ đúng token đó, đúng feature đó (nếu không, test "ĐẠT" ở trên vô nghĩa).
     */
    @Test
    fun boSoSanh_batDuocLoiVaChiDungToken() {
        val root = Json.parseToJsonElement(golden).jsonObject
        val samples = root.getValue("samples").jsonArray
        val s0 = samples[0].jsonObject
        val tokens = s0.getValue("tokens").jsonArray
        val t2 = tokens[2].jsonObject
        val badFeatures = JsonObject(t2.getValue("features").jsonObject + ("lower" to JsonPrimitive("xxx")))
        val badToken = JsonObject(t2 + mapOf("features" to badFeatures, "label" to JsonPrimitive("B-PHONE")))
        val badSample = JsonObject(s0 + ("tokens" to JsonArray(tokens.toMutableList().also { it[2] = badToken })))
        val badRoot = JsonObject(root + ("samples" to JsonArray(samples.toMutableList().also { it[0] = badSample })))

        val report = GoldenVerifier.verify(engine, badRoot.toString())
        println(report)
        assertFalse(report.passed)
        val first = report.firstMismatch ?: error("phải có token lệch")
        val id = s0.getValue("id").jsonPrimitive.content
        assertTrue(first, first.startsWith("mẫu $id, token #2 "))
        assertTrue(first, "feature lower: Kotlin=" in first && "Python=\"xxx\"" in first)
        assertTrue(first, "nhãn Kotlin=" in first && "Python=B-PHONE" in first)
    }

    /** [P3] Máy cài tiếng Thổ Nhĩ Kỳ: toLowerCase() mặc định biến "I" → "ı". Engine phải không bị ảnh hưởng. */
    @Test
    fun localeThoNhiKy_khongAnhHuong() {
        val old = Locale.getDefault()
        try {
            Locale.setDefault(Locale.forLanguageTag("tr-TR"))
            assertEquals("ı", "I".lowercase(Locale.getDefault()))   // bẫy có thật
            val report = GoldenVerifier.verify(engine, golden)
            assertTrue(report.toString(), report.passed)
        } finally {
            Locale.setDefault(old)
        }
    }

    /** [P1] Đầu vào dạng NFD (dấu tách rời) phải cho kết quả y hệt NFC, vì process() chuẩn hoá NFC trước. */
    @Test
    fun dauVaoNfd_giongNfc() {
        val words = listOf(
            OcrWord("Họ", listOf(40, 100, 80, 120), 0.95, 0), OcrWord("và", listOf(85, 100, 110, 120), 0.95, 0),
            OcrWord("tên:", listOf(115, 100, 160, 120), 0.95, 0),
            OcrWord("NGUYỄN", listOf(40, 130, 150, 155), 0.9, 1), OcrWord("THỊ", listOf(160, 130, 200, 155), 0.9, 1),
            OcrWord("ĐOAN", listOf(210, 130, 280, 155), 0.9, 1),
        )
        val today = LocalDate.of(2026, 10, 14)
        val nfcOut = engine.process(words, today)
        val nfdOut = engine.process(words.map { it.copy(text = Normalizer.normalize(it.text, Normalizer.Form.NFD)) }, today)
        assertEquals(nfcOut.tokens.map { it.label }, nfdOut.tokens.map { it.label })
        assertEquals(nfcOut.tokens.map { it.f }, nfdOut.tokens.map { it.f })
    }

    /** [P2] đ/Đ không tự tách dấu khi NFD → fold phải thay tay. */
    @Test
    fun fold_xuLyChuD() {
        assertEquals("dang van duc", fold("Đặng Văn Đức"))
        assertEquals("nguyen", fold(Normalizer.normalize("Nguyễn", Normalizer.Form.NFD)))
        assertEquals("d", Normalizer.normalize("đ", Normalizer.Form.NFD).let(::boDau))
    }

    /** [P5] Định nghĩa isupper/istitle của Python. */
    @Test
    fun isupperIstitle_theoPython() {
        assertTrue(pyIsUpper("LTE") && pyIsUpper("NGUYỄN,") && !pyIsUpper("0912") && !pyIsUpper("Nguyễn"))
        assertTrue(pyIsTitle("Nguyễn") && pyIsTitle("Hà-Nội") && pyIsTitle("No.:075") && !pyIsTitle("NGUYỄN"))
        assertTrue(pyIsDigit("0912") && pyIsDigit("²") && !pyIsDigit("") && !pyIsDigit("09a"))
    }
}
