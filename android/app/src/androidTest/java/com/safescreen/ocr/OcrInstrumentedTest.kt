package com.safescreen.ocr

import android.graphics.BitmapFactory
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/**
 * OCR các ảnh mẫu trong androidTest/assets/ocr/ bằng ĐÚNG code của app, ghi kết quả ra
 * files/ocr_android.json để so với bản Python (ml/safescreen/ocr.py):
 *   gradlew :app:connectedDebugAndroidTest
 *   adb shell run-as com.safescreen cat files/ocr_android.json > ocr_android.json
 */
@RunWith(AndroidJUnit4::class)
class OcrInstrumentedTest {

    @Test
    fun ocrAnhMau() {
        val inst = InstrumentationRegistry.getInstrumentation()
        val testAssets = inst.context.assets
        val names = testAssets.list("ocr").orEmpty().filter { it.endsWith(".png") }.sorted()
        assertTrue("Thiếu ảnh mẫu trong androidTest/assets/ocr", names.isNotEmpty())

        val ocr = TesseractOcr.create(inst.targetContext)
        val out = buildJsonObject {
            ocr.use {
                for (name in names) {
                    val bmp = testAssets.open("ocr/$name").use { BitmapFactory.decodeStream(it) }
                    val r = ocr.recognize(bmp)
                    Log.i("SafeScreenOcr", "$name: ${r.tokens.size} từ, tiền xử lý ${r.preprocessMs} ms, Tesseract ${r.recognizeMs} ms")
                    put(name, buildJsonObject {
                        put("preprocess_ms", r.preprocessMs)
                        put("recognize_ms", r.recognizeMs)
                        put("words", buildJsonArray {
                            for (t in r.tokens) add(buildJsonObject {
                                put("text", t.text)
                                put("bbox", JsonArray(t.toOcrWord().bbox.map { JsonPrimitive(it) }))
                                put("conf", t.toOcrWord().conf)
                            })
                        })
                    })
                }
            }
        }
        File(inst.targetContext.filesDir, "ocr_android.json").writeText(out.toString())
    }
}
