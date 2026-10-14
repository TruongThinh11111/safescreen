package com.safescreen

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.safescreen.engine.SafeScreenEngine
import com.safescreen.mask.MaskMode
import com.safescreen.mask.masked
import com.safescreen.ocr.TesseractOcr
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
 * Chạy TOÀN BỘ pipeline (OCR → CRF → F → che) trên ảnh mẫu androidTest/assets/ocr/, ghi ra files/pipeline/:
 *   - regions.json: vùng che của từng ảnh → so với SafeScreen()(img) bên Python (risk_mask.py)
 *   - <ảnh>_<chế độ>.png: ảnh đã che theo 3 chế độ, để xem bằng mắt
 *   adb shell run-as com.safescreen tar -cf - files/pipeline | tar -xf -
 */
@RunWith(AndroidJUnit4::class)
class PipelineInstrumentedTest {

    @Test
    fun pipelineAnhMau() {
        val inst = InstrumentationRegistry.getInstrumentation()
        val app = inst.targetContext
        val testAssets = inst.context.assets
        val names = testAssets.list("ocr").orEmpty().filter { it.endsWith(".png") }.sorted()
        assertTrue(names.isNotEmpty())

        val engine = SafeScreenEngine(
            app.assets.open("crf_model.json").bufferedReader().use { it.readText() },
            app.assets.open("config.json").bufferedReader().use { it.readText() },
        )
        val outDir = File(app.filesDir, "pipeline").apply { deleteRecursively(); mkdirs() }
        val report = buildJsonObject {
            TesseractOcr.create(app).use { ocr ->
                for (name in names) {
                    val bmp = testAssets.open("ocr/$name").use { BitmapFactory.decodeStream(it) }
                    val o = ocr.recognize(bmp)
                    val t0 = System.nanoTime()
                    val r = engine.process(o.tokens.map { it.toOcrWord() }, alreadySorted = false)
                    val crfMs = (System.nanoTime() - t0) / 1e6
                    val maskMs = MaskMode.entries.associateWith { mode ->
                        val t1 = System.nanoTime()
                        val out = bmp.masked(r.regions.map { it.bbox }, mode, engine.maskPadPx)
                        val ms = (System.nanoTime() - t1) / 1e6
                        File(outDir, "${name.removeSuffix(".png")}_${mode.configName}.png").outputStream()
                            .use { out.compress(Bitmap.CompressFormat.PNG, 100, it) }
                        ms
                    }
                    Log.i("SafeScreenPipeline", "$name: ${r.regions.size} vùng, OCR ${o.preprocessMs + o.recognizeMs} ms, " +
                        "CRF ${"%.1f".format(crfMs)} ms, che ${maskMs.map { "${it.key.configName} ${"%.1f".format(it.value)} ms" }}")
                    put(name, buildJsonArray {
                        for (g in r.regions) add(buildJsonObject {
                            put("bbox", JsonArray(g.bbox.map { JsonPrimitive(it) }))
                            put("labels", JsonArray(g.labels.map { JsonPrimitive(it) }))
                            put("F", g.f)
                            put("text", g.text)
                        })
                    })
                }
            }
        }
        File(outDir, "regions.json").writeText(report.toString())
    }
}
