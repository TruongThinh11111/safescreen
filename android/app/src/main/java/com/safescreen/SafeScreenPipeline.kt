package com.safescreen

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import com.safescreen.engine.EngineResult
import com.safescreen.engine.SafeScreenEngine
import com.safescreen.mask.MaskMode
import com.safescreen.ocr.OcrResult
import com.safescreen.ocr.TesseractOcr
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext

/**
 * Bộ máy OCR → CRF → F, DÙNG CHUNG cho màn hình chính và service bảo vệ tự động.
 * Nạp 1 lần cho cả tiến trình (Tesseract + 2 mô hình chiếm ~100 MB, nạp 1–3 s), giữ tới khi app bị tắt.
 * Tesseract không chịu được 2 luồng cùng lúc → mọi lần gọi chạy lần lượt qua Mutex.
 */
object SafeScreenPipeline {

    class Output(val ocr: OcrResult, val result: EngineResult, val crfMs: Long, val initMs: Long?)

    private val lock = Mutex()
    private var tesseract: TesseractOcr? = null
    @Volatile private var engine: SafeScreenEngine? = null

    /** Số px nới rộng mỗi vùng che (config mask.pad_px); 4 nếu mô hình chưa nạp */
    val maskPadPx: Int get() = engine?.maskPadPx ?: 4

    /** OCR + CRF + hàm rủi ro F + gộp vùng. onStage: báo bước đang làm (để hiện lên giao diện). */
    suspend fun analyze(context: Context, bitmap: Bitmap, onStage: (String) -> Unit = {}): Output = lock.withLock {
        val app = context.applicationContext
        var initMs: Long? = null
        if (tesseract == null || engine == null) {
            onStage("Đang nạp mô hình (lần đầu)…")
            val t0 = SystemClock.elapsedRealtime()
            withContext(Dispatchers.IO) {
                if (engine == null) engine = SafeScreenEngine(
                    app.assets.open("crf_model.json").bufferedReader().use { it.readText() },
                    app.assets.open("config.json").bufferedReader().use { it.readText() },
                )
                if (tesseract == null) tesseract = TesseractOcr.create(app)
            }
            initMs = SystemClock.elapsedRealtime() - t0
            // Lần đầu dùng app: chế độ che mặc định lấy từ config.json (black_box)
            val settings = AppSettings(app)
            if (settings.maskModeOrNull == null) settings.maskModeOrNull = MaskMode.fromConfig(engine!!.defaultMaskMode)
        }
        onStage("Đang OCR…")
        val ocr = withContext(Dispatchers.Default) { tesseract!!.recognize(bitmap) }
        onStage("Đang nhận diện thông tin nhạy cảm…")
        val t1 = SystemClock.elapsedRealtime()
        val result = withContext(Dispatchers.Default) {
            // Python xếp từ theo hàng trước khi vào CRF → alreadySorted = false để engine tự xếp
            engine!!.process(ocr.tokens.map { it.toOcrWord() }, alreadySorted = false)
        }
        Output(ocr, result, SystemClock.elapsedRealtime() - t1, initMs)
    }
}
