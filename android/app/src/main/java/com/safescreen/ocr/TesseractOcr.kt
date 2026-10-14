package com.safescreen.ocr

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import com.googlecode.tesseract.android.TessBaseAPI
import com.safescreen.engine.OcrWord
import java.io.Closeable
import java.io.File

/**
 * 1 từ OCR đọc được. Toạ độ tính trên ẢNH GỐC (đã chia lại cho hệ số phóng của bước tiền xử lý).
 * conf: độ tin cậy của Tesseract, thang 0–100.
 */
data class OcrToken(val text: String, val left: Int, val top: Int, val width: Int, val height: Int, val conf: Float) {
    /** Đổi sang dạng engine dùng (giống JSON của ocr.py): bbox [x0,y0,x1,y1], conf thang 0–1. */
    fun toOcrWord(): OcrWord = OcrWord(
        text = text,
        bbox = listOf(left, top, left + width, top + height),
        // pytesseract (Output.DICT) CẮT conf thành số nguyên trước (88.81 → 88), rồi ocr.py mới chia 100
        // → dữ liệu train toàn là 0.88, 0.96... Làm y hệt để c_ocr của hàm rủi ro không bị lệch.
        conf = conf.toInt() / 100.0,
    )
}

/** Kết quả 1 lần OCR + thời gian từng bước (mili giây). */
class OcrResult(val tokens: List<OcrToken>, val preprocessMs: Long, val recognizeMs: Long)

/**
 * Bọc Tesseract4Android, cấu hình GIỐNG run_ocr() trong ml/safescreen/ocr.py:
 *   lang = "vie+eng", psm = 3 (tự chia khối), oem = mặc định (pytesseract không truyền --oem).
 *
 * KHÔNG an toàn khi gọi từ nhiều luồng cùng lúc → MainViewModel dùng Mutex để gọi lần lượt.
 * Nhớ gọi close() khi không dùng nữa (giải phóng bộ nhớ C++ ~100 MB).
 */
class TesseractOcr private constructor(private val api: TessBaseAPI) : Closeable {

    companion object {
        const val LANG = "vie+eng"
        const val PSM = TessBaseAPI.PageSegMode.PSM_AUTO   // = --psm 3
        const val OEM = TessBaseAPI.OEM_DEFAULT            // = không truyền --oem

        /** Chép tessdata (nếu cần) rồi khởi tạo Tesseract. Chậm (~1–3 s) → gọi trên luồng nền. */
        fun create(context: Context): TesseractOcr {
            val dataPath = TessdataInstaller.install(context)
            val api = TessBaseAPI()
            // dataPath = thư mục CHA của "tessdata/"
            if (!api.init(dataPath.absolutePath, LANG, OEM)) {
                api.recycle()
                throw IllegalStateException("Không khởi tạo được Tesseract với $LANG tại $dataPath")
            }
            api.pageSegMode = PSM
            return TesseractOcr(api)
        }
    }

    /** OCR 1 ảnh → danh sách từ (theo thứ tự Tesseract trả về; engine tự xếp lại theo hàng). */
    fun recognize(bitmap: Bitmap): OcrResult {
        val t0 = SystemClock.elapsedRealtime()

        // 1. Tiền xử lý y hệt Python: RGB → xám → chuẩn hoá nền → phóng về rộng 1440 px
        val w = bitmap.width
        val h = bitmap.height
        val argb = IntArray(w * h)
        bitmap.getPixels(argb, 0, w, 0, 0, w, h)
        val pre = ImagePreprocessor.preprocess(ImagePreprocessor.rgbToGray(argb, w, h))
        val img = pre.image
        val t1 = SystemClock.elapsedRealtime()

        // 2. Đưa ảnh xám (1 byte/điểm ảnh) cho Tesseract và nhận dạng
        api.setImage(img.pixels, img.width, img.height, 1, img.width)
        api.getUTF8Text()  // chạy nhận dạng; phải gọi trước getResultIterator()

        // 3. Duyệt từng TỪ (RIL_WORD) = các dòng level 5 trong image_to_data của pytesseract
        val tokens = ArrayList<OcrToken>()
        val level = TessBaseAPI.PageIteratorLevel.RIL_WORD
        val it = api.resultIterator
        if (it != null) {
            try {
                it.begin()
                do {
                    val text = it.getUTF8Text(level)?.trim().orEmpty()
                    val conf = it.confidence(level)
                    if (text.isEmpty() || conf < 0) continue  // giống Python: bỏ chữ rỗng và conf = -1
                    val r = it.getBoundingRect(level)
                    // Đổi toạ độ về ảnh gốc giống Python: round(x / s); Math.rint = round() của Python (về số chẵn)
                    val x0 = Math.rint(r.left / pre.scale).toInt()
                    val y0 = Math.rint(r.top / pre.scale).toInt()
                    val x1 = Math.rint(r.right / pre.scale).toInt()
                    val y1 = Math.rint(r.bottom / pre.scale).toInt()
                    tokens += OcrToken(text, x0, y0, x1 - x0, y1 - y0, conf)
                } while (it.next(level))
            } finally {
                it.delete()
            }
        }
        api.clear()  // xoá ảnh + kết quả cũ, giữ nguyên mô hình đã nạp
        val t2 = SystemClock.elapsedRealtime()
        return OcrResult(tokens, preprocessMs = t1 - t0, recognizeMs = t2 - t1)
    }

    override fun close() = api.recycle()
}

/**
 * Tesseract chỉ đọc được file thật trên bộ nhớ, không đọc được thẳng từ assets trong APK.
 * → Chép assets/tessdata/\*.traineddata vào filesDir/tessdata ở lần chạy đầu
 *   (và chép lại sau mỗi lần cài bản mới, phòng khi mô hình được thay).
 */
object TessdataInstaller {

    private const val ASSET_DIR = "tessdata"

    /** Trả về thư mục cha của tessdata/ (thứ TessBaseAPI.init cần). */
    fun install(context: Context): File {
        val root = context.filesDir
        val dir = File(root, ASSET_DIR).apply { mkdirs() }
        val names = context.assets.list(ASSET_DIR).orEmpty().filter { it.endsWith(".traineddata") }
        check(names.isNotEmpty()) { "Thiếu assets/$ASSET_DIR/*.traineddata" }

        // Mốc "đã chép cho bản cài này": thời điểm cài/cập nhật app
        @Suppress("DEPRECATION")
        val stamp = context.packageManager.getPackageInfo(context.packageName, 0).lastUpdateTime.toString()
        val marker = File(dir, ".installed")
        val upToDate = marker.isFile && marker.readText() == stamp && names.all { File(dir, it).isFile }
        if (upToDate) return root

        for (name in names) {
            // Chép ra file tạm rồi đổi tên: nếu app bị tắt giữa chừng sẽ không để lại file hỏng
            val tmp = File(dir, "$name.tmp")
            context.assets.open("$ASSET_DIR/$name").use { input -> tmp.outputStream().use { input.copyTo(it) } }
            val target = File(dir, name)
            target.delete()
            check(tmp.renameTo(target)) { "Không đổi tên được $tmp" }
        }
        marker.writeText(stamp)
        return root
    }
}
