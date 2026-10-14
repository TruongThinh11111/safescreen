package com.safescreen

import android.app.Application
import android.content.Intent
import android.content.IntentSender
import android.graphics.Bitmap
import android.net.Uri
import android.os.Build
import android.os.SystemClock
import android.provider.MediaStore
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.safescreen.auto.AutoActions
import com.safescreen.auto.AutoMaskStore
import com.safescreen.auto.AutoProtectService
import com.safescreen.auto.AutoRecord
import com.safescreen.auto.Notifications
import com.safescreen.auto.PermissionStatus
import com.safescreen.engine.EngineResult
import com.safescreen.mask.GallerySaver
import com.safescreen.mask.MaskMode
import com.safescreen.mask.masked
import com.safescreen.ocr.OcrResult
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.IOException
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/** Thời gian từng bước (mili giây). Bước nào chưa chạy thì null. */
data class Timings(
    val ocrPreprocessMs: Long? = null,
    val ocrTesseractMs: Long? = null,
    val crfMs: Long? = null,
    val maskMs: Long? = null,
    val initMs: Long? = null,  // nạp Tesseract + mô hình CRF (chỉ lần đầu)
)

enum class Tab { MANUAL, AUTO }

/**
 * Trạng thái màn hình chính:
 *   - Tab "Che thủ công": chọn ảnh → OCR → CRF + F → che → lưu
 *   - Tab "Bảo vệ tự động": bật/tắt service, trạng thái quyền, danh sách ảnh đã che, khôi phục ảnh gốc
 * ViewModel sống qua lần xoay màn hình → không phải chạy lại pipeline.
 */
class MainViewModel(app: Application) : AndroidViewModel(app) {

    private val settings = AppSettings(app)
    private val store = AutoMaskStore(app)

    var tab by mutableStateOf(Tab.MANUAL)

    // =================================================================
    // Che thủ công
    // =================================================================
    var original by mutableStateOf<Bitmap?>(null)
        private set
    var ocr by mutableStateOf<OcrResult?>(null)
        private set
    var result by mutableStateOf<EngineResult?>(null)
        private set
    var masked by mutableStateOf<Bitmap?>(null)
        private set
    var maskMode by mutableStateOf(settings.maskMode)
        private set
    var timings by mutableStateOf(Timings())
        private set
    var busy by mutableStateOf<String?>(null)
        private set
    var error by mutableStateOf<String?>(null)
        private set
    var saveMessage by mutableStateOf<String?>(null)
        private set

    private var job: Job? = null

    fun onImagePicked(uri: Uri) {
        job?.cancel()  // đang xử lý ảnh trước mà chọn ảnh mới → bỏ ảnh trước
        job = viewModelScope.launch {
            original = null; ocr = null; result = null; masked = null
            timings = Timings(); error = null; saveMessage = null
            try {
                busy = "Đang đọc ảnh…"
                val bmp = withContext(Dispatchers.IO) { ImageLoader.decode(getApplication(), uri) }
                original = bmp

                // OCR → CRF + hàm rủi ro F + quyết định che + gộp vùng (dùng chung bộ máy với service)
                val out = SafeScreenPipeline.analyze(getApplication(), bmp) { busy = it }
                ocr = out.ocr
                result = out.result
                timings = Timings(out.ocr.preprocessMs, out.ocr.recognizeMs, out.crfMs, initMs = out.initMs)
                maskMode = settings.maskMode  // lần đầu: vừa được lấy từ config.json

                busy = "Đang che…"
                remask(bmp)
                busy = null
            } catch (e: CancellationException) {
                throw e  // phải đứng TRƯỚC IllegalStateException (CancellationException là lớp con của nó)
            } catch (e: IOException) {
                fail("Không đọc được ảnh: ${e.message}")
            } catch (e: SecurityException) {
                fail("Mất quyền truy cập ảnh, hãy chọn lại.")
            } catch (e: IllegalStateException) {
                fail("Lỗi xử lý: ${e.message}")
            }
        }
    }

    /** Đổi cách che: chỉ chạy lại bước che. Chế độ này cũng được bảo vệ tự động dùng. */
    fun changeMaskMode(mode: MaskMode) {
        if (mode == maskMode) return
        maskMode = mode
        settings.maskModeOrNull = mode
        val bmp = original ?: return
        if (result == null) return  // pipeline chưa chạy xong; lúc xong sẽ dùng chế độ mới
        viewModelScope.launch { remask(bmp) }
    }

    private suspend fun remask(bmp: Bitmap) {
        val r = result ?: return
        val mode = maskMode
        val t0 = SystemClock.elapsedRealtime()
        val out = withContext(Dispatchers.Default) { bmp.masked(r.regions.map { it.bbox }, mode, SafeScreenPipeline.maskPadPx) }
        if (mode != maskMode || original !== bmp) return  // người dùng đã đổi chế độ / ảnh trong lúc chờ
        masked = out
        saveMessage = null
        timings = timings.copy(maskMs = SystemClock.elapsedRealtime() - t0)
    }

    /** Lưu ảnh ĐÃ CHE (không có khung/nhãn demo) vào thư viện ảnh. */
    fun saveMasked() {
        val bmp = masked ?: return
        viewModelScope.launch {
            saveMessage = try {
                val name = "safescreen_" + SimpleDateFormat("yyyyMMdd_HHmmss", Locale.ROOT).format(Date()) + ".png"
                withContext(Dispatchers.IO) { GallerySaver.savePng(getApplication(), bmp, name) }
                "Đã lưu vào Ảnh › ${GallerySaver.ALBUM} ($name)"
            } catch (e: IOException) {
                "Lưu thất bại: ${e.message}"
            } catch (e: SecurityException) {
                "Lưu thất bại: chưa có quyền ghi bộ nhớ"
            }
        }
    }

    private fun fail(message: String) {
        error = message
        busy = null
    }

    // =================================================================
    // Bảo vệ tự động
    // =================================================================
    var autoProtect by mutableStateOf(settings.autoProtect)
        private set
    var permissions by mutableStateOf(PermissionStatus.read(app))
        private set
    var records by mutableStateOf<List<AutoRecord>>(emptyList())
        private set
    var selected by mutableStateOf<AutoRecord?>(null)
        private set
    var selectedImage by mutableStateOf<Bitmap?>(null)
        private set
    var autoMessage by mutableStateOf<String?>(null)
    /** Khác null → màn hình chính phải mở hộp thoại hệ thống xin quyền sửa ảnh (rồi gọi onWriteRequestResult) */
    var writeRequest by mutableStateOf<IntentSender?>(null)
        private set
    private var pendingRestore: AutoRecord? = null

    /** Gọi mỗi khi quay lại app (ví dụ từ màn hình Cài đặt): đọc lại quyền và danh sách kết quả */
    fun refreshAuto() {
        permissions = PermissionStatus.read(getApplication())
        // Đang để Bật mà service không chạy (vừa cấp quyền, hoặc bị hệ thống tắt) → bật lại
        if (settings.autoProtect && !AutoProtectService.isRunning) AutoProtectService.start(getApplication())
        viewModelScope.launch {
            records = withContext(Dispatchers.IO) { store.load() }
            selected?.let { s -> selected = records.firstOrNull { it.mediaId == s.mediaId } }
        }
    }

    fun enableAutoProtect(enabled: Boolean) {
        AutoActions.setEnabled(getApplication(), enabled)
        autoProtect = enabled
        autoMessage = if (enabled) "Đã bật. Hãy chụp thử một màn hình có thông tin cá nhân." else null
    }

    /** Mở từ thông báo: chuyển sang tab tự động, chọn đúng ảnh, bấm "Khôi phục" thì khôi phục luôn */
    fun handleIntent(intent: Intent?) {
        val id = intent?.getLongExtra(Notifications.EXTRA_RECORD_ID, -1) ?: -1
        if (id < 0) return
        tab = Tab.AUTO
        val restore = intent!!.getBooleanExtra(Notifications.EXTRA_RESTORE, false)
        viewModelScope.launch {
            val r = withContext(Dispatchers.IO) { store.get(id) } ?: run { autoMessage = "Không tìm thấy kết quả"; return@launch }
            select(r)
            if (restore) restore(r)
        }
    }

    fun select(r: AutoRecord?) {
        selected = r
        selectedImage = null
        if (r == null || r.restored) return
        viewModelScope.launch {
            selectedImage = try {
                withContext(Dispatchers.IO) { ImageLoader.decode(getApplication(), r.displayUri) }
            } catch (e: Exception) {
                if (e is CancellationException) throw e
                autoMessage = "Không mở được ảnh (đã bị xoá?)"
                null
            }
        }
    }

    fun restore(r: AutoRecord) {
        viewModelScope.launch {
            val res = withContext(Dispatchers.IO) { AutoActions.restore(getApplication(), r) }
            when (res) {
                AutoActions.RestoreResult.RESTORED -> {
                    autoMessage = "Đã khôi phục ảnh gốc: ${r.displayName}"
                    selectedImage = null
                    refreshAuto()
                }
                AutoActions.RestoreResult.NOTHING_TO_RESTORE -> autoMessage = "Không có bản gốc để khôi phục"
                AutoActions.RestoreResult.NEED_PERMISSION -> {
                    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
                        // App đang mở → được hiện hộp thoại hệ thống (có MANAGE_MEDIA thì hệ thống tự đồng ý)
                        pendingRestore = r
                        writeRequest = MediaStore.createWriteRequest(
                            getApplication<Application>().contentResolver, listOf(Uri.parse(r.uri))
                        ).intentSender
                    } else {
                        autoMessage = "Không có quyền ghi lại ảnh này"
                    }
                }
            }
        }
    }

    fun onWriteRequestResult(granted: Boolean) {
        writeRequest = null
        val r = pendingRestore ?: return
        pendingRestore = null
        if (granted) restore(r) else autoMessage = "Bạn đã từ chối quyền sửa ảnh → chưa khôi phục"
    }
}
