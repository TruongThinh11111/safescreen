package com.safescreen.auto

import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.database.ContentObserver
import android.graphics.Bitmap
import android.os.Build
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.os.SystemClock
import android.util.Log
import androidx.core.app.ServiceCompat
import androidx.core.content.ContextCompat
import com.safescreen.AppSettings
import com.safescreen.ImageLoader
import com.safescreen.SafeScreenPipeline
import com.safescreen.mask.GallerySaver
import com.safescreen.mask.masked
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * Foreground service: theo dõi MediaStore, mỗi khi có ẢNH CHỤP MÀN HÌNH MỚI (đã ghi xong) thì
 * chạy OCR → CRF → F → che, rồi ghi đè ảnh đã che lên chính file đó (bản gốc cất vào filesDir).
 *
 * Log: adb logcat -s SafeScreenAuto   ·   số liệu: files/auto/latency_log.csv
 */
class AutoProtectService : Service() {

    companion object {
        private const val TAG = "SafeScreenAuto"

        @Volatile var isRunning = false
            private set

        fun start(context: Context) =
            ContextCompat.startForegroundService(context, Intent(context, AutoProtectService::class.java))

        fun stop(context: Context) = context.stopService(Intent(context, AutoProtectService::class.java))
    }

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    // CONFLATED: MediaStore báo thay đổi dồn dập (tạo bản ghi, IS_PENDING 1→0, quét lại…) → gộp thành 1 lần quét
    private val trigger = Channel<Unit>(Channel.CONFLATED)
    private lateinit var settings: AppSettings
    private lateinit var store: AutoMaskStore
    private val observer = object : ContentObserver(Handler(Looper.getMainLooper())) {
        override fun onChange(selfChange: Boolean, uri: android.net.Uri?) {
            Log.d(TAG, "MediaStore thay đổi: $uri")
            trigger.trySend(Unit)
        }
    }

    override fun onCreate() {
        super.onCreate()
        settings = AppSettings(this)
        store = AutoMaskStore(this)
        Notifications.createChannels(this)
        // Android 14+ bắt buộc khai loại foreground service; "specialUse" vì không loại có sẵn nào khớp
        // (dataSync / mediaProcessing bị giới hạn 6 giờ/ngày từ Android 15)
        ServiceCompat.startForeground(
            this, Notifications.SERVICE_ID, Notifications.service(this),
            if (Build.VERSION.SDK_INT >= 34) ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE else 0,
        )
        if (settings.baselineMediaId < 0) settings.baselineMediaId = ScreenshotQuery.maxId(this)
        contentResolver.registerContentObserver(ScreenshotQuery.collection, true, observer)
        isRunning = true
        Log.i(TAG, "Bắt đầu bảo vệ, chỉ xử lý ảnh có _ID > ${settings.baselineMediaId}")

        scope.launch {
            for (t in trigger) {
                delay(300)  // chờ loạt thông báo thay đổi lắng xuống
                scan()
            }
        }
        trigger.trySend(Unit)  // quét 1 lần lúc khởi động: bắt ảnh chụp trong lúc service bị hệ thống tắt
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int = START_STICKY  // bị tắt thì tự bật lại

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onDestroy() {
        contentResolver.unregisterContentObserver(observer)
        scope.cancel()
        isRunning = false
        Log.i(TAG, "Dừng bảo vệ")
        super.onDestroy()
    }

    private suspend fun scan() {
        val images = try {
            ScreenshotQuery.newScreenshots(this, settings.baselineMediaId)
        } catch (e: SecurityException) {
            Log.w(TAG, "Không đọc được MediaStore (thiếu quyền đọc ảnh?)", e)
            return
        }
        Log.d(TAG, "Quét: ${images.size} ảnh chụp màn hình mới ${images.map { it.id }}")
        for (img in images) {
            val handled = settings.handledIds
            if (img.id in handled) continue
            // Đánh dấu TRƯỚC khi xử lý: chính lần ghi đè của mình cũng làm MediaStore báo thay đổi → không xử lý lại
            settings.handledIds = handled + img.id
            try {
                process(img)
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {  // 1 ảnh lỗi không được làm chết cả service
                Log.e(TAG, "Lỗi khi xử lý ${img.displayName}", e)
            }
        }
    }

    private suspend fun process(img: MediaImage) {
        val detectedAt = System.currentTimeMillis()
        // "Lúc ảnh được lưu": ưu tiên thời điểm file ghi xong (ms); không có thì DATE_ADDED (chỉ chính xác tới giây)
        val savedAt = img.fileWrittenMs ?: img.dateAddedMs
        val savedSource = if (img.fileWrittenMs != null) "file_mtime" else "date_added"
        fun now() = SystemClock.elapsedRealtime()

        var t = now()
        val bmp = withContext(Dispatchers.IO) { ImageLoader.decode(this@AutoProtectService, img.uri) }
        val decodeMs = now() - t

        val out = SafeScreenPipeline.analyze(this, bmp)
        val regions = out.result.regions
        val ocrMs = out.ocr.preprocessMs + out.ocr.recognizeMs
        if (regions.isEmpty()) {
            Log.i(TAG, "${img.displayName}: không có thông tin nhạy cảm → giữ nguyên (OCR $ocrMs ms, CRF ${out.crfMs} ms)")
            store.appendLatency(listOf(img.id, img.displayName, img.width, img.height, savedAt, savedSource, detectedAt,
                decodeMs, ocrMs, out.crfMs, null, null, null, 0, "none", null, System.currentTimeMillis() - detectedAt))
            return
        }

        // Che
        val mode = settings.maskMode
        t = now()
        val masked = bmp.masked(regions.map { it.bbox }, mode, SafeScreenPipeline.maskPadPx)
        val maskMs = now() - t

        // Cất bản gốc (nguyên từng byte) vào bộ nhớ riêng TRƯỚC khi ghi đè
        t = now()
        val backup = store.backupFile(img.id)
        withContext(Dispatchers.IO) {
            contentResolver.openInputStream(img.uri)!!.use { input -> backup.outputStream().use { input.copyTo(it) } }
        }
        val backupMs = now() - t

        // Thay ảnh gốc bằng bản đã che. Thứ tự thử:
        //   1. ghi đè tại chỗ (nếu app đã có quyền ghi file này)
        //   2. có MANAGE_MEDIA: xoá ảnh gốc (không hỏi) + tạo ảnh đã che cùng thư mục/tên/thời điểm chụp
        //   3. không được: lưu bản che riêng vào Pictures/SafeScreen, ảnh gốc giữ nguyên
        t = now()
        // Ảnh đã bị thu nhỏ lúc đọc (ảnh cuộn rất dài) → không thay, kẻo làm hỏng kích thước ảnh gốc
        val sameSize = img.width <= 0 || (bmp.width == img.width && bmp.height == img.height)
        val format = if (img.mime == "image/png") Bitmap.CompressFormat.PNG else Bitmap.CompressFormat.JPEG
        val write: (java.io.OutputStream) -> Unit = { masked.compress(format, if (format == Bitmap.CompressFormat.PNG) 100 else 95, it) }
        var method = AutoRecord.METHOD_COPY
        var finalUri = img.uri
        if (sameSize && withContext(Dispatchers.IO) { MediaWriter.tryWrite(this@AutoProtectService, img.uri, write) }) {
            method = AutoRecord.METHOD_IN_PLACE
        } else if (sameSize && Build.VERSION.SDK_INT >= Build.VERSION_CODES.R && MediaWriter.deleteSilently(this, img.uri)) {
            finalUri = try {
                withContext(Dispatchers.IO) {
                    MediaWriter.insertImage(this@AutoProtectService, img.relativePath, img.displayName, img.mime,
                        img.dateTakenMs ?: savedAt, onCreated = { settings.handledIds = settings.handledIds + it }, write = write)
                }
            } catch (e: Exception) {
                if (e is CancellationException) throw e
                // Đã xoá ảnh gốc mà không tạo được ảnh mới → đặt lại ảnh gốc từ bản sao lưu, không để mất ảnh
                Log.e(TAG, "Tạo ảnh đã che thất bại, đặt lại ảnh gốc", e)
                withContext(Dispatchers.IO) {
                    MediaWriter.insertImage(this@AutoProtectService, img.relativePath, img.displayName, img.mime,
                        img.dateTakenMs ?: savedAt, onCreated = { settings.handledIds = settings.handledIds + it },
                        write = { os -> backup.inputStream().use { it.copyTo(os) } })
                }
                throw e
            }
            method = AutoRecord.METHOD_REPLACED
        }
        var copyUri: String? = null
        if (method == AutoRecord.METHOD_COPY) {
            val name = "safescreen_" + img.displayName.substringBeforeLast('.') + ".png"
            copyUri = withContext(Dispatchers.IO) { GallerySaver.savePng(this@AutoProtectService, masked, name) }.toString()
            backup.delete()  // ảnh gốc vẫn còn nguyên trong Screenshots → không cần giữ bản sao
        }
        val writeMs = now() - t
        val doneAt = System.currentTimeMillis()

        val record = AutoRecord(
            mediaId = img.id, uri = finalUri.toString(), displayName = img.displayName,
            labels = regions.flatMap { r -> r.labels },
            maskMode = mode.configName, method = method, maskedCopyUri = copyUri, restored = false,
            createdAtMs = doneAt, savedToDoneMs = doneAt - savedAt, detectToDoneMs = doneAt - detectedAt,
        )
        store.upsert(record)
        store.appendLatency(listOf(img.id, img.displayName, img.width, img.height, savedAt, savedSource, detectedAt,
            decodeMs, ocrMs, out.crfMs, maskMs, backupMs, writeMs, regions.size,
            method, doneAt - savedAt, doneAt - detectedAt))
        Log.i(TAG, "${img.displayName}: ${regions.size} vùng (${record.labelSummary}) → " +
            "${when (method) { AutoRecord.METHOD_IN_PLACE -> "GHI ĐÈ TẠI CHỖ"; AutoRecord.METHOD_REPLACED -> "THAY ẢNH GỐC"; else -> "LƯU BẢN RIÊNG" }} | lưu→phát hiện ${detectedAt - savedAt} ms ($savedSource), " +
            "đọc $decodeMs, OCR $ocrMs, CRF ${out.crfMs}, che $maskMs, sao lưu $backupMs, ghi $writeMs ms | " +
            "TỔNG lưu→xong ${doneAt - savedAt} ms, phát hiện→xong ${doneAt - detectedAt} ms")
        Notifications.result(this, record)
    }
}
