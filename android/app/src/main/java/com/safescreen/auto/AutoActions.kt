package com.safescreen.auto

import android.content.Context
import android.net.Uri
import android.util.Log
import com.safescreen.AppSettings

/** Các thao tác người dùng bấm từ màn hình chính. */
object AutoActions {

    enum class RestoreResult { RESTORED, NEED_PERMISSION, NOTHING_TO_RESTORE }

    /**
     * Ghi bản gốc (đã cất trong filesDir) trở lại file ảnh chụp màn hình.
     * NEED_PERMISSION: app chưa có quyền ghi file này → màn hình chính gọi createWriteRequest rồi thử lại.
     */
    fun restore(context: Context, record: AutoRecord): RestoreResult {
        val store = AutoMaskStore(context)
        val backup = store.backupFile(record.mediaId)
        if (!record.overwritten || record.restored || !backup.isFile) return RestoreResult.NOTHING_TO_RESTORE
        val ok = MediaWriter.tryWrite(context, Uri.parse(record.uri)) { os -> backup.inputStream().use { it.copyTo(os) } }
        if (!ok) return RestoreResult.NEED_PERMISSION
        backup.delete()
        store.upsert(record.copy(restored = true))
        Notifications.cancelResult(context, record.mediaId)
        Log.i("SafeScreenAuto", "Đã khôi phục ảnh gốc ${record.displayName}")
        return RestoreResult.RESTORED
    }

    /** Bật/tắt bảo vệ. Bật lại = chỉ xử lý ảnh chụp SAU thời điểm này. */
    fun setEnabled(context: Context, enabled: Boolean) {
        val settings = AppSettings(context)
        settings.autoProtect = enabled
        if (enabled) {
            settings.baselineMediaId = ScreenshotQuery.maxId(context)
            settings.handledIds = emptySet()
            AutoProtectService.start(context)
        } else {
            AutoProtectService.stop(context)
        }
    }
}
