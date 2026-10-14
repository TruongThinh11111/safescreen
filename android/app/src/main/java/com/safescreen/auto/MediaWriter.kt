package com.safescreen.auto

import android.app.ActivityOptions
import android.app.PendingIntent
import android.content.ContentUris
import android.content.ContentValues
import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Build
import android.provider.MediaStore
import android.util.Log
import kotlinx.coroutines.delay
import java.io.IOException
import java.io.OutputStream

/**
 * Thay ảnh chụp màn hình (file của HỆ THỐNG, không phải của app) bằng bản đã che, qua MediaStore.
 *
 * Từ Android 10, app chỉ được sửa file của chính mình. Với file của app khác:
 *   - createWriteRequest (xin quyền GHI): LUÔN hiện hộp thoại "Cho phép SafeScreen sửa ảnh này?",
 *     kể cả khi đã có quyền "Quản lý phương tiện" (MANAGE_MEDIA) — đã kiểm chứng trên Android 16/17.
 *   - createDeleteRequest (xin XOÁ): có MANAGE_MEDIA thì hệ thống xoá luôn, KHÔNG hỏi.
 * → Cách "ghi đè" không cần hỏi: cất bản gốc → xoá ảnh gốc → tạo ảnh đã che cùng thư mục, cùng tên,
 *   cùng thời điểm chụp. Ảnh mới thuộc sở hữu của SafeScreen nên về sau sửa/khôi phục không cần hỏi nữa.
 */
object MediaWriter {

    private const val TAG = "SafeScreenAuto"

    /** Máy có hỗ trợ "Quản lý phương tiện" (Android 12+) không */
    val manageMediaSupported: Boolean get() = Build.VERSION.SDK_INT >= Build.VERSION_CODES.S

    fun canManageMedia(context: Context): Boolean =
        Build.VERSION.SDK_INT >= Build.VERSION_CODES.S && MediaStore.canManageMedia(context)

    /** Mở màn hình Cài đặt › Quản lý phương tiện của app (Android 12+) */
    fun manageMediaSettingsIntent(context: Context): Intent? =
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            Intent(android.provider.Settings.ACTION_REQUEST_MANAGE_MEDIA, Uri.parse("package:${context.packageName}"))
        } else null

    /** Thử ghi đè tại chỗ (cắt file về 0 rồi ghi). false = app chưa có quyền ghi Uri này. */
    fun tryWrite(context: Context, uri: Uri, write: (OutputStream) -> Unit): Boolean = try {
        // "wt" = ghi + cắt bỏ nội dung cũ (chỉ "w" thì có máy không cắt → ảnh mới ngắn hơn sẽ dính đuôi ảnh cũ)
        val os = context.contentResolver.openOutputStream(uri, "wt") ?: error("không mở được $uri")
        os.use(write)
        true
    } catch (e: SecurityException) {
        false
    }

    private fun exists(context: Context, uri: Uri): Boolean =
        context.contentResolver.query(uri, arrayOf(MediaStore.MediaColumns._ID), null, null, null)?.use { it.count > 0 } ?: false

    /**
     * Xoá ảnh của app khác mà KHÔNG hỏi người dùng (cần MANAGE_MEDIA, Android 12+). Dùng được cả khi app chạy nền:
     * PendingIntent của MediaStore mở 1 activity hệ thống nằm trong danh sách được phép mở từ nền.
     */
    suspend fun deleteSilently(context: Context, uri: Uri): Boolean {
        if (!canManageMedia(context)) return false
        // Android 15+: thiếu quyền "Hiển thị trên ứng dụng khác" thì activity xoá bị giữ ở nền, không làm gì
        // (log hệ thống: "only the creator of the PendingIntent allows BAL") → khỏi gửi
        if (Build.VERSION.SDK_INT >= 35 && !android.provider.Settings.canDrawOverlays(context)) return false
        val options = ActivityOptions.makeBasic()
        if (Build.VERSION.SDK_INT >= 36) {
            options.setPendingIntentBackgroundActivityStartMode(ActivityOptions.MODE_BACKGROUND_ACTIVITY_START_ALLOW_ALWAYS)
        } else if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
            @Suppress("DEPRECATION")
            options.setPendingIntentBackgroundActivityStartMode(ActivityOptions.MODE_BACKGROUND_ACTIVITY_START_ALLOWED)
        }
        try {
            MediaStore.createDeleteRequest(context.contentResolver, listOf(uri))
                .send(context, 0, null, null, null, null, options.toBundle())
        } catch (e: PendingIntent.CanceledException) {
            Log.w(TAG, "createDeleteRequest bị huỷ", e)
            return false
        }
        repeat(50) {  // activity hệ thống xoá bất đồng bộ → chờ tối đa 5 s
            if (!exists(context, uri)) return true
            delay(100)
        }
        Log.w(TAG, "Đã gửi createDeleteRequest nhưng ảnh vẫn còn: $uri (hệ thống đang hỏi người dùng?)")
        return false
    }

    /**
     * Tạo ảnh mới (thuộc SafeScreen) đúng thư mục + tên + thời điểm chụp của ảnh cũ.
     * onCreated(id) được gọi NGAY khi có _ID (trước khi ảnh hiện ra) để service đánh dấu "đã xử lý",
     * không tự xử lý lại chính ảnh mình vừa tạo.
     */
    fun insertImage(
        context: Context, relativePath: String, displayName: String, mime: String, dateTakenMs: Long?,
        onCreated: (Long) -> Unit, write: (OutputStream) -> Unit,
    ): Uri {
        require(Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q)
        val resolver = context.contentResolver
        val values = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, displayName)
            put(MediaStore.Images.Media.MIME_TYPE, mime)
            put(MediaStore.Images.Media.RELATIVE_PATH, relativePath)
            // Giữ thời điểm chụp → thư viện ảnh xếp ảnh đã che đúng chỗ ảnh cũ
            if (dateTakenMs != null) put(MediaStore.Images.Media.DATE_TAKEN, dateTakenMs)
            put(MediaStore.Images.Media.IS_PENDING, 1)
        }
        val uri = resolver.insert(ScreenshotQuery.collection, values) ?: throw IOException("MediaStore không tạo được ảnh mới")
        onCreated(ContentUris.parseId(uri))
        try {
            resolver.openOutputStream(uri, "wt")?.use(write) ?: throw IOException("không mở được $uri")
            resolver.update(uri, ContentValues().apply { put(MediaStore.Images.Media.IS_PENDING, 0) }, null, null)
            return uri
        } catch (e: Exception) {
            resolver.delete(uri, null, null)
            throw e
        }
    }
}
