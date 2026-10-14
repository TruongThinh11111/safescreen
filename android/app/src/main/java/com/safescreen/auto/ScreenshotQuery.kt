package com.safescreen.auto

import android.content.ContentUris
import android.content.Context
import android.net.Uri
import android.os.Build
import android.provider.MediaStore
import java.io.File

/** 1 ảnh trong MediaStore */
data class MediaImage(
    val id: Long,
    val uri: Uri,
    val displayName: String,
    val relativePath: String,
    val mime: String,
    val width: Int,
    val height: Int,
    val dateAddedMs: Long,     // lúc tạo bản ghi (độ phân giải 1 s)
    val fileWrittenMs: Long?,  // lúc file ghi xong (theo hệ thống file, độ phân giải ms) — null nếu không đọc được
    val dateTakenMs: Long?,    // thời điểm chụp (giữ lại khi tạo ảnh đã che để thư viện xếp đúng chỗ)
)

/** Tìm ảnh chụp màn hình MỚI trong MediaStore. */
object ScreenshotQuery {

    val collection: Uri = MediaStore.Images.Media.EXTERNAL_CONTENT_URI

    /** _ID lớn nhất hiện có → làm mốc "ảnh mới" lúc bật bảo vệ */
    fun maxId(context: Context): Long =
        context.contentResolver.query(collection, arrayOf(MediaStore.Images.Media._ID), null, null,
            "${MediaStore.Images.Media._ID} DESC")?.use { c -> if (c.moveToFirst()) c.getLong(0) else 0L } ?: 0L

    /**
     * Ảnh chụp màn hình có _ID > afterId, thêm vào trong 1 giờ qua, ĐÃ GHI XONG (IS_PENDING = 0).
     * Thư mục: Samsung lưu ở DCIM/Screenshots, Pixel/máy ảo ở Pictures/Screenshots → lọc "Screenshots".
     */
    fun newScreenshots(context: Context, afterId: Long): List<MediaImage> {
        val me = context.packageName
        val m = MediaStore.Images.Media._ID
        val since = System.currentTimeMillis() / 1000 - 3600
        val (selection, args) = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            // IS_PENDING = 1: ứng dụng chụp màn hình đang ghi dở → bỏ qua, lần sau (khi = 0) mới đọc
            "$m > ? AND ${MediaStore.Images.Media.DATE_ADDED} >= ? AND ${MediaStore.Images.Media.IS_PENDING} = 0 " +
                "AND ${MediaStore.Images.Media.RELATIVE_PATH} LIKE ?" to
                arrayOf(afterId.toString(), since.toString(), "%Screenshots%")
        } else {
            @Suppress("DEPRECATION")
            "$m > ? AND ${MediaStore.Images.Media.DATE_ADDED} >= ? AND ${MediaStore.Images.Media.DATA} LIKE ?" to
                arrayOf(afterId.toString(), since.toString(), "%/Screenshots/%")
        }
        @Suppress("DEPRECATION")
        val projection = arrayOf(
            m, MediaStore.Images.Media.DISPLAY_NAME, MediaStore.Images.Media.MIME_TYPE,
            MediaStore.Images.Media.WIDTH, MediaStore.Images.Media.HEIGHT, MediaStore.Images.Media.DATE_ADDED,
            MediaStore.Images.Media.DATA,
        ) + if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            arrayOf(MediaStore.Images.Media.RELATIVE_PATH, MediaStore.Images.Media.DATE_TAKEN, MediaStore.Images.Media.OWNER_PACKAGE_NAME)
        } else emptyArray()

        val out = mutableListOf<MediaImage>()
        context.contentResolver.query(collection, projection, selection, args, "$m ASC")?.use { c ->
            while (c.moveToNext()) {
                // Bỏ qua ảnh do chính SafeScreen tạo (ảnh đã che đặt lại vào thư mục Screenshots).
                // Lọc ở đây chứ KHÔNG đưa owner_package_name vào điều kiện SQL: từ Android 11, MediaProvider
                // không cho app thường lọc theo cột này (kết quả rỗng) — đã gặp khi test.
                if (projection.size > 9 && c.getString(9) == me) continue
                val id = c.getLong(0)
                val path = c.getString(6)
                val written = path?.let { File(it).lastModified() }?.takeIf { it > 0 }  // 0 = không đọc được
                out += MediaImage(
                    id = id,
                    uri = ContentUris.withAppendedId(collection, id),
                    displayName = c.getString(1) ?: "screenshot_$id",
                    relativePath = if (projection.size > 7) c.getString(7).orEmpty() else path.orEmpty(),
                    mime = c.getString(2) ?: "image/png",
                    width = c.getInt(3),
                    height = c.getInt(4),
                    dateAddedMs = c.getLong(5) * 1000,
                    fileWrittenMs = written,
                    dateTakenMs = if (projection.size > 8 && !c.isNull(8)) c.getLong(8) else null,
                )
            }
        }
        return out
    }
}
