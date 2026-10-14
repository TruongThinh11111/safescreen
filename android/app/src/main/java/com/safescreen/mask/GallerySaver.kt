package com.safescreen.mask

import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import java.io.IOException

/** Lưu ảnh vào thư viện ảnh của máy (album Pictures/SafeScreen) qua MediaStore. */
object GallerySaver {

    const val ALBUM = "SafeScreen"

    /**
     * Android 10+ (API 29): không cần quyền gì.
     * Android 8–9 (API 26–28): cần quyền WRITE_EXTERNAL_STORAGE (MainScreen xin trước khi gọi).
     * Ảnh lưu dạng PNG từ Bitmap → không mang theo EXIF/vị trí GPS của ảnh gốc.
     */
    fun savePng(context: Context, bitmap: Bitmap, displayName: String): Uri {
        val resolver = context.contentResolver
        val values = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, displayName)
            put(MediaStore.Images.Media.MIME_TYPE, "image/png")
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                put(MediaStore.Images.Media.RELATIVE_PATH, "${Environment.DIRECTORY_PICTURES}/$ALBUM")
                put(MediaStore.Images.Media.IS_PENDING, 1)  // ẩn khỏi thư viện tới khi ghi xong
            }
        }
        val collection = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            MediaStore.Images.Media.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY)
        } else {
            MediaStore.Images.Media.EXTERNAL_CONTENT_URI
        }
        val uri = resolver.insert(collection, values) ?: throw IOException("MediaStore không tạo được file mới")
        try {
            val ok = resolver.openOutputStream(uri)?.use { bitmap.compress(Bitmap.CompressFormat.PNG, 100, it) } ?: false
            if (!ok) throw IOException("Không ghi được ảnh")
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                resolver.update(uri, ContentValues().apply { put(MediaStore.Images.Media.IS_PENDING, 0) }, null, null)
            }
            return uri
        } catch (e: Exception) {
            resolver.delete(uri, null, null)  // ghi hỏng thì xoá, không để lại file rỗng trong thư viện
            throw e
        }
    }
}
