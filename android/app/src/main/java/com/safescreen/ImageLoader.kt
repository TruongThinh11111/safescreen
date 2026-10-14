package com.safescreen

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import java.io.IOException
import kotlin.math.max

/** Đọc ảnh từ Uri (Photo Picker trả về) thành Bitmap. Gọi trên luồng nền vì đọc file khá chậm. */
object ImageLoader {

    // Cạnh dài tối đa: ảnh lớn hơn sẽ được thu nhỏ để tránh hết bộ nhớ.
    // Phải đủ lớn để ảnh chụp màn hình (vd 1080x2400, 1440x3200) giữ NGUYÊN kích thước:
    // Python OCR trên ảnh gốc, thu nhỏ trước sẽ làm chữ bé đi và OCR khác hẳn.
    private const val MAX_SIDE = 4096

    fun decode(context: Context, uri: Uri): Bitmap {
        val resolver = context.contentResolver
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            // Android 9+ (API 28): ImageDecoder, tự xoay ảnh đúng chiều theo EXIF
            val source = ImageDecoder.createSource(resolver, uri)
            ImageDecoder.decodeBitmap(source) { decoder, info, _ ->
                // Mặc định ImageDecoder tạo bitmap HARDWARE (không đọc được pixel).
                // Ép dùng bộ nhớ thường để sau này OCR/xử lý pixel được.
                decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
                decoder.setTargetSampleSize(sampleSize(info.size.width, info.size.height))
            }
        } else {
            // Android 8.x (API 26–27): BitmapFactory. Bước 1 chỉ đọc kích thước, bước 2 mới đọc pixel.
            val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
            resolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, bounds) }
            val options = BitmapFactory.Options().apply {
                inSampleSize = sampleSize(bounds.outWidth, bounds.outHeight)
            }
            resolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, options) }
                ?: throw IOException("Không đọc được ảnh: $uri")
        }
    }

    /** Hệ số thu nhỏ (lũy thừa của 2): 1 = giữ nguyên, 2 = còn 1/2 mỗi chiều, ... */
    private fun sampleSize(width: Int, height: Int): Int {
        var sample = 1
        while (max(width, height) / sample > MAX_SIDE) sample *= 2
        return sample
    }
}
