package com.safescreen.auto

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.PowerManager
import android.provider.Settings
import androidx.core.app.NotificationManagerCompat
import androidx.core.content.ContextCompat

/** Quyền đọc ảnh: TẤT CẢ / MỘT PHẦN ("Chọn ảnh", Android 14+) / KHÔNG */
enum class MediaAccess { FULL, PARTIAL, NONE }

/** Trạng thái các quyền mà bảo vệ tự động cần (đọc lại mỗi lần quay về app). */
data class PermissionStatus(
    val media: MediaAccess,
    val notifications: Boolean,
    val manageMediaSupported: Boolean,
    val manageMedia: Boolean,
    val overlay: Boolean,
    val batteryUnrestricted: Boolean,
) {
    /** Đủ điều kiện thay thẳng ảnh gốc khi app đang chạy nền */
    val canReplaceOriginal: Boolean get() = manageMedia && overlay

    companion object {
        fun read(context: Context): PermissionStatus {
            fun granted(p: String) = ContextCompat.checkSelfPermission(context, p) == PackageManager.PERMISSION_GRANTED
            val media = when {
                Build.VERSION.SDK_INT >= 33 && granted(Manifest.permission.READ_MEDIA_IMAGES) -> MediaAccess.FULL
                Build.VERSION.SDK_INT >= 34 && granted(Manifest.permission.READ_MEDIA_VISUAL_USER_SELECTED) -> MediaAccess.PARTIAL
                Build.VERSION.SDK_INT < 33 && granted(Manifest.permission.READ_EXTERNAL_STORAGE) -> MediaAccess.FULL
                else -> MediaAccess.NONE
            }
            val pm = context.getSystemService(PowerManager::class.java)
            return PermissionStatus(
                media = media,
                notifications = NotificationManagerCompat.from(context).areNotificationsEnabled(),
                manageMediaSupported = MediaWriter.manageMediaSupported,
                manageMedia = MediaWriter.canManageMedia(context),
                overlay = Settings.canDrawOverlays(context),
                batteryUnrestricted = pm.isIgnoringBatteryOptimizations(context.packageName),
            )
        }

        /** Các quyền runtime cần xin khi bật bảo vệ */
        fun runtimePermissions(): Array<String> = buildList {
            if (Build.VERSION.SDK_INT >= 33) {
                add(Manifest.permission.READ_MEDIA_IMAGES)
                add(Manifest.permission.POST_NOTIFICATIONS)
            } else {
                add(Manifest.permission.READ_EXTERNAL_STORAGE)
            }
            // Khai báo cả quyền "chọn ảnh" để phân biệt được người dùng chọn "Cho phép tất cả" hay chỉ vài ảnh
            if (Build.VERSION.SDK_INT >= 34) add(Manifest.permission.READ_MEDIA_VISUAL_USER_SELECTED)
        }.toTypedArray()

        /** Cài đặt › "Hiển thị trên ứng dụng khác" của app */
        fun overlaySettingsIntent(context: Context) =
            Intent(Settings.ACTION_MANAGE_OVERLAY_PERMISSION, Uri.parse("package:${context.packageName}"))

        /** Màn hình thông tin ứng dụng (quyền, pin…) */
        fun appDetailsIntent(context: Context) =
            Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS, Uri.parse("package:${context.packageName}"))
    }
}
