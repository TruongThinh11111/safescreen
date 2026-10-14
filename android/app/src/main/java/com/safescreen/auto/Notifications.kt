package com.safescreen.auto

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.os.Build
import androidx.core.app.NotificationCompat
import androidx.core.app.NotificationManagerCompat
import com.safescreen.MainActivity
import com.safescreen.R

/** Thông báo của tính năng bảo vệ tự động. */
object Notifications {

    const val SERVICE_ID = 1
    private const val CH_SERVICE = "protect"
    private const val CH_RESULT = "result"

    const val EXTRA_RECORD_ID = "com.safescreen.RECORD_ID"
    const val EXTRA_RESTORE = "com.safescreen.RESTORE"

    fun createChannels(context: Context) {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) return
        val nm = context.getSystemService(NotificationManager::class.java)
        nm.createNotificationChannel(
            NotificationChannel(CH_SERVICE, "Bảo vệ tự động", NotificationManager.IMPORTANCE_LOW).apply {
                description = "Thông báo thường trực khi SafeScreen đang theo dõi ảnh chụp màn hình"
            }
        )
        nm.createNotificationChannel(
            NotificationChannel(CH_RESULT, "Kết quả che ảnh", NotificationManager.IMPORTANCE_DEFAULT).apply {
                description = "Báo khi một ảnh chụp màn hình vừa được che"
            }
        )
    }

    /** Thông báo thường trực của foreground service */
    fun service(context: Context): Notification =
        NotificationCompat.Builder(context, CH_SERVICE)
            .setSmallIcon(R.drawable.ic_stat_shield)
            .setContentTitle("SafeScreen đang bảo vệ")
            .setContentText("Ảnh chụp màn hình mới sẽ được tự động che thông tin nhạy cảm")
            .setOngoing(true)
            .setContentIntent(openApp(context, requestCode = 0, recordId = null, restore = false))
            .setForegroundServiceBehavior(NotificationCompat.FOREGROUND_SERVICE_IMMEDIATE)
            .build()

    /** "Đã che N vùng nhạy cảm (…)" + nút "Khôi phục ảnh gốc" */
    @android.annotation.SuppressLint("MissingPermission")  // đã kiểm tra areNotificationsEnabled()
    fun result(context: Context, r: AutoRecord) {
        val nm = NotificationManagerCompat.from(context)
        if (!nm.areNotificationsEnabled()) return
        val id = notificationId(r.mediaId)
        val seconds = r.savedToDoneMs?.let { " · ${"%.1f".format(it / 1000.0)} s" }.orEmpty()
        val builder = NotificationCompat.Builder(context, CH_RESULT)
            .setSmallIcon(R.drawable.ic_stat_shield)
            .setAutoCancel(true)
            .setContentIntent(openApp(context, id * 2, r.mediaId, restore = false))
        if (r.overwritten) {
            builder.setContentTitle("Đã che ${r.labels.size} vùng nhạy cảm")
                .setContentText("${r.labelSummary}$seconds")
                .addAction(0, "Khôi phục ảnh gốc", openApp(context, id * 2 + 1, r.mediaId, restore = true))
        } else {
            builder.setContentTitle("Đã lưu bản che riêng (${r.labels.size} vùng: ${r.labelSummary})")
                .setStyle(NotificationCompat.BigTextStyle().bigText(
                    "Chưa ghi đè được ảnh gốc (chưa cấp quyền Quản lý phương tiện). " +
                        "Bản đã che nằm trong Ảnh › SafeScreen; ảnh gốc vẫn còn trong Screenshots."
                ))
        }
        nm.notify(id, builder.build())
    }

    fun cancelResult(context: Context, mediaId: Long) = NotificationManagerCompat.from(context).cancel(notificationId(mediaId))

    private fun notificationId(mediaId: Long) = 1000 + (mediaId % 100_000).toInt()

    private fun openApp(context: Context, requestCode: Int, recordId: Long?, restore: Boolean): PendingIntent {
        val intent = Intent(context, MainActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_SINGLE_TOP
            if (recordId != null) putExtra(EXTRA_RECORD_ID, recordId)
            if (restore) putExtra(EXTRA_RESTORE, true)
        }
        return PendingIntent.getActivity(context, requestCode, intent,
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT)
    }
}
