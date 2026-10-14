package com.safescreen.auto

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import com.safescreen.AppSettings

/** Bật lại bảo vệ sau khi khởi động máy hoặc cập nhật app (nếu người dùng đang để Bật). */
class BootReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        if (intent.action != Intent.ACTION_BOOT_COMPLETED && intent.action != Intent.ACTION_MY_PACKAGE_REPLACED) return
        if (AppSettings(context).autoProtect) AutoProtectService.start(context)
    }
}
