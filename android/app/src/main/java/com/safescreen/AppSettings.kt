package com.safescreen

import android.content.Context
import com.safescreen.mask.MaskMode

/** Cài đặt lưu bền (SharedPreferences): dùng chung cho màn hình chính và service bảo vệ tự động. */
class AppSettings(context: Context) {

    private val prefs = context.applicationContext.getSharedPreferences("safescreen", Context.MODE_PRIVATE)

    /** Chế độ che đang chọn. null = chưa chọn lần nào (lấy mask.default_mode trong config.json). */
    var maskModeOrNull: MaskMode?
        get() = prefs.getString(KEY_MASK_MODE, null)?.let { MaskMode.fromConfig(it) }
        set(v) = prefs.edit().putString(KEY_MASK_MODE, v?.configName).apply()

    val maskMode: MaskMode get() = maskModeOrNull ?: MaskMode.BLACK_BOX

    /** Người dùng đã bật bảo vệ tự động chưa (service tự bật lại sau khi khởi động máy nếu true) */
    var autoProtect: Boolean
        get() = prefs.getBoolean(KEY_AUTO, false)
        set(v) = prefs.edit().putBoolean(KEY_AUTO, v).apply()

    /** Chỉ xử lý ảnh có _ID LỚN HƠN mốc này = ảnh mới chụp sau lúc bật bảo vệ */
    var baselineMediaId: Long
        get() = prefs.getLong(KEY_BASELINE, -1)
        set(v) = prefs.edit().putLong(KEY_BASELINE, v).apply()

    /** Các ảnh đã xử lý (kể cả ảnh không có gì nhạy cảm) → không xử lý lại khi service khởi động lại */
    var handledIds: Set<Long>
        get() = prefs.getStringSet(KEY_HANDLED, emptySet())!!.mapNotNull { it.toLongOrNull() }.toSet()
        set(v) = prefs.edit().putStringSet(KEY_HANDLED, v.sortedDescending().take(300).map { it.toString() }.toSet()).apply()

    private companion object {
        const val KEY_MASK_MODE = "mask_mode"
        const val KEY_AUTO = "auto_protect"
        const val KEY_BASELINE = "baseline_media_id"
        const val KEY_HANDLED = "handled_ids"
    }
}
