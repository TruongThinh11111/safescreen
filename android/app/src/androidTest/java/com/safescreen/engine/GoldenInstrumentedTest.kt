package com.safescreen.engine

import android.os.Build
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.util.regex.Pattern

/**
 * Golden test chạy TRÊN MÁY ANDROID (emulator hoặc điện thoại):  gradlew :app:connectedDebugAndroidTest
 * Đây là nơi duy nhất kiểm chứng được regex của Android (ICU), vốn khác java.util.regex của JVM.
 * Xem kết quả: adb logcat -s SafeScreenGolden
 */
@RunWith(AndroidJUnit4::class)
class GoldenInstrumentedTest {

    private val tag = "SafeScreenGolden"

    @Test
    fun golden_khopPython_trenAndroid() {
        val inst = InstrumentationRegistry.getInstrumentation()
        val appAssets = inst.targetContext.assets     // assets của app: mô hình
        val testAssets = inst.context.assets          // assets của gói test: golden_test.json
        val engine = SafeScreenEngine(
            appAssets.open("crf_model.json").bufferedReader().use { it.readText() },
            appAssets.open("config.json").bufferedReader().use { it.readText() },
        )
        val golden = testAssets.open("golden_test.json").bufferedReader().use { it.readText() }

        // Ghi lại regex của máy có nhận cờ UNICODE_CHARACTER_CLASS không (chỉ để biết, không ảnh hưởng kết quả)
        val unicodeFlag = try {
            Pattern.compile("\\w", Pattern.UNICODE_CHARACTER_CLASS); "nhận"
        } catch (e: IllegalArgumentException) {
            "từ chối (${e.message})"
        }
        val report = GoldenVerifier.verify(engine, golden)
        val msg = "[Android API ${Build.VERSION.SDK_INT}, ${Build.MODEL}, cờ UNICODE_CHARACTER_CLASS: $unicodeFlag] $report"
        msg.chunked(3000).forEach { Log.i(tag, it) }
        assertTrue(msg, report.passed)
    }
}
