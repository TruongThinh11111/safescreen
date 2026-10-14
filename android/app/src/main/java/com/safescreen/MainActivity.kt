package com.safescreen

import android.Manifest
import android.content.pm.PackageManager
import android.os.Build
import android.content.Intent
import android.os.Bundle
import android.view.WindowManager
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.viewModels
import androidx.activity.compose.LocalActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts.PickVisualMedia
import androidx.activity.result.contract.ActivityResultContracts.RequestPermission
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.FilterChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.drawWithContent
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.TextMeasurer
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.drawText
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import com.safescreen.engine.EngineResult
import com.safescreen.mask.MaskMode
import java.util.Locale
import kotlin.math.min

/** Activity duy nhất của app: chỉ dựng giao diện Compose, mọi logic nằm trong MainViewModel. */
class MainActivity : ComponentActivity() {
    private val vm: MainViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()  // vẽ tràn màn hình; Scaffold bên dưới tự chừa chỗ cho thanh trạng thái
        // Mở từ thông báo "Đã che…" (chỉ lần tạo đầu, không phải lúc xoay màn hình → không khôi phục 2 lần)
        if (savedInstanceState == null) vm.handleIntent(intent)
        setContent {
            MaterialTheme {
                MainScreen(vm)
            }
        }
    }

    /** App đang mở mà bấm thông báo → singleTop: nhận Intent mới ở đây thay vì mở màn hình mới */
    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        vm.handleIntent(intent)
    }
}

@Composable
fun MainScreen(vm: MainViewModel) {
    Scaffold { innerPadding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding)
                .padding(horizontal = 16.dp, vertical = 8.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                FilterChip(selected = vm.tab == Tab.MANUAL, onClick = { vm.tab = Tab.MANUAL }, label = { Text("Che thủ công") })
                FilterChip(selected = vm.tab == Tab.AUTO, onClick = { vm.tab = Tab.AUTO }, label = { Text("Bảo vệ tự động") })
            }
            when (vm.tab) {
                Tab.MANUAL -> ManualTab(vm)
                Tab.AUTO -> AutoProtectTab(vm)
            }
        }
    }
}

@Composable
private fun ColumnScope.ManualTab(vm: MainViewModel) {
    val context = LocalContext.current
    var showOriginal by rememberSaveable { mutableStateOf(false) }
    var showDebug by rememberSaveable { mutableStateOf(false) }

    val pickImage = rememberLauncherForActivityResult(PickVisualMedia()) { uri ->
        if (uri != null) {
            showOriginal = false
            vm.onImagePicked(uri)
        }
    }
    // Android 8–9 cần xin quyền ghi bộ nhớ trước khi lưu ảnh; Android 10+ lưu thẳng
    val askWritePermission = rememberLauncherForActivityResult(RequestPermission()) { granted ->
        if (granted) vm.saveMasked()
        else Toast.makeText(context, "Cần quyền ghi bộ nhớ để lưu ảnh", Toast.LENGTH_SHORT).show()
    }
    fun onSaveClick() {
        val needPermission = Build.VERSION.SDK_INT < Build.VERSION_CODES.Q &&
            ContextCompat.checkSelfPermission(context, Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED
        if (needPermission) askWritePermission.launch(Manifest.permission.WRITE_EXTERNAL_STORAGE) else vm.saveMasked()
    }

    val original = vm.original
    val masked = vm.masked
    // Đang hiện ảnh GỐC khi: người dùng bật "Xem ảnh gốc", hoặc chưa che xong
    val displayed = if (showOriginal || masked == null) original else masked
    SecureWhen(displayed != null && displayed === original)

    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        Button(
            onClick = { pickImage.launch(PickVisualMediaRequest(PickVisualMedia.ImageOnly)) },
            enabled = vm.busy == null,
        ) { Text("Chọn ảnh") }
        OutlinedButton(onClick = ::onSaveClick, enabled = masked != null && vm.busy == null) {
            Text("Lưu ảnh đã che")
        }
    }

    // Chọn cách che (bảo vệ tự động cũng dùng chế độ này)
    MaskModeChips(vm)

    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        Switch(checked = showOriginal, onCheckedChange = { showOriginal = it })
        Text("Ảnh gốc", style = MaterialTheme.typography.bodyMedium)
        Switch(checked = showDebug, onCheckedChange = { showDebug = it })
        Text("Khung + nhãn + F", style = MaterialTheme.typography.bodyMedium)
    }

    StatusLine(vm)

    if (displayed != null) {
        val textMeasurer = rememberTextMeasurer()
        val result = vm.result
        Image(
            bitmap = displayed.asImageBitmap(),
            contentDescription = if (displayed === original) "Ảnh gốc" else "Ảnh đã che",
            contentScale = ContentScale.Fit,
            modifier = Modifier
                .fillMaxWidth()
                .weight(1f)
                .then(
                    if (showDebug && result != null) Modifier.debugOverlay(result, displayed.width, displayed.height, textMeasurer)
                    else Modifier
                ),
        )
    } else if (vm.busy == null && vm.error == null) {
        Text("Chưa chọn ảnh nào")
    }
}

@Composable
fun MaskModeChips(vm: MainViewModel) {
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        for (mode in MaskMode.entries) {
            FilterChip(
                selected = vm.maskMode == mode,
                onClick = { vm.changeMaskMode(mode) },
                label = { Text(mode.displayName) },
            )
        }
    }
}

/** Dòng trạng thái: đang làm gì / lỗi / thời gian từng bước / kết quả lưu. */
@Composable
private fun StatusLine(vm: MainViewModel) {
    val small = MaterialTheme.typography.bodySmall
    val busy = vm.busy
    val error = vm.error
    when {
        busy != null -> Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            CircularProgressIndicator(modifier = Modifier.size(18.dp), strokeWidth = 2.dp)
            Text(busy, style = small)
        }
        error != null -> Text(error, color = MaterialTheme.colorScheme.error, style = small)
    }
    val t = vm.timings
    val ocrMs = if (t.ocrPreprocessMs != null && t.ocrTesseractMs != null) t.ocrPreprocessMs + t.ocrTesseractMs else null
    if (ocrMs != null) {
        val parts = buildList {
            add("OCR $ocrMs ms (tiền xử lý ${t.ocrPreprocessMs} + Tesseract ${t.ocrTesseractMs})")
            t.crfMs?.let { add("CRF + F $it ms") }
            t.maskMs?.let { add("che $it ms") }
        }
        Text(parts.joinToString(" · "), style = small)
        val r = vm.result
        val words = vm.ocr?.tokens?.size ?: 0
        Text(
            buildString {
                append("$words từ")
                if (r != null) append(" · ${r.tokens.count { it.mask }} từ nhạy cảm → ${r.regions.size} vùng che")
                t.initMs?.let { append(" · nạp mô hình lần đầu $it ms") }
            },
            style = small,
        )
    }
    vm.saveMessage?.let { Text(it, style = small, color = MaterialTheme.colorScheme.primary) }
}

/**
 * Bật FLAG_SECURE khi màn hình đang hiện ẢNH GỐC: chặn chụp/quay màn hình, và ảnh thu nhỏ
 * trong màn hình đa nhiệm (Recents) bị che đen. Tắt lại khi chỉ hiện ảnh đã che.
 */
@Composable
private fun SecureWhen(secure: Boolean) {
    val window = LocalActivity.current?.window ?: return
    DisposableEffect(secure) {
        if (secure) window.addFlags(WindowManager.LayoutParams.FLAG_SECURE)
        else window.clearFlags(WindowManager.LayoutParams.FLAG_SECURE)
        // Rời tab (hoặc đóng màn hình) khi đang bật → gỡ cờ, kẻo các màn hình khác cũng bị chặn chụp
        onDispose { if (secure) window.clearFlags(WindowManager.LayoutParams.FLAG_SECURE) }
    }
}

private val GREEN = Color(0xFF00C853)    // từ bình thường
private val ORANGE = Color(0xFFFF9100)   // CRF gắn nhãn nhạy cảm nhưng F < τ → KHÔNG che
private val RED = Color(0xFFE53935)      // F ≥ τ → che

/**
 * DEMO: vẽ khung quanh mọi từ (xanh / cam / đỏ) và nhãn + điểm F trên mỗi vùng che.
 * Ảnh hiển thị kiểu Fit → bị thu nhỏ theo hệ số s và căn giữa → đổi toạ độ ảnh sang toạ độ màn hình.
 */
private fun Modifier.debugOverlay(result: EngineResult, imageWidth: Int, imageHeight: Int, textMeasurer: TextMeasurer): Modifier =
    drawWithContent {
        drawContent()
        val s = min(size.width / imageWidth, size.height / imageHeight)
        val ox = (size.width - imageWidth * s) / 2
        val oy = (size.height - imageHeight * s) / 2
        fun box(b: List<Int>, color: Color, width: Float) = drawRect(
            color = color,
            topLeft = Offset(ox + b[0] * s, oy + b[1] * s),
            size = Size((b[2] - b[0]) * s, (b[3] - b[1]) * s),
            style = Stroke(width),
        )
        for (t in result.tokens) {
            val color = when {
                t.mask -> RED
                t.label != "O" -> ORANGE
                else -> GREEN
            }
            box(t.word.bbox, color, 1.dp.toPx())
        }
        val tagStyle = TextStyle(color = Color.White, fontSize = 9.sp, background = RED)
        for (r in result.regions) {
            box(r.bbox, RED, 2.dp.toPx())
            val tag = r.labels.joinToString("+") + " · F " + String.format(Locale.ROOT, "%.2f", r.f)
            val layout = textMeasurer.measure(tag, tagStyle)
            val y = (oy + r.bbox[1] * s - layout.size.height).coerceAtLeast(0f)  // đặt ngay trên khung
            drawText(layout, topLeft = Offset(ox + r.bbox[0] * s, y))
        }
    }
