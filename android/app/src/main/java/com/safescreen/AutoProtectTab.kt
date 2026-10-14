package com.safescreen

import android.app.Activity
import android.content.Context
import android.content.Intent
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.IntentSenderRequest
import androidx.activity.result.contract.ActivityResultContracts.RequestMultiplePermissions
import androidx.activity.result.contract.ActivityResultContracts.StartIntentSenderForResult
import androidx.compose.foundation.Image
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.LifecycleResumeEffect
import com.safescreen.auto.AutoRecord
import com.safescreen.auto.MediaAccess
import com.safescreen.auto.MediaWriter
import com.safescreen.auto.PermissionStatus
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/** Tab "Bảo vệ tự động": công tắc bật/tắt, trạng thái quyền, hướng dẫn pin Samsung, danh sách ảnh đã che. */
@Composable
fun ColumnScope.AutoProtectTab(vm: MainViewModel) {
    val context = LocalContext.current

    // Quay lại app (vd. từ Cài đặt) → đọc lại quyền + danh sách ảnh đã che
    LifecycleResumeEffect(Unit) {
        vm.refreshAuto()
        onPauseOrDispose { }
    }

    // Bật công tắc → xin quyền trước; chỉ bật khi được đọc TẤT CẢ ảnh
    val askPermissions = rememberLauncherForActivityResult(RequestMultiplePermissions()) {
        vm.refreshAuto()
        when (PermissionStatus.read(context).media) {
            MediaAccess.FULL -> vm.enableAutoProtect(true)
            MediaAccess.PARTIAL -> vm.autoMessage = "Bạn mới cho phép một vài ảnh. Hãy vào Cài đặt › Quyền › Ảnh và video › chọn \"Cho phép tất cả\"."
            MediaAccess.NONE -> vm.autoMessage = "Cần quyền đọc ảnh để phát hiện ảnh chụp màn hình mới."
        }
    }
    // Hộp thoại hệ thống "Cho phép SafeScreen sửa ảnh này?" (dùng khi khôi phục ảnh gốc)
    val writeLauncher = rememberLauncherForActivityResult(StartIntentSenderForResult()) {
        vm.onWriteRequestResult(it.resultCode == Activity.RESULT_OK)
    }
    val request = vm.writeRequest
    LaunchedEffect(request) {
        if (request != null) writeLauncher.launch(IntentSenderRequest.Builder(request).build())
    }

    val p = vm.permissions
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .weight(1f)
            .verticalScroll(rememberScrollState()),
        verticalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        Card(Modifier.fillMaxWidth()) {
            Row(Modifier.padding(16.dp), verticalAlignment = Alignment.CenterVertically) {
                Column(Modifier.weight(1f)) {
                    Text("Bảo vệ tự động", style = MaterialTheme.typography.titleMedium)
                    Text(
                        "Tự che thông tin nhạy cảm trên mọi ảnh chụp màn hình mới",
                        style = MaterialTheme.typography.bodySmall,
                    )
                }
                Switch(
                    checked = vm.autoProtect,
                    onCheckedChange = { on ->
                        if (!on) vm.enableAutoProtect(false)
                        else if (p.media == MediaAccess.FULL && p.notifications) vm.enableAutoProtect(true)
                        else askPermissions.launch(PermissionStatus.runtimePermissions())
                    },
                )
            }
        }
        Text("Chế độ che", style = MaterialTheme.typography.labelLarge)
        MaskModeChips(vm)

        vm.autoMessage?.let { Text(it, color = MaterialTheme.colorScheme.primary, style = MaterialTheme.typography.bodySmall) }

        // ---- Trạng thái quyền
        Text("Quyền", style = MaterialTheme.typography.labelLarge)
        PermissionRow(
            ok = p.media == MediaAccess.FULL,
            title = "Đọc ảnh: " + when (p.media) {
                MediaAccess.FULL -> "cho phép tất cả"
                MediaAccess.PARTIAL -> "chỉ vài ảnh (phải chọn \"Cho phép tất cả\")"
                MediaAccess.NONE -> "chưa cấp"
            },
            action = if (p.media == MediaAccess.FULL) null else "Cài đặt" to { open(context, PermissionStatus.appDetailsIntent(context)) },
        )
        PermissionRow(
            ok = p.notifications,
            title = "Thông báo: " + if (p.notifications) "bật" else "tắt (sẽ không báo kết quả)",
            action = if (p.notifications) null else "Cài đặt" to { open(context, PermissionStatus.appDetailsIntent(context)) },
        )
        PermissionRow(
            ok = p.manageMedia,
            title = when {
                !p.manageMediaSupported -> "Quản lý phương tiện: máy không hỗ trợ (Android < 12) → lưu bản che riêng"
                p.manageMedia -> "Quản lý phương tiện: đã bật (xoá ảnh gốc không cần hỏi)"
                else -> "Quản lý phương tiện: chưa bật → chỉ lưu bản che riêng vào Ảnh › SafeScreen"
            },
            action = if (p.manageMedia || !p.manageMediaSupported) null
            else "Bật" to { MediaWriter.manageMediaSettingsIntent(context)?.let { open(context, it) } },
        )
        if (p.manageMediaSupported) PermissionRow(
            ok = p.overlay,
            // Android 15+: service chạy nền chỉ mở được hộp thoại xoá của hệ thống khi app có quyền này
            title = "Hiển thị trên ứng dụng khác: " + if (p.overlay) "đã bật" else "chưa bật → chỉ lưu bản che riêng",
            action = if (p.overlay) null else "Bật" to { open(context, PermissionStatus.overlaySettingsIntent(context)) },
        )
        Text(
            if (p.canReplaceOriginal) "→ Ảnh chụp màn hình sẽ được THAY bằng bản đã che (bản gốc cất trong app, khôi phục được)."
            else "→ Ảnh gốc giữ nguyên trong Screenshots, bản đã che lưu riêng ở Ảnh › SafeScreen.",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.primary,
        )
        PermissionRow(
            ok = p.batteryUnrestricted,
            title = "Pin: " + if (p.batteryUnrestricted) "không hạn chế" else "đang bị tối ưu hoá (hệ thống có thể tắt bảo vệ)",
            action = if (p.batteryUnrestricted) null else "Cài đặt" to { open(context, PermissionStatus.appDetailsIntent(context)) },
        )
        if (!p.batteryUnrestricted) SamsungBatteryGuide()

        // ---- Kết quả
        HorizontalDivider()
        val sel = vm.selected
        if (sel != null) RecordDetail(vm, sel)
        Text("Ảnh đã che gần đây (${vm.records.size})", style = MaterialTheme.typography.labelLarge)
        if (vm.records.isEmpty()) Text("Chưa có", style = MaterialTheme.typography.bodySmall)
        for (r in vm.records) RecordRow(r, selected = r.mediaId == sel?.mediaId) { vm.select(r) }
    }
}

@Composable
private fun PermissionRow(ok: Boolean, title: String, action: Pair<String, () -> Unit>?) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Text(if (ok) "✓" else "✗", color = if (ok) Color(0xFF2E7D32) else Color(0xFFC62828), fontWeight = FontWeight.Bold)
        Text(title, Modifier.weight(1f).padding(horizontal = 8.dp), style = MaterialTheme.typography.bodySmall)
        if (action != null) TextButton(onClick = action.second) { Text(action.first) }
    }
}

/** Samsung (One UI) hay tắt app chạy nền rất mạnh tay → phải tự tay cho SafeScreen "không bao giờ ngủ" */
@Composable
private fun SamsungBatteryGuide() {
    var expanded by rememberSaveable { mutableStateOf(false) }
    Card(Modifier.fillMaxWidth()) {
        Column(Modifier.padding(12.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text("Hướng dẫn tắt tối ưu hoá pin (Samsung)", Modifier.weight(1f), style = MaterialTheme.typography.bodyMedium)
                TextButton(onClick = { expanded = !expanded }) { Text(if (expanded) "Ẩn" else "Xem") }
            }
            if (expanded) Text(
                """
                1. Cài đặt › Ứng dụng › SafeScreen › Pin › chọn "Không hạn chế".
                2. Cài đặt › Pin (hoặc Chăm sóc thiết bị › Pin) › Giới hạn sử dụng nền:
                   • Tắt "Đưa ứng dụng không dùng vào trạng thái ngủ".
                   • Mục "Ứng dụng không bao giờ ngủ" › dấu + › chọn SafeScreen.
                3. Mở Đa nhiệm (nút ô vuông), bấm vào biểu tượng SafeScreen › "Khoá ứng dụng này" để không bị đóng khi dọn RAM.
                4. Nếu có "Tự động tối ưu hoá hàng ngày" (Chăm sóc thiết bị) thì tắt, vì nó khởi động lại máy và đóng app.
                Tên mục có thể khác chút tuỳ phiên bản One UI.
                """.trimIndent(),
                style = MaterialTheme.typography.bodySmall,
            )
        }
    }
}

private val timeFormat = SimpleDateFormat("HH:mm:ss dd/MM", Locale.ROOT)

@Composable
private fun RecordRow(r: AutoRecord, selected: Boolean, onClick: () -> Unit) {
    val status = when {
        r.restored -> "đã khôi phục gốc"
        r.overwritten -> "đã che lên ảnh gốc"
        else -> "đã lưu bản che riêng"
    }
    Column(
        Modifier
            .fillMaxWidth()
            .clickable(onClick = onClick)
            .padding(vertical = 4.dp)
    ) {
        Text(
            r.displayName,
            style = MaterialTheme.typography.bodyMedium,
            fontWeight = if (selected) FontWeight.Bold else FontWeight.Normal,
        )
        Text(
            "${timeFormat.format(Date(r.createdAtMs))} · ${r.labels.size} vùng: ${r.labelSummary} · $status" +
                (r.savedToDoneMs?.let { " · ${"%.1f".format(it / 1000.0)} s" } ?: ""),
            style = MaterialTheme.typography.bodySmall,
        )
    }
}

/** Ảnh đang chọn (mở từ thông báo hoặc bấm vào danh sách): ảnh ĐÃ CHE + nút khôi phục */
@Composable
private fun RecordDetail(vm: MainViewModel, r: AutoRecord) {
    Card(Modifier.fillMaxWidth()) {
        Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text(r.displayName, style = MaterialTheme.typography.titleSmall)
            Text("${r.labels.size} vùng nhạy cảm: ${r.labelSummary}", style = MaterialTheme.typography.bodySmall)
            r.savedToDoneMs?.let {
                Text("Từ lúc lưu ảnh đến lúc che xong: ${"%.2f".format(it / 1000.0)} s", style = MaterialTheme.typography.bodySmall)
            }
            when {
                r.restored -> Text("Ảnh gốc đã được khôi phục.", style = MaterialTheme.typography.bodySmall)
                else -> vm.selectedImage?.let {
                    Image(
                        bitmap = it.asImageBitmap(),
                        contentDescription = "Ảnh đã che",
                        contentScale = ContentScale.Fit,
                        modifier = Modifier.fillMaxWidth().heightIn(max = 480.dp),
                    )
                }
            }
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                if (r.overwritten && !r.restored) Button(onClick = { vm.restore(r) }) { Text("Khôi phục ảnh gốc") }
                OutlinedButton(onClick = { vm.select(null) }) { Text("Đóng") }
            }
        }
    }
}

private fun open(context: Context, intent: Intent) {
    context.startActivity(intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
}
