package com.safescreen.auto

import android.content.Context
import android.net.Uri
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.boolean
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.long
import kotlinx.serialization.json.put
import java.io.File

/** Kết quả che tự động 1 ảnh chụp màn hình. */
data class AutoRecord(
    val mediaId: Long,
    val uri: String,                 // ảnh chụp màn hình trong MediaStore
    val displayName: String,
    val labels: List<String>,        // loại của từng vùng che (có thể lặp: NAME, NAME, PHONE)
    val maskMode: String,
    val method: String,              // METHOD_IN_PLACE / METHOD_REPLACED: ảnh gốc đã bị thay; METHOD_COPY: lưu bản che riêng
    val maskedCopyUri: String?,      // bản che riêng (khi không ghi đè được)
    val restored: Boolean,           // đã khôi phục ảnh gốc
    val createdAtMs: Long,
    val savedToDoneMs: Long?,        // từ lúc ảnh chụp màn hình được lưu → ghi đè xong
    val detectToDoneMs: Long,        // từ lúc service phát hiện ảnh → ghi đè xong
) {
    /** "NAME ×2, PHONE" */
    val labelSummary: String
        get() = labels.groupingBy { it }.eachCount().entries.sortedByDescending { it.value }
            .joinToString(", ") { (l, n) -> if (n > 1) "${labelVi(l)} ×$n" else labelVi(l) }

    /** true = ảnh trong thư viện đã là bản che (người dùng không còn thấy bản gốc) */
    val overwritten: Boolean get() = method != METHOD_COPY

    val displayUri: Uri get() = Uri.parse(if (overwritten) uri else maskedCopyUri ?: uri)

    companion object {
        const val METHOD_IN_PLACE = "in_place"   // ghi đè đúng file cũ (khi app đã có quyền ghi file đó)
        const val METHOD_REPLACED = "replaced"   // xoá ảnh gốc + tạo ảnh đã che cùng tên/thư mục (nhờ MANAGE_MEDIA)
        const val METHOD_COPY = "copy"           // không thay được ảnh gốc → lưu bản che riêng vào Pictures/SafeScreen

        fun labelVi(label: String) = when (label) {
            "NAME" -> "tên"; "PHONE" -> "SĐT"; "EMAIL" -> "email"; "ADDRESS" -> "địa chỉ"; "DOB" -> "ngày sinh"
            "CCCD" -> "CCCD"; "ACCOUNT" -> "số TK"; "MONEY" -> "số tiền"
            else -> label
        }
    }
}

/**
 * Lưu kết quả che tự động trong bộ nhớ RIÊNG của app (filesDir — app khác và người dùng không đọc được):
 *   auto/records.json      danh sách kết quả (giữ 50 ảnh gần nhất)
 *   auto/originals/<id>    bản gốc trước khi ghi đè (để khôi phục)
 *   auto/latency_log.csv   thời gian từng bước, để làm số liệu báo cáo
 */
class AutoMaskStore(context: Context) {

    private val dir = File(context.applicationContext.filesDir, "auto").apply { mkdirs() }
    private val originals = File(dir, "originals").apply { mkdirs() }
    private val recordsFile = File(dir, "records.json")
    val latencyLog = File(dir, "latency_log.csv")

    fun backupFile(mediaId: Long) = File(originals, mediaId.toString())

    fun load(): List<AutoRecord> = synchronized(LOCK) {
        if (!recordsFile.isFile) return emptyList()
        // Bản ghi hỏng / định dạng cũ thì bỏ qua, không để 1 bản ghi lỗi làm hỏng cả danh sách
        Json.parseToJsonElement(recordsFile.readText()).jsonArray.mapNotNull { runCatching { decode(it.jsonObject) }.getOrNull() }
    }

    fun get(mediaId: Long) = load().firstOrNull { it.mediaId == mediaId }

    fun upsert(r: AutoRecord) = synchronized(LOCK) {
        val all = listOf(r) + load().filter { it.mediaId != r.mediaId }
        val keep = all.take(MAX_RECORDS)
        all.drop(MAX_RECORDS).forEach { backupFile(it.mediaId).delete() }  // bản gốc của ảnh cũ bị bỏ cũng xoá luôn
        recordsFile.writeText(JsonArray(keep.map(::encode)).toString())
    }

    fun appendLatency(line: List<Any?>) = synchronized(LOCK) {
        if (!latencyLog.isFile) latencyLog.writeText(LATENCY_HEADER + "\n")
        latencyLog.appendText(line.joinToString(",") { it?.toString()?.replace(",", " ") ?: "" } + "\n")
    }

    private fun encode(r: AutoRecord): JsonObject = buildJsonObject {
        put("mediaId", r.mediaId); put("uri", r.uri); put("displayName", r.displayName)
        put("labels", buildJsonArray { r.labels.forEach { add(JsonPrimitive(it)) } })
        put("maskMode", r.maskMode); put("method", r.method); put("maskedCopyUri", r.maskedCopyUri)
        put("restored", r.restored); put("createdAtMs", r.createdAtMs)
        put("savedToDoneMs", r.savedToDoneMs); put("detectToDoneMs", r.detectToDoneMs)
    }

    private fun decode(o: JsonObject) = AutoRecord(
        mediaId = o.getValue("mediaId").jsonPrimitive.long,
        uri = o.getValue("uri").jsonPrimitive.content,
        displayName = o.getValue("displayName").jsonPrimitive.content,
        labels = o.getValue("labels").jsonArray.map { it.jsonPrimitive.content },
        maskMode = o.getValue("maskMode").jsonPrimitive.content,
        method = o.getValue("method").jsonPrimitive.content,
        maskedCopyUri = o["maskedCopyUri"]?.jsonPrimitive?.takeIf { it.isString }?.content,
        restored = o.getValue("restored").jsonPrimitive.boolean,
        createdAtMs = o.getValue("createdAtMs").jsonPrimitive.long,
        savedToDoneMs = o["savedToDoneMs"]?.jsonPrimitive?.takeIf { it.content != "null" }?.long,
        detectToDoneMs = o.getValue("detectToDoneMs").jsonPrimitive.long,
    )

    companion object {
        private val LOCK = Any()
        private const val MAX_RECORDS = 50
        const val LATENCY_HEADER = "media_id,display_name,width,height,saved_at_ms,saved_at_source,detected_at_ms," +
            "decode_ms,ocr_ms,crf_ms,mask_ms,backup_ms,write_ms,regions,outcome,saved_to_done_ms,detect_to_done_ms"
    }
}
