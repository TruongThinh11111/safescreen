package com.safescreen.engine

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.boolean
import kotlinx.serialization.json.booleanOrNull
import kotlinx.serialization.json.double
import kotlinx.serialization.json.doubleOrNull
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import java.time.LocalDate
import java.util.Locale
import kotlin.math.abs
import kotlin.math.max

/**
 * So bản Kotlin với models/golden_test.json (do ml/safescreen/export_android.py tạo, đã kiểm chứng khớp crfsuite).
 * Dùng chung cho JVM unit test và instrumented test trên máy Android.
 *
 *   - bằng TUYỆT ĐỐI: văn bản, kết quả regex, đặc trưng, thuộc tính, nhãn Viterbi, quyết định che, vùng che,
 *     thứ tự đọc (khi đầu vào bị xáo trộn)
 *   - sai khác < tolerance (1e-6): xác suất biên, p_sensitive, F
 */
object GoldenVerifier {

    class Report(
        val samples: Int,
        val tokens: Int,
        val errors: List<String>,
        val totalErrors: Int,
        val maxMarginalDiff: Double,
        val maxFDiff: Double,
        val millisPerSample: Double,
        /** Token đầu tiên bị lệch + MỌI chỗ khác của nó (null nếu không token nào lệch) */
        val firstMismatch: String? = null,
    ) {
        val passed get() = totalErrors == 0

        override fun toString() = buildString {
            append("${if (passed) "ĐẠT" else "KHÔNG ĐẠT"}: $samples mẫu, $tokens token, $totalErrors lỗi, ")
            append(String.format(Locale.ROOT, "max|Δmarginal|=%.3e, max|ΔF|=%.3e, %.1f ms/ảnh",
                maxMarginalDiff, maxFDiff, millisPerSample))
            firstMismatch?.let { append("\n\n>>> TOKEN LỆCH ĐẦU TIÊN: ").append(it).append("\n\nDanh sách lỗi:") }
            errors.forEach { append("\n  - ").append(it) }
        }
    }

    fun verify(engine: SafeScreenEngine, goldenJson: String, maxListedErrors: Int = 30): Report {
        val golden = Json.parseToJsonElement(goldenJson).jsonObject
        val today = LocalDate.parse(golden.getValue("reference_date").jsonPrimitive.content)
        val tol = golden.getValue("tolerance").jsonPrimitive.double
        val errors = mutableListOf<String>()
        var totalErrors = 0
        fun fail(msg: String) {
            totalErrors++
            if (errors.size < maxListedErrors) errors.add(msg)
        }

        var nTokens = 0
        var maxMarg = 0.0
        var maxF = 0.0
        var nanos = 0L
        var firstMismatch: String? = null
        val samples = golden.getValue("samples").jsonArray
        for (sEl in samples) {
            val s = sEl.jsonObject
            val id = s.getValue("id").jsonPrimitive.content
            val gTokens = s.getValue("tokens").jsonArray.map { it.jsonObject }
            val words = gTokens.map {
                OcrWord(
                    text = it.getValue("text").jsonPrimitive.content,
                    bbox = it.getValue("bbox").jsonArray.map { v -> v.jsonPrimitive.int },
                    conf = it.getValue("conf").jsonPrimitive.double,
                    line = it.getValue("line").jsonPrimitive.int,
                )
            }

            val t0 = System.nanoTime()
            val out = engine.process(words, today)
            nanos += System.nanoTime() - t0

            // ---- văn bản + regex
            val (text, _) = engine.wordsToText(out.tokens.map { it.word })
            if (text != s.getValue("text").jsonPrimitive.content) fail("$id: văn bản ghép khác")
            val gotRx = engine.findEntities(text, today)
            val expRx = s.getValue("regex_matches").jsonArray.map {
                val o = it.jsonObject
                RegexMatch(o.getValue("start").jsonPrimitive.int, o.getValue("end").jsonPrimitive.int,
                    o.getValue("label").jsonPrimitive.content, o.getValue("score").jsonPrimitive.double,
                    o.getValue("ctx").jsonPrimitive.boolean)
            }
            if (gotRx != expRx) fail("$id: regex khác\n    Kotlin=$gotRx\n    Python=$expRx")

            // ---- từng token
            if (out.tokens.size != gTokens.size) fail("$id: số token ${out.tokens.size} ≠ ${gTokens.size}")
            for ((i, pair) in out.tokens.zip(gTokens).withIndex()) {
                val (got, exp) = pair
                nTokens++
                val problems = mutableListOf<String>()  // mọi chỗ khác của token này

                val featDiffs = diffFeatures(got.features, exp.getValue("features").jsonObject)
                featDiffs.forEach { problems += "feature $it" }
                val expAttrs = exp.getValue("attributes").jsonObject
                if (got.attributes.keys != expAttrs.keys ||
                    got.attributes.any { (k, v) -> v != expAttrs.getValue(k).jsonPrimitive.double }
                ) problems += "thuộc tính (feature đã đổi sang số) khác"
                val expLabel = exp.getValue("label").jsonPrimitive.content
                if (got.label != expLabel) problems += "nhãn Kotlin=${got.label}, Python=$expLabel"
                val expMarg = exp.getValue("marginals").jsonObject
                engine.labels.forEachIndexed { j, lab ->
                    val d = abs(got.marginals[j] - expMarg.getValue(lab).jsonPrimitive.double)
                    maxMarg = max(maxMarg, d)
                    if (d >= tol) problems += "marginal[$lab] lệch $d"
                }
                val dF = abs(got.f - exp.getValue("F").jsonPrimitive.double)
                maxF = max(maxF, dF)
                if (dF >= tol) problems += "F lệch $dF"
                val expMask = exp.getValue("mask").jsonPrimitive.boolean
                if (got.mask != expMask) problems += "quyết định che Kotlin=${got.mask}, Python=$expMask"

                if (problems.isEmpty()) continue
                problems.forEach { fail("$id/#$i '${got.word.text}': $it") }
                // Lỗi ở 1 token thường kéo lỗi sang token bên cạnh (feature dùng cả từ trước/sau, Viterbi
                // dùng cả câu) → token lệch ĐẦU TIÊN thường là gốc của vấn đề, in đầy đủ để sửa nó trước.
                if (firstMismatch == null) {
                    firstMismatch = buildString {
                        append("mẫu $id, token #$i '${got.word.text}' bbox=${got.word.bbox}")
                        problems.forEach { append("\n    ").append(it) }
                    }
                }
            }

            // ---- vùng che
            val expRegions = s.getValue("regions").jsonArray.map {
                val o = it.jsonObject
                o.getValue("bbox").jsonArray.map { v -> v.jsonPrimitive.int } to
                    o.getValue("labels").jsonArray.map { v -> v.jsonPrimitive.content }
            }
            if (out.regions.map { it.bbox to it.labels } != expRegions) fail("$id: vùng che khác")

            // ---- thứ tự đọc: xáo trộn đầu vào (bỏ line) rồi xếp lại phải ra đúng như cũ
            val shuffled = words.map { it.copy(line = -1) }.shuffled(java.util.Random(words.size.toLong()))
            if (engine.sortReadingOrder(shuffled) != words) fail("$id: sortReadingOrder khác")
        }
        return Report(samples.size, nTokens, errors, totalErrors, maxMarg, maxF, nanos / 1e6 / samples.size, firstMismatch)
    }

    /** Liệt kê MỌI feature khác nhau (rỗng = giống hệt). Số so theo giá trị (1 == 1.0). */
    private fun diffFeatures(got: Map<String, Any>, exp: JsonObject): List<String> {
        val out = mutableListOf<String>()
        for (k in exp.keys - got.keys) out += "$k: chỉ Python có (=${exp.getValue(k)})"
        for (k in got.keys - exp.keys) out += "$k: chỉ Kotlin có (=${got.getValue(k)})"
        for ((k, v) in got) {
            val j = exp[k]?.jsonPrimitive ?: continue
            if (!sameValue(v, j)) out += "$k: Kotlin=$v, Python=$j"
        }
        return out
    }

    private fun sameValue(v: Any, j: JsonPrimitive): Boolean = when (v) {
        is String -> j.isString && j.content == v
        is Boolean -> !j.isString && j.booleanOrNull == v
        is Number -> !j.isString && j.booleanOrNull == null && j.doubleOrNull == v.toDouble()
        else -> false
    }
}
