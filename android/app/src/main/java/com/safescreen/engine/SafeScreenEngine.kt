package com.safescreen.engine

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.double
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import java.time.DateTimeException
import java.time.LocalDate
import java.util.regex.Matcher
import java.util.regex.Pattern
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min

/*
 * SafeScreenEngine — toàn bộ phần suy luận SAU OCR, port 1–1 từ ml/safescreen/crf_pure.py.
 * Chỉ đọc 2 file do export_android.py tạo: crf_model.json + config.json.
 *
 * Luồng xử lý 1 ảnh (danh sách từ OCR):
 *   0. sortReadingOrder  1. wordsToText  2. regex  3. đặc trưng CRF  4. Viterbi + marginal
 *   5. hàm rủi ro F, quyết định che, gộp vùng
 */

/** 1 từ OCR. bbox = [x0, y0, x1, y1] theo pixel ảnh gốc; conf 0–1; line = số hàng (sau sortReadingOrder). */
data class OcrWord(val text: String, val bbox: List<Int>, val conf: Double, val line: Int = 0)

data class RegexMatch(val start: Int, val end: Int, val label: String, val score: Double, val ctx: Boolean)

/** Kết quả regex + từ khóa ngữ cảnh cho 1 từ. */
data class Annotation(val rx: String, val rxScore: Double, val rxCtx: Boolean, val kw: List<String>)

class TokenResult(
    val word: OcrWord,
    val annotation: Annotation,
    val features: Map<String, Any>,        // giá trị: String / Boolean / Int / Double
    val attributes: Map<String, Double>,   // thuộc tính crfsuite
    val label: String,                     // nhãn Viterbi
    val marginals: DoubleArray,            // theo thứ tự SafeScreenEngine.labels
    val pSensitive: Double,                // 1 − P(O)
    val riskX: DoubleArray,                // [p_crf, r_regex, c_ocr, ctx]
    val f: Double,                         // điểm rủi ro F
    val mask: Boolean,                     // F ≥ τ
)

data class Region(val bbox: List<Int>, val labels: List<String>, val f: Double, val text: String)

class EngineResult(val tokens: List<TokenResult>, val regions: List<Region>)

class SafeScreenEngine(modelJson: String, configJson: String) {

    // ---------------- CRF ----------------
    val labels: List<String>
    private val labelIndex: Map<String, Int>
    private val state: Map<String, List<Pair<Int, Double>>>   // thuộc tính → [(chỉ số nhãn, trọng số)]
    private val trans: Array<DoubleArray>                      // trans[i][j]: nhãn i → nhãn j (thiếu = 0)
    private val oIndex: Int

    // ---------------- regex ----------------
    private val rx: Map<String, Pattern>
    private val tight: Double
    private val loose: Double
    private val window: Int
    private val rxKeywords: Map<String, List<String>>
    private val dobNegative: List<String>
    private val provinceCodes: Set<String>
    private val mobilePrefixes: Set<String>
    private val hotlinePrefixes: List<String>
    private val priority: Map<String, Int>
    private val ruleOrder: List<String>
    private val lookalikes: Map<Char, Char>
    private val dashes: Map<Char, Char>

    // ---------------- đặc trưng CRF ----------------
    private val crfKeywords: List<Pair<String, List<String>>>   // giữ thứ tự khóa như Python dict
    private val kwWindow: Int
    private val punct: Pattern
    private val spaces: Pattern
    private val repeat: Pattern
    private val lenMax: Int
    private val shapeFullMax: Int
    private val offsets: List<Int>

    // ---------------- hàm rủi ro + che ----------------
    val lambdas: DoubleArray
    val beta: Double
    val tau: Double
    private val ctxWindow: Int
    private val riskKeywords: List<String>
    private val mergeVOverlap: Double
    private val mergeHGap: Double
    /** Nới rộng mỗi vùng che thêm bao nhiêu px mỗi phía (config mask.pad_px) */
    val maskPadPx: Int
    /** Chế độ che mặc định: "black_box" | "pixelate" | "gaussian_blur" (config mask.default_mode) */
    val defaultMaskMode: String

    init {
        val m = Json.parseToJsonElement(modelJson).jsonObject
        val c = Json.parseToJsonElement(configJson).jsonObject

        labels = m.getValue("labels").strList()
        labelIndex = labels.withIndex().associate { it.value to it.index }
        // JSON được đọc chính xác: số thực in dạng round-trip ở Python → String.toDouble() ra đúng float64
        state = m.getValue("state_features").jsonObject.mapValues { (_, v) ->
            v.jsonObject.map { (lab, w) -> labelIndex.getValue(lab) to w.jsonPrimitive.double }
        }
        val tf = m.getValue("transition_features").jsonObject
        trans = Array(labels.size) { i ->
            DoubleArray(labels.size) { j -> tf[labels[i]]?.jsonObject?.get(labels[j])?.jsonPrimitive?.double ?: 0.0 }
        }
        oIndex = labelIndex.getValue("O")

        val rc = c.getValue("regex").jsonObject
        rx = rc.getValue("patterns").jsonObject.mapValues { (_, d) ->
            compileRegex(d.jsonObject.getValue("pattern").jsonPrimitive.content, d.jsonObject.getValue("flags").strList())
        }
        tight = rc.getValue("tight").jsonPrimitive.double
        loose = rc.getValue("loose").jsonPrimitive.double
        window = rc.getValue("window").jsonPrimitive.int
        rxKeywords = rc.getValue("keywords").jsonObject.mapValues { it.value.strList() }
        dobNegative = rc.getValue("dob_negative").strList()
        provinceCodes = rc.getValue("province_codes").strList().toSet()
        mobilePrefixes = rc.getValue("mobile_prefixes").strList().toSet()
        hotlinePrefixes = rc.getValue("hotline_prefixes").strList()
        priority = rc.getValue("priority").jsonObject.mapValues { it.value.jsonPrimitive.int }
        ruleOrder = rc.getValue("rule_order").strList()
        lookalikes = rc.getValue("digit_lookalikes").charMap()
        dashes = rc.getValue("dashes").charMap()

        val fc = c.getValue("crf_features").jsonObject
        crfKeywords = fc.getValue("keywords").jsonObject.map { (k, v) -> k to v.strList() }
        kwWindow = fc.getValue("kw_window").jsonPrimitive.int
        val fp = fc.getValue("patterns").jsonObject
        punct = compileRegex(fp.getValue("punct_to_space").jsonPrimitive.content)
        spaces = compileRegex(fp.getValue("multi_space").jsonPrimitive.content)
        repeat = compileRegex(fp.getValue("repeated_char").jsonPrimitive.content)
        lenMax = fc.getValue("len_max").jsonPrimitive.int
        shapeFullMax = fc.getValue("shape_full_max").jsonPrimitive.int
        offsets = fc.getValue("neighbor_offsets").jsonArray.map { it.jsonPrimitive.int }

        val r = c.getValue("risk").jsonObject
        lambdas = r.getValue("lambdas").jsonArray.map { it.jsonPrimitive.double }.toDoubleArray()
        beta = r.getValue("beta").jsonPrimitive.double
        tau = r.getValue("tau").jsonPrimitive.double
        ctxWindow = r.getValue("ctx_window").jsonPrimitive.int
        riskKeywords = r.getValue("keywords").strList()
        val mc = c.getValue("mask").jsonObject
        mergeVOverlap = mc.getValue("merge_min_v_overlap").jsonPrimitive.double
        mergeHGap = mc.getValue("merge_max_h_gap").jsonPrimitive.double
        maskPadPx = mc.getValue("pad_px").jsonPrimitive.int
        defaultMaskMode = mc.getValue("default_mode").jsonPrimitive.content
    }

    // =================================================================
    // 0–1. Thứ tự đọc + ghép văn bản
    // =================================================================

    private class Row(val y0: Int, val y1: Int, val words: MutableList<OcrWord>)

    /**
     * Gom từ theo hàng ngang. Khóa sắp xếp ĐẦY ĐỦ (tâm dọc kiểu Double, x0, y0, x1, y1, chữ) để kết quả không
     * phụ thuộc thứ tự Tesseract trả về.
     */
    fun sortReadingOrder(words: List<OcrWord>): List<OcrWord> {
        val rowKey = compareBy<OcrWord>({ (it.bbox[1] + it.bbox[3]) / 2.0 }, { it.bbox[0] }, { it.bbox[1] },
            { it.bbox[2] }, { it.bbox[3] }, { it.text })
        val rows = mutableListOf<Row>()
        for (w in words.sortedWith(rowKey)) {
            val cy = (w.bbox[1] + w.bbox[3]) / 2.0
            val last = rows.lastOrNull()
            if (last != null && last.y0 <= cy && cy <= last.y1) last.words.add(w)
            else rows.add(Row(w.bbox[1], w.bbox[3], mutableListOf(w)))
        }
        val inRow = compareBy<OcrWord>({ it.bbox[0] }, { it.bbox[1] }, { it.bbox[2] }, { it.bbox[3] }, { it.text })
        return rows.flatMapIndexed { i, row -> row.words.sortedWith(inRow).map { it.copy(line = i) } }
    }

    /** Cùng dòng nối ' ', khác dòng nối '\n'. Trả về văn bản + [start, end) của từng từ. */
    fun wordsToText(words: List<OcrWord>): Pair<String, List<IntArray>> {
        val sb = StringBuilder()
        val spans = ArrayList<IntArray>(words.size)
        var prevLine: Int? = null
        for (w in words) {
            if (prevLine != null) sb.append(if (w.line == prevLine) ' ' else '\n')
            spans.add(intArrayOf(sb.length, sb.length + w.text.length))
            sb.append(w.text)
            prevLine = w.line
        }
        return sb.toString() to spans
    }

    // =================================================================
    // 2. Regex
    // =================================================================

    /** Gạch dài → '-'; trong cụm số: O/o → 0, l/I/| → 1. Độ dài chuỗi KHÔNG đổi. */
    fun normalizeOcr(text: String): String {
        val t = buildString(text.length) { text.forEach { append(dashes[it] ?: it) } }
        return replaceEach(rx.getValue("number_run"), t) { s ->
            if (hasDigit(s)) buildString(s.length) { s.forEach { append(lookalikes[it] ?: it) } } else s
        }
    }

    private fun lastPos(window: String, keywords: List<String>): Int = keywords.maxOfOrNull { window.lastIndexOf(it) } ?: -1

    private fun windowBefore(t: String, start: Int): String = fold(t.substring(max(0, start - window), start))

    private fun hasKeyword(t: String, start: Int, label: String): Boolean =
        lastPos(windowBefore(t, start), rxKeywords.getValue(label)) >= 0

    private fun matchCccd(t: String): List<RegexMatch> {
        val out = mutableListOf<RegexMatch>()
        rx.getValue("cccd").forEachMatch(t) { m ->
            val g = m.group()
            val digits = rx.getValue("non_digit").matcher(g).replaceAll("")
            val structured = digits.substring(0, 3) in provinceCodes && digits[3] in "0123"
            val ctx = hasKeyword(t, m.start(), "CCCD")
            if (!structured && !ctx) return@forEachMatch
            val score = if (structured && g == digits) tight else loose
            out.add(RegexMatch(m.start(), m.end(), "CCCD", score, ctx))
        }
        return out
    }

    /** [P9] `today` truyền vào để loại ngày tương lai (không gọi LocalDate.now() rải rác). */
    private fun matchDob(t: String, today: LocalDate): List<RegexMatch> {
        val out = mutableListOf<RegexMatch>()
        rx.getValue("date").forEachMatch(t) { m ->
            val y = m.group(4).toInt()
            val dob = try {
                LocalDate.of(y, m.group(3).toInt(), m.group(1).toInt())
            } catch (e: DateTimeException) {
                return@forEachMatch
            }
            if (!(1900 <= y && !dob.isAfter(today))) return@forEachMatch
            val w = windowBefore(t, m.start())
            val pos = lastPos(w, rxKeywords.getValue("DOB"))
            val neg = lastPos(w, dobNegative)
            if (neg > pos) return@forEachMatch
            val ctx = pos >= 0
            val score = if (ctx && ' ' !in m.group()) tight else loose
            out.add(RegexMatch(m.start(), m.end(), "DOB", score, ctx))
        }
        return out
    }

    private fun matchPhone(t: String): List<RegexMatch> {
        val out = mutableListOf<RegexMatch>()
        rx.getValue("phone").forEachMatch(t) { m ->
            val digits = rx.getValue("non_digit").matcher(m.group()).replaceAll("")
            val local = if (digits.startsWith("84")) "0" + digits.substring(2) else digits
            val isTight = rx.getValue("phone_canonical").matcher(m.group().replace('\n', ' ')).matches() &&
                local.substring(0, 3) in mobilePrefixes
            out.add(RegexMatch(m.start(), m.end(), "PHONE", if (isTight) tight else loose,
                hasKeyword(t, m.start(), "PHONE")))
        }
        return out
    }

    private fun matchEmail(t: String): List<RegexMatch> {
        val out = mutableListOf<RegexMatch>()
        rx.getValue("email_tight").forEachMatch(t) { m ->
            out.add(RegexMatch(m.start(), m.end(), "EMAIL", tight, hasKeyword(t, m.start(), "EMAIL")))
        }
        val tightOnes = out.toList()
        rx.getValue("email_loose").forEachMatch(t) { m ->
            if (tightOnes.none { m.start() < it.end && it.start < m.end() }) {
                out.add(RegexMatch(m.start(), m.end(), "EMAIL", loose, hasKeyword(t, m.start(), "EMAIL")))
            }
        }
        return out
    }

    private fun matchMoney(t: String): List<RegexMatch> {
        val out = mutableListOf<RegexMatch>()
        val tg = rx.getValue("thousand_group")
        rx.getValue("money").forEachMatch(t) { m ->
            val num = m.group("num")
            val score = when {
                m.group("cur") != null -> tight
                tg.matcher(num).find() && (num[0] in "+-" || countMatches(tg, num) >= 2) -> loose
                else -> return@forEachMatch
            }
            out.add(RegexMatch(m.start(), m.end(), "MONEY", score, hasKeyword(t, m.start(), "MONEY")))
        }
        rx.getValue("money_prefix").forEachMatch(t) { m ->
            out.add(RegexMatch(m.start(), m.end(), "MONEY", tight, hasKeyword(t, m.start(), "MONEY")))
        }
        rx.getValue("money_short").forEachMatch(t) { m ->
            val score = if (m.group("unit").lowercase() in setOf("tr", "k")) loose else tight
            out.add(RegexMatch(m.start(), m.end(), "MONEY", score, hasKeyword(t, m.start(), "MONEY")))
        }
        return out
    }

    private fun matchAccount(t: String): List<RegexMatch> {
        val out = mutableListOf<RegexMatch>()
        rx.getValue("account").forEachMatch(t) { m ->
            val g = m.group()
            val groups = mutableListOf<IntArray>()   // [start, end) của từng nhóm số trong g
            rx.getValue("digit_run").forEachMatch(g) { d -> groups.add(intArrayOf(d.start(), d.end())) }
            // Số nhóm k lớn nhất: các nhóm trước nhóm cuối đều đúng 4 số, tổng 8–16 số
            var k = groups.size
            var digits = ""
            while (k > 0) {
                digits = groups.take(k).joinToString("") { g.substring(it[0], it[1]) }
                if (groups.take(k - 1).all { it[1] - it[0] == 4 } && digits.length in 8..16) break
                k--
            }
            if (k == 0 || hotlinePrefixes.any { digits.startsWith(it) }) return@forEachMatch
            val ctx = hasKeyword(t, m.start(), "ACCOUNT")
            out.add(RegexMatch(m.start(), m.start() + groups[k - 1][1], "ACCOUNT", if (ctx) tight else loose, ctx))
        }
        return out
    }

    /** [P8] sortedWith là sort ỔN ĐỊNH — giữ đúng thứ tự ứng viên khi các khóa bằng nhau. */
    fun resolve(matches: List<RegexMatch>): List<RegexMatch> {
        val ranked = matches.sortedWith(compareBy({ -it.score }, { if (it.ctx) -1 else 0 },
            { priority.getValue(it.label) }, { -(it.end - it.start) }))
        val kept = mutableListOf<RegexMatch>()
        for (m in ranked) if (kept.all { m.end <= it.start || it.end <= m.start }) kept.add(m)
        return kept.sortedBy { it.start }
    }

    fun findEntities(text: String, today: LocalDate): List<RegexMatch> {
        val norm = normalizeOcr(text)
        val cands = ruleOrder.flatMap { rule ->
            when (rule) {
                "CCCD" -> matchCccd(norm)
                "DOB" -> matchDob(norm, today)
                "PHONE" -> matchPhone(norm)
                "EMAIL" -> matchEmail(norm)
                "MONEY" -> matchMoney(norm)
                "ACCOUNT" -> matchAccount(norm)
                else -> error("luật lạ: $rule")
            }
        }
        // Phải sửa lỗi OCR mới khớp được → chỉ tính là khớp lỏng
        val adjusted = cands.map {
            if (norm.substring(it.start, it.end) != text.substring(it.start, it.end)) it.copy(score = loose) else it
        }
        return resolve(adjusted)
    }

    fun tagWords(words: List<OcrWord>, today: LocalDate): List<RegexMatch?> {
        val (text, spans) = wordsToText(words)
        val tags = arrayOfNulls<RegexMatch>(words.size)
        for (m in findEntities(text, today)) {
            spans.forEachIndexed { i, s -> if (s[0] < m.end && m.start < s[1]) tags[i] = m }
        }
        return tags.toList()
    }

    // =================================================================
    // 3. Đặc trưng CRF
    // =================================================================

    fun annotate(words: List<OcrWord>, today: LocalDate): List<Annotation> {
        val folded = words.map { fold(it.text) }
        return tagWords(words, today).mapIndexed { i, m ->
            var before = folded.subList(max(0, i - kwWindow), i).joinToString(" ")
            before = spaces.matcher(punct.matcher(before).replaceAll(" ")).replaceAll(" ")
            Annotation(
                rx = m?.label ?: "O",
                rxScore = m?.score ?: 0.0,
                rxCtx = m?.ctx ?: false,
                kw = crfKeywords.filter { (_, kws) -> kws.any { it in before } }.map { it.first },
            )
        }
    }

    private fun shortShape(w: String): String = repeat.matcher(wordShape(w)).replaceAll("$1")

    private fun tokenFeats(w: OcrWord, a: Annotation, p: String, into: MutableMap<String, Any>) {
        val t = w.text
        into[p + "lower"] = t.lowercase()          // [P3]
        into[p + "fold"] = fold(t)
        into[p + "istitle"] = pyIsTitle(t)
        into[p + "isupper"] = pyIsUpper(t)
        into[p + "isdigit"] = pyIsDigit(t)
        into[p + "shape"] = shortShape(t)
        into[p + "rx"] = a.rx
    }

    fun word2features(words: List<OcrWord>, ann: List<Annotation>, i: Int): Map<String, Any> {
        val w = words[i]
        val a = ann[i]
        val t = w.text
        val low = t.lowercase()
        val f = LinkedHashMap<String, Any>()
        f["bias"] = 1.0
        tokenFeats(w, a, "", f)
        f["len"] = min(cpLen(t), lenMax)            // [P4] số code point
        f["prefix2"] = cpPrefix(low, 2)
        f["prefix3"] = cpPrefix(low, 3)
        f["suffix2"] = cpSuffix(low, 2)
        f["suffix3"] = cpSuffix(low, 3)
        f["shape_full"] = cpPrefix(wordShape(t), shapeFullMax)
        f["has_diacritic"] = boDau(t) != t
        f["has_digit"] = hasDigit(t)
        f["has_at"] = '@' in t
        f["bol"] = i == 0 || words[i - 1].line != w.line
        f["eol"] = i == words.size - 1 || words[i + 1].line != w.line
        f["rx_tight"] = a.rxScore == tight
        f["rx_loose"] = a.rxScore == loose
        f["rx_ctx"] = a.rxCtx
        for (ent in a.kw) f["kw:$ent"] = true
        for (off in offsets) {
            val j = i + off
            val p = (if (off > 0) "+" else "") + "$off:"   // "-2:", "-1:", "+1:", "+2:"
            if (j in words.indices) {
                tokenFeats(words[j], ann[j], p, f)
                f[p + "same_line"] = words[j].line == w.line
            } else {
                f[p + (if (off < 0) "BOS" else "EOS")] = true
            }
        }
        return f
    }

    /**
     * [P6] Thuộc tính crfsuite: chuỗi → "khóa:giá_trị" = 1.0; bool/số → "khóa" = giá trị.
     * Khóa có thể chứa ':' ("-1:fold") → không bao giờ tách ngược tên thuộc tính theo ':'.
     */
    fun toAttributes(features: Map<String, Any>): Map<String, Double> {
        val attrs = LinkedHashMap<String, Double>()
        for ((k, v) in features) {
            when (v) {
                is String -> attrs["$k:$v"] = 1.0
                is Boolean -> attrs[k] = if (v) 1.0 else 0.0
                is Number -> attrs[k] = v.toDouble()
                else -> error("kiểu đặc trưng lạ: $k = $v")
            }
        }
        return attrs
    }

    // =================================================================
    // 4. Viterbi + forward–backward
    // =================================================================

    fun stateScores(attrs: Map<String, Double>): DoubleArray {
        val s = DoubleArray(labels.size)
        for ((a, v) in attrs) {
            if (v == 0.0) continue
            state[a]?.forEach { (lab, w) -> s[lab] += v * w }   // thuộc tính chưa gặp khi train → bỏ qua
        }
        return s
    }

    /** CRF bậc 1 của crfsuite: KHÔNG có trọng số bắt đầu/kết thúc chuỗi. Hoà điểm → chọn nhãn chỉ số nhỏ nhất. */
    fun viterbi(S: List<DoubleArray>): IntArray {
        val n = S.size
        val L = labels.size
        if (n == 0) return IntArray(0)
        var delta = S[0].copyOf()
        val back = Array(n) { IntArray(L) }
        for (t in 1 until n) {
            val next = DoubleArray(L)
            for (j in 0 until L) {
                var best = 0
                for (i in 1 until L) if (delta[i] + trans[i][j] > delta[best] + trans[best][j]) best = i
                back[t][j] = best
                next[j] = delta[best] + trans[best][j] + S[t][j]
            }
            delta = next
        }
        val path = IntArray(n)
        path[n - 1] = delta.indices.maxByOrNull { delta[it] }!!   // maxBy trả về phần tử lớn nhất ĐẦU TIÊN
        for (t in n - 1 downTo 1) path[t - 1] = back[t][path[t]]
        return path
    }

    private fun logSumExp(xs: DoubleArray): Double {
        val m = xs.max()
        var sum = 0.0
        for (x in xs) sum += exp(x - m)
        return m + ln(sum)
    }

    /** [P10] P(nhãn_t = j | cả chuỗi) = exp(α_t(j) + β_t(j) − log Z), tính trong không gian log. */
    fun marginals(S: List<DoubleArray>): List<DoubleArray> {
        val n = S.size
        val L = labels.size
        if (n == 0) return emptyList()
        val alpha = Array(n) { DoubleArray(L) }
        alpha[0] = S[0].copyOf()
        for (t in 1 until n) for (j in 0 until L) {
            alpha[t][j] = S[t][j] + logSumExp(DoubleArray(L) { i -> alpha[t - 1][i] + trans[i][j] })
        }
        val beta = Array(n) { DoubleArray(L) }
        for (t in n - 2 downTo 0) for (i in 0 until L) {
            beta[t][i] = logSumExp(DoubleArray(L) { j -> trans[i][j] + S[t + 1][j] + beta[t + 1][j] })
        }
        val logZ = logSumExp(alpha[n - 1])
        return List(n) { t -> DoubleArray(L) { j -> exp(alpha[t][j] + beta[t][j] - logZ) } }
    }

    // =================================================================
    // 5. Hàm rủi ro, quyết định che, gộp vùng
    // =================================================================

    fun riskInputs(words: List<OcrWord>, ann: List<Annotation>, pSensitive: DoubleArray): List<DoubleArray> {
        val folded = words.map { punct.matcher(fold(it.text)).replaceAll(" ") }
        return words.indices.map { i ->
            val win = spaces.matcher(
                folded.subList(max(0, i - ctxWindow), min(words.size, i + ctxWindow + 1)).joinToString(" ")
            ).replaceAll(" ")
            val ctx = riskKeywords.any { it in win }
            doubleArrayOf(pSensitive[i], ann[i].rxScore, words[i].conf.coerceIn(0.0, 1.0), if (ctx) 1.0 else 0.0)
        }
    }

    fun riskScore(x: DoubleArray): Double {
        var z = 0.0
        for (k in x.indices) z += lambdas[k] * x[k]
        return sigmoid(z + beta)
    }

    private fun sameLineAdjacent(a: List<Int>, b: List<Int>): Boolean {
        val ha = a[3] - a[1]
        val hb = b[3] - b[1]
        if (min(a[3], b[3]) - max(a[1], b[1]) < mergeVOverlap * min(ha, hb)) return false
        return max(a[0], b[0]) - min(a[2], b[2]) <= mergeHGap * max(ha, hb)
    }

    /** Gộp box liền nhau trên cùng dòng thành 1 vùng (lặp tới khi không gộp thêm được). */
    fun mergeRegions(items: List<Region>): List<Region> {
        val regions = items.toMutableList()
        var merged = true
        while (merged) {
            merged = false
            loop@ for (i in regions.indices) for (j in i + 1 until regions.size) {
                val a = regions[i]
                val b = regions[j]
                if (sameLineAdjacent(a.bbox, b.bbox)) {
                    val (first, second) = if (a.bbox[0] <= b.bbox[0]) a to b else b to a
                    regions[i] = Region(
                        bbox = listOf(min(a.bbox[0], b.bbox[0]), min(a.bbox[1], b.bbox[1]),
                            max(a.bbox[2], b.bbox[2]), max(a.bbox[3], b.bbox[3])),
                        labels = (a.labels + b.labels).toSortedSet().toList(),
                        f = max(a.f, b.f),
                        text = "${first.text} ${second.text}".trim(),
                    )
                    regions.removeAt(j)
                    merged = true
                    break@loop
                }
            }
        }
        return regions.sortedWith(compareBy({ it.bbox[1] }, { it.bbox[0] }))
    }

    private fun displayLabel(label: String, rxLabel: String, marg: DoubleArray): String = when {
        label != "O" -> label.substring(2)
        rxLabel != "O" -> rxLabel
        else -> labels.indices.filter { it != oIndex }.maxByOrNull { marg[it] }!!.let { labels[it].substring(2) }
    }

    // =================================================================
    // Pipeline cho 1 ảnh
    // =================================================================

    /**
     * words: từ OCR. alreadySorted=false → tự sortReadingOrder (gán lại line).
     * today: mốc "hôm nay" cho luật ngày sinh [P9].
     */
    fun process(words: List<OcrWord>, today: LocalDate = LocalDate.now(), alreadySorted: Boolean = true): EngineResult {
        var ws = words.map { it.copy(text = nfc(it.text)) }   // [P1]
        if (!alreadySorted) ws = sortReadingOrder(ws)
        val ann = annotate(ws, today)
        val feats = ws.indices.map { word2features(ws, ann, it) }
        val attrs = feats.map { toAttributes(it) }
        val S = attrs.map { stateScores(it) }
        val path = viterbi(S)
        val marg = marginals(S)
        val pSens = DoubleArray(ws.size) { 1.0 - marg[it][oIndex] }
        val X = riskInputs(ws, ann, pSens)
        val tokens = ws.indices.map { i ->
            val f = riskScore(X[i])
            TokenResult(ws[i], ann[i], feats[i], attrs[i], labels[path[i]], marg[i], pSens[i], X[i], f, f >= tau)
        }
        val flagged = tokens.filter { it.mask }.map {
            Region(it.word.bbox, listOf(displayLabel(it.label, it.annotation.rx, it.marginals)), it.f, it.word.text)
        }
        return EngineResult(tokens, mergeRegions(flagged))
    }

    companion object {
        /**
         * [P7] Regex lưu theo cú pháp Java. JVM: cần UNICODE_CHARACTER_CLASS để \w \d \s hiểu Unicode như Python.
         * Android (ICU) mặc định đã Unicode; nếu ICU từ chối cờ này thì compile lại không có cờ.
         */
        fun compileRegex(pattern: String, flags: List<String> = emptyList()): Pattern {
            var f = 0
            if ("CASE_INSENSITIVE" in flags) f = f or Pattern.CASE_INSENSITIVE or Pattern.UNICODE_CASE
            return try {
                Pattern.compile(pattern, f or Pattern.UNICODE_CHARACTER_CLASS)
            } catch (e: IllegalArgumentException) {
                Pattern.compile(pattern, f)
            }
        }

        private inline fun Pattern.forEachMatch(s: String, action: (Matcher) -> Unit) {
            val m = matcher(s)
            while (m.find()) action(m)
        }

        private fun countMatches(p: Pattern, s: String): Int {
            val m = p.matcher(s)
            var n = 0
            while (m.find()) n++
            return n
        }

        /** Thay từng đoạn khớp `p` bằng kết quả của `fn` (như re.sub với hàm trong Python). */
        private inline fun replaceEach(p: Pattern, s: String, fn: (String) -> String): String {
            val m = p.matcher(s)
            val sb = StringBuilder(s.length)
            var last = 0
            while (m.find()) {
                sb.append(s, last, m.start()).append(fn(m.group()))
                last = m.end()
            }
            return sb.append(s, last, s.length).toString()
        }

        private fun JsonElement.strList(): List<String> = jsonArray.map { it.jsonPrimitive.content }

        private fun JsonElement.charMap(): Map<Char, Char> =
            jsonObject.entries.associate { (k, v) -> k.single() to v.jsonPrimitive.content.single() }
    }
}
