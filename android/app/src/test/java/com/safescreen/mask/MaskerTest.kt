package com.safescreen.mask

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class MaskerTest {

    private val white = 0xFFFFFFFF.toInt()
    private val black = 0xFF000000.toInt()
    private val w = 60
    private val h = 40

    /** Ảnh sọc: mỗi cột một màu xám khác nhau (để blur/pixelate có cái mà làm mờ) */
    private fun stripes() = IntArray(w * h) { i -> val v = (i % w) * 4; (0xFF shl 24) or (v shl 16) or (v shl 8) or v }

    /** Mọi điểm NGOÀI vùng [x0,x1)×[y0,y1) phải giữ nguyên */
    private fun assertOutsideUnchanged(before: IntArray, after: IntArray, x0: Int, y0: Int, x1: Int, y1: Int) {
        for (y in 0 until h) for (x in 0 until w) {
            if (x in x0 until x1 && y in y0 until y1) continue
            assertEquals("điểm ($x,$y) ngoài vùng che bị đổi", before[y * w + x], after[y * w + x])
        }
    }

    @Test
    fun hopDen_cheDungVungCongPadding() {
        val px = IntArray(w * h) { white }
        Masker.maskPixels(px, w, h, listOf(listOf(10, 10, 20, 15)), MaskMode.BLACK_BOX, pad = 4)
        // Như PIL draw.rectangle([6, 6, 24, 19]): tô cả cạnh phải/dưới
        for (y in 0 until h) for (x in 0 until w) {
            val inside = x in 6..24 && y in 6..19
            assertEquals("($x,$y)", if (inside) black else white, px[y * w + x])
        }
    }

    @Test
    fun hopDen_satMepAnhKhongTran() {
        val px = IntArray(w * h) { white }
        Masker.maskPixels(px, w, h, listOf(listOf(0, 0, w, h)), MaskMode.BLACK_BOX, pad = 4)
        assertTrue(px.all { it == black })
    }

    @Test
    fun pixelate_moiOLaMotMauVaNgoaiVungGiuNguyen() {
        val before = stripes()
        val px = before.copyOf()
        Masker.maskPixels(px, w, h, listOf(listOf(12, 12, 44, 28)), MaskMode.PIXELATE, pad = 4)
        // vùng sau padding: x 8..48, y 8..32 (cao 24 → ô 12 px)
        assertOutsideUnchanged(before, px, 8, 8, 48, 32)
        val inside = (8 until 32).flatMap { y -> (8 until 48).map { x -> px[y * w + x] } }.toSet()
        assertTrue("pixelate phải gom thành ít màu (được ${inside.size})", inside.size <= 6)
        assertTrue("pixelate phải làm đổi ảnh", (8 until 48).any { x -> px[8 * w + x] != before[8 * w + x] })
    }

    @Test
    fun blur_lamMoTrongVungVaNgoaiVungGiuNguyen() {
        val before = stripes()
        val px = before.copyOf()
        Masker.maskPixels(px, w, h, listOf(listOf(12, 12, 44, 28)), MaskMode.BLUR, pad = 4)
        assertOutsideUnchanged(before, px, 8, 8, 48, 32)
        // Sọc ngang đổi màu đều → sau làm mờ, chênh lệch giữa 2 cột cạnh nhau phải nhỏ hơn
        val row = 20 * w
        val diffBefore = (9 until 47).sumOf { kotlin.math.abs((before[row + it + 1] and 0xFF) - (before[row + it] and 0xFF)) }
        val diffAfter = (9 until 47).sumOf { kotlin.math.abs((px[row + it + 1] and 0xFF) - (px[row + it] and 0xFF)) }
        assertTrue("blur phải làm mượt ($diffAfter < $diffBefore)", diffAfter < diffBefore)
    }

    @Test
    fun blur_anhMotMauThiGiuNguyen() {
        val px = IntArray(w * h) { 0xFF336699.toInt() }
        Masker.maskPixels(px, w, h, listOf(listOf(5, 5, 50, 30)), MaskMode.BLUR, pad = 4)
        assertTrue(px.all { it == 0xFF336699.toInt() })
    }

    @Test
    fun hopRong_boQua() {
        val px = IntArray(w * h) { white }
        Masker.maskPixels(px, w, h, listOf(listOf(30, 30, 30, 30)), MaskMode.BLACK_BOX, pad = 0)
        assertTrue(px.all { it == white })
    }
}
