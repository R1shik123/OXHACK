package app.sightline

import kotlin.math.hypot
import kotlin.math.sqrt

/**
 * Shared coordinate frame anchored to the printed marker (same math as ios/Sightline/SharedFrame.swift
 * and server/sightline/frame.py). Matrices are 4x4 column-major FloatArrays, like ARCore Pose.toMatrix.
 *
 * ARCore's AugmentedImage centre pose has the image in its local x-z plane and +Y as the image normal,
 * exactly like ARKit's image anchor, so the same rule applies:
 *   p = M.col3, n = M.col1 (normal), f = normalize(n.x, 0, n.z) (col2 if the marker lies flat)
 *   S = [X Y Z p] with Y up, Z = f, X = Y x Z;  T = inverse(S)
 */
object SharedFrame {
    fun col(m: FloatArray, c: Int) = floatArrayOf(m[c * 4], m[c * 4 + 1], m[c * 4 + 2])

    private fun horizontal(v: FloatArray): FloatArray {
        val l = hypot(v[0], v[2])
        return if (l > 1e-6f) floatArrayOf(v[0] / l, 0f, v[2] / l) else floatArrayOf(0f, 0f, 1f)
    }

    /** Origin and horizontal forward of one marker observation. */
    fun basis(m: FloatArray): Pair<FloatArray, FloatArray> {
        var n = col(m, 1)
        if (hypot(n[0], n[2]) < 0.3f) n = col(m, 2)
        return col(m, 3) to horizontal(n)
    }

    /** S from averaged marker observations. */
    fun sharedToLocal(samples: List<FloatArray>): FloatArray {
        val p = FloatArray(3)
        val f = FloatArray(3)
        for (m in samples) {
            val (o, fw) = basis(m)
            for (i in 0..2) { p[i] += o[i] / samples.size; f[i] += fw[i] }
        }
        val z = horizontal(f)
        val y = floatArrayOf(0f, 1f, 0f)
        val x = floatArrayOf(y[1] * z[2] - y[2] * z[1], y[2] * z[0] - y[0] * z[2], y[0] * z[1] - y[1] * z[0])
        return floatArrayOf(x[0], x[1], x[2], 0f, y[0], y[1], y[2], 0f, z[0], z[1], z[2], 0f, p[0], p[1], p[2], 1f)
    }

    /** Robust S: median origin + median heading of many sightings (outliers can't drag it). Returns S and spread (m). */
    fun sharedToLocalMedian(samples: List<FloatArray>): Pair<FloatArray, Float> {
        val bases = samples.map { basis(it) }
        fun median(v: List<Float>): Float = v.sorted().let { if (it.size % 2 == 1) it[it.size / 2] else (it[it.size / 2 - 1] + it[it.size / 2]) / 2 }
        val p = floatArrayOf(median(bases.map { it.first[0] }), median(bases.map { it.first[1] }), median(bases.map { it.first[2] }))
        val ref = kotlin.math.atan2(bases[0].second[0], bases[0].second[2])
        val yaw = ref + median(bases.map {
            var d = kotlin.math.atan2(it.second[0], it.second[2]) - ref
            while (d > Math.PI) d -= (2 * Math.PI).toFloat()
            while (d < -Math.PI) d += (2 * Math.PI).toFloat()
            d
        })
        val spread = median(bases.map { val o = it.first; sqrt((o[0] - p[0]) * (o[0] - p[0]) + (o[1] - p[1]) * (o[1] - p[1]) + (o[2] - p[2]) * (o[2] - p[2])) })
        val f = floatArrayOf(kotlin.math.sin(yaw), 0f, kotlin.math.cos(yaw))
        val m = FloatArray(16)
        m[12] = p[0]; m[13] = p[1]; m[14] = p[2]; m[15] = 1f
        val s = sharedToLocal(listOf(m.also { mm -> mm[4] = f[0]; mm[5] = 0f; mm[6] = f[2] }))
        return s to spread
    }

    /** Inverse of a rotation + translation matrix. */
    fun rigidInverse(s: FloatArray): FloatArray {
        val out = FloatArray(16)
        for (c in 0..2) for (r in 0..2) out[c * 4 + r] = s[r * 4 + c]
        for (r in 0..2) out[12 + r] = -(out[r] * s[12] + out[4 + r] * s[13] + out[8 + r] * s[14])
        out[15] = 1f
        return out
    }

    fun point(m: FloatArray, v: FloatArray) = floatArrayOf(
        m[0] * v[0] + m[4] * v[1] + m[8] * v[2] + m[12],
        m[1] * v[0] + m[5] * v[1] + m[9] * v[2] + m[13],
        m[2] * v[0] + m[6] * v[1] + m[10] * v[2] + m[14],
    )

    fun dir(m: FloatArray, v: FloatArray) = floatArrayOf(
        m[0] * v[0] + m[4] * v[1] + m[8] * v[2],
        m[1] * v[0] + m[5] * v[1] + m[9] * v[2],
        m[2] * v[0] + m[6] * v[1] + m[10] * v[2],
    )

    fun mul(a: FloatArray, b: FloatArray): FloatArray {
        val o = FloatArray(16)
        for (c in 0..3) for (r in 0..3) {
            var s = 0f
            for (k in 0..3) s += a[k * 4 + r] * b[c * 4 + k]
            o[c * 4 + r] = s
        }
        return o
    }

    fun normalize(v: FloatArray): FloatArray {
        val l = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
        return if (l > 1e-6f) floatArrayOf(v[0] / l, v[1] / l, v[2] / l) else v
    }
}
