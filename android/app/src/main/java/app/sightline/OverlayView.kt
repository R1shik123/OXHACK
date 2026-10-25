package app.sightline

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.graphics.RectF
import android.view.View
import org.json.JSONArray
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.sin

/** 2D overlay on top of the camera: status, marker prompt, crosshair + wall distance, x-ray markers/card, minimap. */
class OverlayView(ctx: Context) : View(ctx) {
    var ui: Tracker.UiState? = null
    var status = "Connecting…"
    var connected = false
    var myColor = Color.WHITE
    var xrayImage: Bitmap? = null
    var pins: List<MapPin> = emptyList()
    var pairingPrompt: String? = null
    private val walls = HashMap<String, FloatArray>() // id -> x1,z1,x2,z2,kindCode
    private val cells = HashMap<Long, Int>()
    private var floorY = -1.4f
    private val dp = resources.displayMetrics.density
    private val text = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.WHITE; textSize = 13 * dp }
    private val fill = Paint(Paint.ANTI_ALIAS_FLAG)
    private val stroke = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.STROKE; strokeWidth = 3 * dp }

    /** Drop everything learned from the laptop (called when the session ends). */
    fun clearSession() {
        ui = null
        xrayImage = null
        pins = emptyList()
        walls.clear()
        cells.clear()
        connected = false
        status = ""
        invalidate()
    }

    /** Clean walls merged on the server: [x1, z1, x2, z2, y0, y1, kind]. These replace raw plane pieces. */
    fun applyWalls(list: JSONArray) {
        walls.clear()
        for (i in 0 until list.length()) {
            val w = list.getJSONArray(i)
            val code = when (w.optString(6)) { "door" -> 1f; "window" -> 2f; else -> 0f }
            walls["m$i"] = floatArrayOf(w.getDouble(0).toFloat(), w.getDouble(1).toFloat(), w.getDouble(2).toFloat(), w.getDouble(3).toFloat(), code)
        }
        mergedWalls = true
    }

    private var mergedWalls = false

    fun applyPlanes(upsert: JSONArray, remove: JSONArray, reset: Boolean) {
        for (i in 0 until upsert.length()) {
            val p = upsert.getJSONObject(i)
            val c = p.getJSONArray("corners")
            if (p.optString("kind") == "floor" && c.length() > 0) floorY = c.getJSONArray(0).getDouble(1).toFloat()
        }
        if (mergedWalls) return // server sends clean merged walls; raw pieces only until the first merge arrives
        if (reset) walls.clear()
        for (i in 0 until remove.length()) walls.remove(remove.getString(i))
        for (i in 0 until upsert.length()) {
            val p = upsert.getJSONObject(i)
            val kind = p.optString("kind")
            val c = p.getJSONArray("corners")
            val code = when (kind) { "wall", "vertical" -> 0f; "door" -> 1f; "window" -> 2f; else -> continue }
            var best = -1f
            var seg = FloatArray(5)
            for (a in 0 until c.length()) for (b in a + 1 until c.length()) {
                val pa = c.getJSONArray(a)
                val pb = c.getJSONArray(b)
                val d = hypot(pa.getDouble(0) - pb.getDouble(0), pa.getDouble(2) - pb.getDouble(2)).toFloat()
                if (d > best) {
                    best = d
                    seg = floatArrayOf(pa.getDouble(0).toFloat(), pa.getDouble(2).toFloat(), pb.getDouble(0).toFloat(), pb.getDouble(2).toFloat(), code)
                }
            }
            walls[p.getString("id")] = seg
        }
    }

    /** 0x02 | layer | size f32 | N x (int16 x,y,z, uint8 r,g,b): keep obstacle cells for the minimap. */
    fun addVoxels(b: ByteArray) {
        if (b.size < 6 || b[0].toInt() != 2) return
        val bb = java.nio.ByteBuffer.wrap(b).order(java.nio.ByteOrder.LITTLE_ENDIAN)
        val size = bb.getFloat(2)
        var o = 6
        while (o + 9 <= b.size && cells.size < 40000) {
            val y = bb.getShort(o + 2) * size
            if (y > floorY + 0.15f && y < floorY + 1.9f) {
                val cx = Math.round(bb.getShort(o) * size / 0.2f).toLong()
                val cz = Math.round(bb.getShort(o + 4) * size / 0.2f).toLong()
                cells.putIfAbsent((cx shl 32) or (cz and 0xffffffffL), Color.rgb(b[o + 6].toInt() and 0xff, b[o + 7].toInt() and 0xff, b[o + 8].toInt() and 0xff))
            }
            o += 9
        }
    }

    override fun onDraw(c: Canvas) {
        pairingPrompt?.let { prompt ->
            fill.color = Color.argb(170, 0, 0, 0)
            c.drawRoundRect(RectF(24 * dp, height / 2f - 70 * dp, width - 24 * dp, height / 2f + 70 * dp), 16 * dp, 16 * dp, fill)
            text.color = Color.WHITE
            text.textSize = 16 * dp
            prompt.split("\n").forEachIndexed { i, line -> c.drawText(line, 40 * dp, height / 2f - 20 * dp + i * 26 * dp, text) }
            if (status.isNotEmpty() && !connected) { text.color = Color.rgb(255, 160, 0); c.drawText(status, 40 * dp, height / 2f + 50 * dp, text) }
            text.textSize = 13 * dp
            // viewfinder square
            stroke.color = Color.WHITE
            stroke.strokeWidth = 3 * dp
            val s = width * 0.6f
            c.drawRect((width - s) / 2, height / 2f - 90 * dp - s, (width + s) / 2, height / 2f - 90 * dp, stroke)
            return
        }
        val u = ui
        // status bar
        fill.color = Color.argb(150, 0, 0, 0)
        c.drawRoundRect(RectF(12 * dp, 12 * dp, width - 12 * dp, 52 * dp), 12 * dp, 12 * dp, fill)
        text.color = if (connected) Color.GREEN else Color.RED
        c.drawText(if (connected) "● Live" else "● Offline", 24 * dp, 37 * dp, text)
        text.color = Color.LTGRAY
        c.drawText((u?.tracking ?: "") + " · " + (u?.scanText ?: "") + (u?.calibText?.let { if (it.isNotEmpty()) " · $it" else "" } ?: ""), 100 * dp, 37 * dp, text)
        if (!connected) { text.color = Color.rgb(255, 160, 0); c.drawText(status, 24 * dp, 72 * dp, text) }

        if (u == null || !u.aligned) {
            text.color = Color.rgb(255, 160, 0)
            text.textSize = 18 * dp
            c.drawText("Point at the entrance marker", 24 * dp, height / 2f, text)
            if (u != null && u.lockProgress > 0) c.drawText("Locking… ${(u.lockProgress * 100).toInt()}%", 24 * dp, height / 2f + 28 * dp, text)
            text.textSize = 13 * dp
            return
        }

        // colleagues in trouble: red banners under the status bar
        var by = 64 * dp
        for (p in u.projected.map { it.peer }.filter { it.alert != null && it.connected }) {
            val a = p.alert!!
            fill.color = if (a.acked) Color.argb(160, 140, 30, 30) else Color.argb(235, 220, 40, 40)
            c.drawRoundRect(RectF(12 * dp, by, width - 12 * dp, by + 40 * dp), 10 * dp, 10 * dp, fill)
            text.color = Color.WHITE
            val walls = p.wallsBetween?.let { if (it > 0) " · $it wall${if (it > 1) "s" else ""}" else "" } ?: ""
            c.drawText("⚠ ${p.name}: ${a.title} · ${a.seconds.toInt()} s" + (p.distance?.let { " · %.1f m".format(it) } ?: "") + walls, 24 * dp, by + 26 * dp, text)
            by += 46 * dp
        }

        // crosshair + distance to the wall in front
        stroke.color = Color.WHITE
        stroke.strokeWidth = 2 * dp
        val cx = width / 2f
        val cy = height / 2f
        c.drawLine(cx - 14 * dp, cy, cx + 14 * dp, cy, stroke)
        c.drawLine(cx, cy - 14 * dp, cx, cy + 14 * dp, stroke)
        u.wallDist?.let { text.color = Color.WHITE; c.drawText("%.1f m".format(it), cx - 18 * dp, cy + 34 * dp, text) }

        // colleagues: markers through walls
        for (p in u.projected) {
            if (!p.onScreen) continue
            stroke.color = p.peer.color
            stroke.strokeWidth = 3 * dp
            c.drawCircle(p.x, p.y, 22 * dp, stroke)
            text.color = Color.WHITE
            val walls = p.peer.wallsBetween?.let { if (it > 0) " · through $it wall${if (it > 1) "s" else ""}" else "" } ?: ""
            c.drawText(p.peer.name + (p.peer.distance?.let { " %.1f m".format(it) } ?: "") + walls, p.x + 28 * dp, p.y, text)
        }
        // x-ray card: colleague's live camera
        val target = u.xray
        val img = xrayImage
        if (target != null && img != null) {
            val p = u.projected.firstOrNull { it.peer.id == target.id }
            val w = 180 * dp
            val h = w * img.height / img.width
            val left = ((p?.x ?: cx) - w / 2).coerceIn(12 * dp, width - w - 12 * dp)
            val top = ((p?.y ?: cy) - h - 40 * dp).coerceIn(80 * dp, height - h - 220 * dp)
            c.drawBitmap(img, null, RectF(left, top, left + w, top + h), null)
            stroke.color = target.color
            c.drawRect(left, top, left + w, top + h, stroke)
        }
        drawMinimap(c, u)
    }

    private fun drawMinimap(c: Canvas, u: Tracker.UiState) {
        val me = u.me ?: return
        val size = 150 * dp
        val left = width - size - 12 * dp
        val top = height - size - 110 * dp
        val r = 7f // metres from centre to edge
        val s = size / (2 * r)
        fun mx(x: Float) = left + size / 2 + (x - me[0]) * s
        fun mz(z: Float) = top + size / 2 + (z - me[2]) * s
        fill.color = Color.argb(160, 0, 0, 0)
        c.drawRoundRect(RectF(left, top, left + size, top + size), 14 * dp, 14 * dp, fill)
        c.save()
        c.clipRect(left, top, left + size, top + size)
        val cell = 0.2f * s
        for ((k, col) in cells) {
            val x = (k shr 32).toFloat() * 0.2f
            val z = (k.toInt()).toFloat() * 0.2f
            fill.color = col
            c.drawRect(mx(x) - cell / 2, mz(z) - cell / 2, mx(x) + cell / 2, mz(z) + cell / 2, fill)
        }
        stroke.strokeWidth = 2.5f * dp
        for (w in walls.values) {
            stroke.color = when (w[4]) { 1f -> Color.GREEN; 2f -> Color.CYAN; else -> Color.WHITE }
            c.drawLine(mx(w[0]), mz(w[1]), mx(w[2]), mz(w[3]), stroke)
        }
        for (pin in pins) {
            fill.color = when (pin.kind) { "hazard" -> Color.RED; "victim" -> Color.rgb(255, 160, 0); else -> Color.GREEN }
            c.drawCircle(mx(pin.pos[0]), mz(pin.pos[2]), 4 * dp, fill)
        }
        for (p in u.projected) {
            val pos = p.peer.pos ?: continue
            fill.color = p.peer.color
            c.drawCircle(mx(pos[0]), mz(pos[2]), 5 * dp, fill)
        }
        val h = u.heading ?: 0f
        val ox = left + size / 2
        val oy = top + size / 2
        val dx = sin(h)
        val dz = cos(h)
        val arrow = Path().apply {
            moveTo(ox + dx * 10 * dp, oy + dz * 10 * dp)
            lineTo(ox - dx * 6 * dp - dz * 6 * dp, oy - dz * 6 * dp + dx * 6 * dp)
            lineTo(ox - dx * 6 * dp + dz * 6 * dp, oy - dz * 6 * dp - dx * 6 * dp)
            close()
        }
        fill.color = myColor
        c.drawPath(arrow, fill)
        c.restore()
    }
}
