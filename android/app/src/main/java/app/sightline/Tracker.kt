package app.sightline

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.YuvImage
import android.media.Image
import com.google.ar.core.AugmentedImage
import com.google.ar.core.Camera
import com.google.ar.core.Frame
import com.google.ar.core.Plane
import com.google.ar.core.Session
import com.google.ar.core.TrackingState
import com.google.ar.core.exceptions.NotYetAvailableException
import org.json.JSONArray
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.Executors
import kotlin.math.atan2

/**
 * Per-frame work on the GL thread: marker lock, pose, planes, ARCore-depth points, camera thumbnails,
 * wall distance and colleague screen positions. Mirrors ios/Sightline/ARManager.swift.
 * ARCore's Depth API gives metric depth from the camera alone on most phones, so no ML model is needed here.
 */
class Tracker(private val net: Net, private val onUi: (UiState) -> Unit) {
    data class Projected(val peer: Peer, val x: Float, val y: Float, val onScreen: Boolean)
    data class UiState(
        val aligned: Boolean, val lockProgress: Float, val tracking: String, val wallDist: Float?,
        val me: FloatArray?, val heading: Float?, val projected: List<Projected>, val xray: Peer?, val scanText: String,
        val calibText: String = "",
    )

    var scanning = true
    var xrayEnabled = true
    @Volatile var peers: List<Peer> = emptyList()

    private var toShared: FloatArray? = null
    private var toLocal: FloatArray? = null
    private val samples = ArrayList<FloatArray>()
    private val reanchor = ArrayList<FloatArray>()
    private var lastReanchor = 0L
    private var calibText = ""
    private var lastPose = 0L
    private var lastDepth = 0L
    private var lastJpeg = 0L
    private var lastStatus = 0L
    private val lastPlaneSend = HashMap<Plane, Long>()
    private val planeIds = HashMap<Plane, String>()
    private var sentXray: String? = null
    private var wallDist: Float? = null
    private var pointsSent = 0
    private val worker = Executors.newSingleThreadExecutor()
    @Volatile private var jpegBusy = false

    fun realign() {
        toShared = null
        toLocal = null
        samples.clear()
        reanchor.clear()
        calibText = ""
        net.send(JSONObject().put("t", "planes_clear"))
        net.send(JSONObject().put("t", "status").put("aligned", false))
    }

    /** Called after (re)connecting: resend everything the server needs. */
    fun resend(session: Session) {
        net.send(JSONObject().put("t", "status").put("aligned", toShared != null))
        lastPlaneSend.clear()
        if (toShared != null) for (p in session.getAllTrackables(Plane::class.java)) sendPlane(p, force = true)
    }

    fun update(session: Session, frame: Frame, viewW: Int, viewH: Int) {
        val now = System.currentTimeMillis()
        val camera = frame.camera
        val tracking = if (camera.trackingState == TrackingState.TRACKING) "normal" else camera.trackingFailureReason.name.lowercase()

        // 1. marker: median lock from 30 close, face-on sightings; later sightings correct drift
        for (img in frame.getUpdatedTrackables(AugmentedImage::class.java)) {
            if (img.trackingState != TrackingState.TRACKING || img.trackingMethod != AugmentedImage.TrackingMethod.FULL_TRACKING) continue
            val m = FloatArray(16)
            img.centerPose.toMatrix(m, 0)
            val cp = camera.pose
            val toCam = floatArrayOf(cp.tx() - m[12], cp.ty() - m[13], cp.tz() - m[14])
            val dist = kotlin.math.sqrt(toCam[0] * toCam[0] + toCam[1] * toCam[1] + toCam[2] * toCam[2])
            val n = SharedFrame.normalize(floatArrayOf(m[4], m[5], m[6]))
            val cos = (n[0] * toCam[0] + n[1] * toCam[1] + n[2] * toCam[2]) / kotlin.math.max(dist, 1e-3f)
            if (dist > 1.5f || cos < 0.64f) continue // too far or too oblique (> ~50 degrees)
            if (toShared == null) {
                samples.add(m)
                if (samples.size >= 30) {
                    val (s, spread) = SharedFrame.sharedToLocalMedian(samples)
                    toLocal = s
                    toShared = SharedFrame.rigidInverse(s)
                    calibText = "Locked ±%.0f mm".format(spread * 1000)
                    resend(session)
                }
            } else if (now - lastReanchor > 5000) {
                reanchor.add(m)
                if (reanchor.size >= 15) {
                    val (s, spread) = SharedFrame.sharedToLocalMedian(reanchor)
                    reanchor.clear()
                    lastReanchor = now
                    val old = toLocal!!
                    val moved = kotlin.math.sqrt((s[12] - old[12]).let { it * it } + (s[13] - old[13]).let { it * it } + (s[14] - old[14]).let { it * it })
                    val dot = (s[8] * old[8] + s[10] * old[10]).coerceIn(-1f, 1f)
                    if (spread < 0.02f && moved < 0.6f && kotlin.math.acos(dot) < 0.26f && moved > 0.01f) {
                        toLocal = s
                        toShared = SharedFrame.rigidInverse(s)
                        calibText = "Drift corrected %.0f cm".format(moved * 100)
                        resend(session)
                    }
                }
            }
        }
        val T = toShared
        val S = toLocal
        var me: FloatArray? = null
        var heading: Float? = null
        val projected = ArrayList<Projected>()
        var xray: Peer? = null

        if (T != null && S != null && camera.trackingState == TrackingState.TRACKING) {
            val cm = FloatArray(16)
            camera.pose.toMatrix(cm, 0)
            val cam = SharedFrame.mul(T, cm)
            me = floatArrayOf(cam[12], cam[13], cam[14])
            val fwd = SharedFrame.normalize(floatArrayOf(-cam[8], -cam[9], -cam[10]))
            heading = atan2(fwd[0], fwd[2])

            if (now - lastPose >= 100) {
                lastPose = now
                net.send(JSONObject().put("t", "pose").put("pos", JSONArray(me.toList())).put("fwd", JSONArray(fwd.toList()))
                    .put("tracking", tracking).put("wall_dist", wallDist ?: JSONObject.NULL))
                wallDist = wallDistance(frame, viewW, viewH, camera)
            }
            if (now - lastStatus >= 5000) {
                lastStatus = now
                net.send(JSONObject().put("t", "status").put("aligned", true).put("tracking", tracking))
            }
            for (p in frame.getUpdatedTrackables(Plane::class.java)) sendPlane(p, force = false)
            if (scanning && now - lastDepth >= 450) {
                lastDepth = now
                depthPoints(frame, camera, T)
            }
            if (now - lastJpeg >= 333 && !jpegBusy) {
                lastJpeg = now
                thumbnail(frame)
            }

            // colleagues on screen (x-ray): shared -> local -> clip -> screen
            val view = FloatArray(16)
            val proj = FloatArray(16)
            camera.getViewMatrix(view, 0)
            camera.getProjectionMatrix(proj, 0, 0.05f, 100f)
            val vp = SharedFrame.mul(proj, view)
            var best = Math.toRadians(30.0).toFloat()
            for (peer in peers) {
                val pos = peer.pos ?: continue
                if (!peer.connected || peer.stale) continue
                val local = SharedFrame.point(S, floatArrayOf(pos[0], pos[1] + 0.15f, pos[2]))
                val clip = floatArrayOf(
                    vp[0] * local[0] + vp[4] * local[1] + vp[8] * local[2] + vp[12],
                    vp[1] * local[0] + vp[5] * local[1] + vp[9] * local[2] + vp[13],
                    vp[3] * local[0] + vp[7] * local[1] + vp[11] * local[2] + vp[15],
                )
                val inFront = clip[2] > 0
                val sx = (clip[0] / clip[2] * 0.5f + 0.5f) * viewW
                val sy = (1f - (clip[1] / clip[2] * 0.5f + 0.5f)) * viewH
                projected.add(Projected(peer, sx, sy, inFront && sx in 0f..viewW.toFloat() && sy in 0f..viewH.toFloat()))
                val d = floatArrayOf(pos[0] - me[0], pos[1] - me[1], pos[2] - me[2])
                val n = SharedFrame.normalize(d)
                val ang = kotlin.math.acos((n[0] * fwd[0] + n[1] * fwd[1] + n[2] * fwd[2]).coerceIn(-1f, 1f))
                if (xrayEnabled && ang < best) { best = ang; xray = peer }
            }
            if (xray?.id != sentXray) {
                sentXray = xray?.id
                net.send(JSONObject().put("t", "xray").put("target", xray?.id ?: JSONObject.NULL))
            }
        }
        onUi(UiState(T != null, samples.size / 30f, tracking, wallDist, me, heading, projected, xray,
            if (scanning) "3D scan · ${pointsSent / 1000}k pts" else "3D scan off", calibText))
    }

    private fun sendPlane(p: Plane, force: Boolean) {
        val T = toShared ?: return
        if (p.trackingState != TrackingState.TRACKING || p.subsumedBy != null) {
            planeIds.remove(p)?.let { net.send(JSONObject().put("t", "plane_remove").put("id", it)) }
            return
        }
        val now = System.currentTimeMillis()
        if (!force && now - (lastPlaneSend[p] ?: 0L) < 700) return
        lastPlaneSend[p] = now
        val id = planeIds.getOrPut(p) { "a" + System.identityHashCode(p).toString(16) }
        val ex = p.extentX / 2
        val ez = p.extentZ / 2
        val corners = JSONArray()
        for ((x, z) in listOf(-ex to -ez, ex to -ez, ex to ez, -ex to ez)) {
            val w = p.centerPose.transformPoint(floatArrayOf(x, 0f, z))
            corners.put(JSONArray(SharedFrame.point(T, w).toList()))
        }
        val kind = when (p.type) {
            Plane.Type.HORIZONTAL_UPWARD_FACING -> "floor"
            Plane.Type.HORIZONTAL_DOWNWARD_FACING -> "ceiling"
            else -> "vertical" // ARCore has no wall/door/window labels; the server treats "vertical" as a wall
        }
        net.send(JSONObject().put("t", "plane").put("id", id).put("kind", kind).put("corners", corners))
    }

    private fun wallDistance(frame: Frame, w: Int, h: Int, camera: Camera): Float? {
        val hit = frame.hitTest(w / 2f, h / 2f).firstOrNull() ?: return null
        val c = camera.pose
        val p = hit.hitPose
        val dx = p.tx() - c.tx()
        val dy = p.ty() - c.ty()
        val dz = p.tz() - c.tz()
        return Math.round(kotlin.math.sqrt(dx * dx + dy * dy + dz * dz) * 10) / 10f
    }

    /** ARCore Depth API -> coloured points in the shared frame (16 B each, same wire format as iOS). */
    private fun depthPoints(frame: Frame, camera: Camera, T: FloatArray) {
        val depth: Image = try { frame.acquireDepthImage16Bits() } catch (e: NotYetAvailableException) { return } catch (e: Exception) { return }
        val color: Image? = try { frame.acquireCameraImage() } catch (e: Exception) { null }
        try {
            val dw = depth.width
            val dh = depth.height
            val plane = depth.planes[0]
            val buf = plane.buffer.order(ByteOrder.nativeOrder())
            val rowStride = plane.rowStride
            val intr = camera.textureIntrinsics
            val dims = intr.imageDimensions
            val sx = dw.toFloat() / dims[0]
            val sy = dh.toFloat() / dims[1]
            val fx = intr.focalLength[0] * sx
            val fy = intr.focalLength[1] * sy
            val cx = intr.principalPoint[0] * sx
            val cy = intr.principalPoint[1] * sy
            val cm = FloatArray(16)
            camera.pose.toMatrix(cm, 0)
            val m = SharedFrame.mul(T, cm)
            val out = ByteBuffer.allocate((dw / 2) * (dh / 2) * 16).order(ByteOrder.LITTLE_ENDIAN)
            var n = 0
            for (y in 0 until dh step 2) for (x in 0 until dw step 2) {
                val mm = buf.getShort(y * rowStride + x * 2).toInt() and 0xffff
                if (mm == 0 || mm > 7000) continue
                val d = mm / 1000f
                val p = SharedFrame.point(m, floatArrayOf((x + 0.5f - cx) / fx * d, -(y + 0.5f - cy) / fy * d, -d))
                val rgb = color?.let { sampleRgb(it, x.toFloat() / dw, y.toFloat() / dh) } ?: intArrayOf(200, 200, 200)
                out.putFloat(p[0]).putFloat(p[1]).putFloat(p[2])
                out.put(rgb[0].toByte()).put(rgb[1].toByte()).put(rgb[2].toByte()).put(if (d < 4) 255.toByte() else 160.toByte())
                n++
            }
            if (n > 0) {
                net.sendBinary(Net.BIN_POINTS, out.array().copyOf(n * 16))
                pointsSent += n
            }
        } finally {
            depth.close()
            color?.close()
        }
    }

    private fun sampleRgb(img: Image, u: Float, v: Float): IntArray {
        val x = (u * img.width).toInt().coerceIn(0, img.width - 1)
        val y = (v * img.height).toInt().coerceIn(0, img.height - 1)
        val yp = img.planes[0]
        val up = img.planes[1]
        val vpl = img.planes[2]
        val Y = (yp.buffer.get(y * yp.rowStride + x * yp.pixelStride).toInt() and 0xff).toFloat()
        val U = (up.buffer.get((y / 2) * up.rowStride + (x / 2) * up.pixelStride).toInt() and 0xff) - 128f
        val V = (vpl.buffer.get((y / 2) * vpl.rowStride + (x / 2) * vpl.pixelStride).toInt() and 0xff) - 128f
        return intArrayOf((Y + 1.402f * V).toInt().coerceIn(0, 255), (Y - 0.344f * U - 0.714f * V).toInt().coerceIn(0, 255), (Y + 1.772f * U).toInt().coerceIn(0, 255))
    }

    /** Camera thumbnail for the commander / x-ray viewers: YUV -> JPEG, rotated to portrait, ~3 Hz. */
    private fun thumbnail(frame: Frame) {
        val img = try { frame.acquireCameraImage() } catch (e: Exception) { return }
        val w = img.width
        val h = img.height
        val nv21 = ByteArray(w * h * 3 / 2)
        try {
            val yp = img.planes[0]
            val up = img.planes[1]
            val vpl = img.planes[2]
            for (row in 0 until h) for (col in 0 until w) nv21[row * w + col] = yp.buffer.get(row * yp.rowStride + col * yp.pixelStride)
            var o = w * h
            for (row in 0 until h / 2) for (col in 0 until w / 2) {
                nv21[o++] = vpl.buffer.get(row * vpl.rowStride + col * vpl.pixelStride)
                nv21[o++] = up.buffer.get(row * up.rowStride + col * up.pixelStride)
            }
        } finally {
            img.close()
        }
        jpegBusy = true
        worker.execute {
            try {
                val raw = ByteArrayOutputStream()
                YuvImage(nv21, ImageFormat.NV21, w, h, null).compressToJpeg(Rect(0, 0, w, h), 70, raw)
                val bmp = BitmapFactory.decodeByteArray(raw.toByteArray(), 0, raw.size())
                val scale = 360f / h
                val rot = Bitmap.createBitmap(bmp, 0, 0, w, h, Matrix().apply { postScale(scale, scale); postRotate(90f) }, true)
                val out = ByteArrayOutputStream()
                rot.compress(Bitmap.CompressFormat.JPEG, 50, out)
                net.sendBinary(Net.BIN_JPEG, out.toByteArray())
            } finally {
                jpegBusy = false
            }
        }
    }
}
