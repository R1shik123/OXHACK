package app.sightline

import android.os.Handler
import android.os.Looper
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import okio.ByteString
import okio.ByteString.Companion.toByteString
import org.json.JSONArray
import org.json.JSONObject

data class Peer(
    val id: String, val name: String, val color: Int, val connected: Boolean, val stale: Boolean,
    val pos: FloatArray?, val wallsBetween: Int?, val distance: Float?, val alert: Alert? = null,
)

/** Fall / no-movement alert raised by the server. */
data class Alert(val kind: String, val seconds: Float, val acked: Boolean) {
    val title get() = if (kind == "fall") "possible fall" else "not moving"
}

data class MapPin(val id: String, val kind: String, val pos: FloatArray)

/** Pinned WSS connection; reconnects forever. Callbacks are delivered on the main thread. */
class Net(private val pairing: Pairing, private val listener: Listener) {
    interface Listener {
        fun onConnected()
        /** Heading towards a hazard pin: (pin id, label, metres), or null when clear. */
        fun onHazard(id: String?, label: String, dist: Float)
        /** Physical width of the entrance marker in use (printed sheet or the commander's screen), metres. */
        fun onMarker(widthM: Float)
        fun onState(peers: List<Peer>, pins: List<MapPin>, myAlert: Alert?)
        fun onPlanes(upsert: JSONArray, remove: JSONArray, reset: Boolean)
        fun onWalls(walls: JSONArray)
        fun onVoxels(bytes: ByteArray)
        fun onStatus(connected: Boolean, error: String?, revoked: Boolean)
    }

    companion object {
        const val BIN_JPEG: Byte = 0x01
        const val BIN_POINTS: Byte = 0x02
    }

    val http: OkHttpClient = Pinning.client(pairing.fingerprint)
    private val main = Handler(Looper.getMainLooper())
    private var ws: WebSocket? = null
    @Volatile var connected = false
        private set
    private var closed = false
    private var revoked = false

    fun start() {
        closed = false
        open()
    }

    fun stop() {
        closed = true
        main.removeCallbacksAndMessages(null) // no pending reconnects after we leave
        ws?.close(1000, null)
        ws = null
        http.dispatcher.executorService.shutdown()
        http.connectionPool.evictAll()
    }

    private fun open() {
        if (closed || revoked) return
        if (ws != null) ws?.cancel()
        val req = Request.Builder().url(pairing.wsUrl).header("Authorization", "Bearer ${pairing.token}").build()
        ws = http.newWebSocket(req, object : WebSocketListener() {
            override fun onMessage(webSocket: WebSocket, text: String) = handle(text)
            override fun onMessage(webSocket: WebSocket, bytes: ByteString) {
                val b = bytes.toByteArray()
                main.post { listener.onVoxels(b) }
            }
            override fun onClosing(webSocket: WebSocket, code: Int, reason: String) {
                if (code == 4401 || code == 4403) revoke()
                webSocket.close(code, null)
            }
            override fun onClosed(webSocket: WebSocket, code: Int, reason: String) = lost(null)
            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                if (response?.code == 403 || response?.code == 401) revoke() else lost(t.message)
            }
        })
    }

    private fun revoke() {
        revoked = true
        connected = false
        main.post { listener.onStatus(false, "This phone was removed by the commander. Pair again.", true) }
    }

    private fun lost(err: String?) {
        connected = false
        if (revoked) return
        main.post { listener.onStatus(false, err, false) }
        if (!closed) main.postDelayed({ open() }, 1000)
    }

    private fun handle(text: String) {
        val m = runCatching { JSONObject(text) }.getOrNull() ?: return
        main.post {
            when (m.optString("t")) {
                "welcome" -> {
                    connected = true
                    listener.onStatus(true, null, false)
                    listener.onMarker(m.optDouble("marker_m", 0.18).toFloat())
                    listener.onConnected()
                }
                "marker" -> listener.onMarker(m.optDouble("marker_m", 0.18).toFloat())
                "state" -> {
                    listener.onState(parsePeers(m.optJSONArray("peers")), parsePins(m.optJSONArray("pins")),
                        parseAlert(m.optJSONObject("me")?.optJSONObject("alert")))
                    val h = m.optJSONObject("me")?.optJSONObject("hazard")
                    listener.onHazard(h?.optString("id"), h?.optString("label") ?: "", h?.optDouble("dist", 0.0)?.toFloat() ?: 0f)
                }
                "planes" -> listener.onPlanes(m.optJSONArray("upsert") ?: JSONArray(), m.optJSONArray("remove") ?: JSONArray(), m.optBoolean("reset"))
                "walls" -> listener.onWalls(m.optJSONArray("walls") ?: JSONArray())
            }
        }
    }

    fun send(json: JSONObject) {
        if (connected) ws?.send(json.toString())
    }

    fun sendBinary(kind: Byte, payload: ByteArray) {
        if (!connected) return
        val out = ByteArray(payload.size + 1)
        out[0] = kind
        System.arraycopy(payload, 0, out, 1, payload.size)
        ws?.send(out.toByteString())
    }

    /** Authenticated fetch of a colleague's latest camera frame (blocking). */
    fun fetchFrame(peerId: String): ByteArray? = runCatching {
        val req = Request.Builder().url("${pairing.httpBase}/api/frame/$peerId.jpg").header("Authorization", "Bearer ${pairing.token}").build()
        http.newCall(req).execute().use { if (it.code == 200) it.body?.bytes() else null }
    }.getOrNull()

    private fun vec(a: JSONArray?): FloatArray? =
        if (a == null || a.length() < 3) null else floatArrayOf(a.getDouble(0).toFloat(), a.getDouble(1).toFloat(), a.getDouble(2).toFloat())

    private fun parsePeers(a: JSONArray?): List<Peer> = (0 until (a?.length() ?: 0)).map { i ->
        val p = a!!.getJSONObject(i)
        Peer(
            p.getString("id"), p.optString("name"), android.graphics.Color.parseColor(p.optString("color", "#ffffff")),
            p.optBoolean("connected"), p.optBoolean("stale", true), vec(p.optJSONArray("pos")),
            if (p.has("walls_between")) p.optInt("walls_between") else null,
            if (p.has("distance")) p.optDouble("distance").toFloat() else null,
            parseAlert(p.optJSONObject("alert")),
        )
    }

    private fun parseAlert(a: JSONObject?): Alert? =
        a?.let { Alert(it.optString("kind"), it.optDouble("for").toFloat(), it.optBoolean("acked")) }

    private fun parsePins(a: JSONArray?): List<MapPin> = (0 until (a?.length() ?: 0)).mapNotNull { i ->
        val p = a!!.getJSONObject(i)
        vec(p.optJSONArray("pos"))?.let { MapPin(p.getString("id"), p.optString("kind"), it) }
    }
}
