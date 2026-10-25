package app.sightline

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.BitmapFactory
import android.graphics.Color
import android.media.AudioManager
import android.media.ToneGenerator
import android.opengl.GLES20
import android.opengl.GLSurfaceView
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.Gravity
import android.view.WindowManager
import android.widget.Button
import android.widget.EditText
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import com.google.ar.core.ArCoreApk
import com.google.ar.core.AugmentedImageDatabase
import com.google.ar.core.Config
import com.google.ar.core.Session
import org.json.JSONArray
import org.json.JSONObject
import java.util.concurrent.Executors
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10

/**
 * Sightline for Android (ARCore). Same protocol as the iPhone app, hardened:
 *  - pairing only by scanning the commander's QR inside the app (no deep links, nothing stored)
 *  - the pairing lives in memory; leaving the screen pauses the camera, closes the socket,
 *    forgets the pairing and removes the task, so nothing links the phone to the laptop afterwards
 *  - FLAG_SECURE (no screenshots / recents preview), touches ignored while obscured (tap-jacking)
 *  - refuses to run if re-signed, debuggable or rooted (Integrity.kt)
 */
class MainActivity : AppCompatActivity(), GLSurfaceView.Renderer, Net.Listener {
    private var pairing: Pairing? = null
    private var session: Session? = null
    private var net: Net? = null
    private var tracker: Tracker? = null
    private var scanner: QrScanner? = null
    private lateinit var gl: GLSurfaceView
    private lateinit var overlay: OverlayView
    private lateinit var buttons: LinearLayout
    private val bg = BackgroundRenderer()
    private var installRequested = false
    private var viewW = 0
    private var viewH = 0
    private val io = Executors.newSingleThreadExecutor()
    private var xrayFetching = false
    private var confirming = false
    private var blocked: String? = null
    private var okDialog: AlertDialog? = null
    private var hazardDialog: AlertDialog? = null
    private var answeredHazard: String? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON or WindowManager.LayoutParams.FLAG_SECURE)
        blocked = Integrity.problem(this)
        blocked?.let { showBlocked(it); return }
        buildUi()
    }

    private fun showBlocked(reason: String) {
        setContentView(TextView(this).apply {
            setTextColor(Color.WHITE); textSize = 18f; gravity = Gravity.CENTER; setPadding(48, 48, 48, 48)
            text = "FindMyTeam can't run here.\n\n$reason"
        })
    }

    private fun buildUi() {
        gl = GLSurfaceView(this).apply {
            preserveEGLContextOnPause = false
            setEGLContextClientVersion(2)
            setEGLConfigChooser(8, 8, 8, 8, 16, 0)
            setRenderer(this@MainActivity)
            renderMode = GLSurfaceView.RENDERMODE_CONTINUOUSLY
        }
        overlay = OverlayView(this)
        buttons = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER
            fun b(label: String, action: (Button) -> Unit) = addView(Button(context).apply { text = label; setOnClickListener { action(this) } })
            b("Re-align") { tracker?.realign() }
            b("X-ray ✓") { btn -> tracker?.let { it.xrayEnabled = !it.xrayEnabled; btn.text = if (it.xrayEnabled) "X-ray ✓" else "X-ray" } }
            b("3D ✓") { btn -> tracker?.let { it.scanning = !it.scanning; btn.text = if (it.scanning) "3D ✓" else "3D" } }
            b("Hazard") { dropPin("hazard") }
            b("Disconnect") { endSession() }
        }
        val root = FrameLayout(this).apply {
            filterTouchesWhenObscured = true // ignore taps while another app draws over us
            addView(gl)
            addView(this@MainActivity.overlay)
            addView(buttons, FrameLayout.LayoutParams(FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.WRAP_CONTENT, Gravity.BOTTOM).apply { bottomMargin = 40 })
        }
        setContentView(root)
        showPairingState()
    }

    private fun showPairingState() {
        val paired = pairing != null
        buttons.visibility = if (paired) android.view.View.VISIBLE else android.view.View.GONE
        overlay.pairingPrompt = if (paired) null else "Scan the pairing QR code on the commander's laptop\n(Commander page → + Add phone)"
        overlay.invalidate()
    }

    // ---------- pairing (scanned in-app) ----------
    private fun onScanned(text: String) = runOnUiThread {
        if (pairing != null || confirming) return@runOnUiThread
        val link = PairLink.parse(text) ?: run { overlay.status = "That QR code isn't a FindMyTeam pairing code"; return@runOnUiThread }
        confirming = true
        val name = EditText(this).apply { hint = "Your name"; isSingleLine = true }
        AlertDialog.Builder(this)
            .setTitle("Pair this phone?")
            .setMessage("Server ${link.host}:${link.port}\nCertificate ${link.fingerprint.take(16)}…\n\nThe connection ends when you leave the app.")
            .setView(name)
            .setCancelable(false)
            .setPositiveButton("Pair") { _, _ ->
                val n = name.text.toString().trim().take(32).ifEmpty { "Android" }
                io.execute {
                    val r = Pinning.redeem(link, n)
                    runOnUiThread {
                        confirming = false
                        r.onSuccess { p -> pairing = p; startSession() }
                            .onFailure { e -> Toast.makeText(this, e.message, Toast.LENGTH_LONG).show() }
                    }
                }
            }
            .setNegativeButton("Cancel") { _, _ -> confirming = false }
            .show()
    }

    private fun startSession() {
        val p = pairing ?: return
        val n = Net(p, this)
        net = n
        tracker = Tracker(n) { ui -> runOnUiThread { overlay.ui = ui; overlay.invalidate(); fetchXray(ui.xray) } }
        scanner = null
        n.start()
        showPairingState()
    }

    /** Forget everything about the laptop: socket closed, token dropped, map caches cleared. */
    private fun endSession() {
        net?.stop()
        net = null
        tracker = null
        pairing = null
        if (::overlay.isInitialized) { overlay.clearSession(); showPairingState() }
        scanner = QrScanner(::onScanned)
    }

    private fun dropPin(kind: String) {
        val me = overlay.ui?.me ?: return
        net?.send(JSONObject().put("t", "pin").put("kind", kind).put("label", "").put("pos", JSONArray(me.toList())))
    }

    // ---------- lifecycle: camera and network exist only while we're on screen ----------
    override fun onResume() {
        super.onResume()
        if (blocked != null) return
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(arrayOf(Manifest.permission.CAMERA), 1)
            return
        }
        try {
            if (session == null) {
                if (ArCoreApk.getInstance().requestInstall(this, !installRequested) == ArCoreApk.InstallStatus.INSTALL_REQUESTED) {
                    installRequested = true
                    return
                }
                session = Session(this).also { configure(it) }
            }
            session!!.resume()
        } catch (e: Exception) {
            Toast.makeText(this, "ARCore: ${e.message}", Toast.LENGTH_LONG).show()
            return
        }
        if (pairing == null) scanner = QrScanner(::onScanned)
        gl.onResume()
    }

    override fun onPause() {
        super.onPause()
        if (blocked != null) return
        beacon.stop()
        endSession() // disconnect + forget the pairing immediately
        if (::gl.isInitialized) gl.onPause()
        session?.pause() // releases the camera
    }

    override fun onStop() {
        super.onStop()
        // Leaving the screen ends the app: no background work, no lingering task.
        if (!isChangingConfigurations && !installRequested) finishAndRemoveTask()
    }

    override fun onDestroy() {
        super.onDestroy()
        beacon.release()
        session?.close()
        session = null
        io.shutdownNow()
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (grantResults.firstOrNull() != PackageManager.PERMISSION_GRANTED) Toast.makeText(this, "Camera permission is required", Toast.LENGTH_LONG).show()
    }

    private fun configure(s: Session) {
        val db = AugmentedImageDatabase(s)
        assets.open("marker.png").use { db.addImage("entrance", BitmapFactory.decodeStream(it), markerM) } // metres, as the laptop says
        s.configure(Config(s).apply {
            augmentedImageDatabase = db
            planeFindingMode = Config.PlaneFindingMode.HORIZONTAL_AND_VERTICAL
            updateMode = Config.UpdateMode.LATEST_CAMERA_IMAGE
            focusMode = Config.FocusMode.AUTO
            lightEstimationMode = Config.LightEstimationMode.DISABLED
            cloudAnchorMode = Config.CloudAnchorMode.DISABLED // nothing to Google's cloud
            depthMode = if (s.isDepthModeSupported(Config.DepthMode.AUTOMATIC)) Config.DepthMode.AUTOMATIC else Config.DepthMode.DISABLED
        })
    }

    private fun fetchXray(peer: Peer?) {
        val n = net ?: return
        if (peer == null) { overlay.xrayImage = null; return }
        if (xrayFetching) return
        xrayFetching = true
        io.execute {
            val bytes = n.fetchFrame(peer.id)
            val bmp = bytes?.let { BitmapFactory.decodeByteArray(it, 0, it.size) }
            runOnUiThread { if (bmp != null) overlay.xrayImage = bmp; xrayFetching = false }
        }
    }

    // ---------- GL ----------
    override fun onSurfaceCreated(unused: GL10?, config: EGLConfig?) {
        GLES20.glClearColor(0f, 0f, 0f, 1f)
        bg.create()
    }

    override fun onSurfaceChanged(unused: GL10?, width: Int, height: Int) {
        GLES20.glViewport(0, 0, width, height)
        viewW = width
        viewH = height
        session?.setDisplayGeometry(display?.rotation ?: 0, width, height)
    }

    override fun onDrawFrame(unused: GL10?) {
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT or GLES20.GL_DEPTH_BUFFER_BIT)
        val s = session ?: return
        s.setCameraTextureName(bg.textureId)
        val frame = try { s.update() } catch (e: Exception) { return }
        bg.draw(frame)
        val t = tracker
        if (t != null) t.update(s, frame, viewW, viewH) else scanner?.offer(frame)
    }

    private val beacon by lazy { LocatorBeacon() }

    // ---------- Net.Listener ----------
    private var markerM = 0.18f // printed sheet; the laptop may switch to its on-screen marker

    override fun onMarker(widthM: Float) {
        if (widthM !in 0.05f..0.5f || kotlin.math.abs(widthM - markerM) < 0.001f) return
        markerM = widthM
        session?.let { s -> gl.queueEvent { configure(s); tracker?.realign() } } // new size: lock on again
    }

    override fun onHazard(id: String?, label: String, dist: Float) {
        if (id == null) { hazardDialog?.dismiss(); hazardDialog = null; answeredHazard = null; return }
        if (id == answeredHazard || hazardDialog != null) return
        fun answer(action: String) {
            answeredHazard = id
            hazardDialog = null
            net?.send(JSONObject().put("t", "hazard").put("id", id).put("action", action))
        }
        hazardDialog = AlertDialog.Builder(this)
            .setTitle("Hazard ahead")
            .setMessage("${label.ifEmpty { "Marked hazard" }} · ${"%.1f".format(dist)} m\nYou're heading towards a hazard marked by your team. Keep away if you can.\nProceeding alerts the commander.")
            .setCancelable(false)
            .setPositiveButton("Avoid") { _, _ -> answer("avoid") }
            .setNegativeButton("Proceed anyway") { _, _ -> answer("proceed") }
            .show()
    }

    override fun onConnected() { session?.let { s -> gl.queueEvent { tracker?.resend(s) } } }
    override fun onState(peers: List<Peer>, pins: List<MapPin>, myAlert: Alert?) {
        tracker?.peers = peers
        overlay.pins = pins
        if (myAlert == null) { okDialog?.dismiss(); okDialog = null; beacon.stop() }
        else if (okDialog == null) {
            beacon.start()
            okDialog = AlertDialog.Builder(this)
                .setTitle("Are you OK?")
                .setMessage((if (myAlert.kind == "fall") "It looks like you may have fallen." else "You haven't moved for a while.") + "\nThe commander has been alerted.\nYour phone is sounding a locator alarm so the team can find you.")
                .setCancelable(false)
                .setPositiveButton("I'm OK") { _, _ -> beacon.stop(); net?.send(JSONObject().put("t", "ok")); okDialog = null }
                .show()
        }
    }
    override fun onPlanes(upsert: JSONArray, remove: JSONArray, reset: Boolean) = overlay.applyPlanes(upsert, remove, reset)
    override fun onWalls(walls: JSONArray) = overlay.applyWalls(walls)
    override fun onVoxels(bytes: ByteArray) = overlay.addVoxels(bytes)
    override fun onStatus(connected: Boolean, error: String?, revoked: Boolean) {
        overlay.connected = connected
        overlay.status = error ?: "Connecting…"
        overlay.invalidate()
        if (revoked) {
            endSession()
            Toast.makeText(this, error, Toast.LENGTH_LONG).show()
        }
    }
}

/** Loud repeating tone while the laptop reports this responder as fallen / not moving, so the team can
 *  find them by ear (like a firefighter's PASS alarm). Uses the alarm volume, not the ringer. */
class LocatorBeacon {
    private val tone = ToneGenerator(AudioManager.STREAM_ALARM, ToneGenerator.MAX_VOLUME)
    private val handler = Handler(Looper.getMainLooper())
    private var on = false
    private val tick = object : Runnable {
        override fun run() {
            tone.startTone(ToneGenerator.TONE_CDMA_HIGH_PBX_SS, 700)
            handler.postDelayed(this, 1200)
        }
    }

    fun start() { if (!on) { on = true; handler.post(tick) } }
    fun stop() { on = false; handler.removeCallbacks(tick); tone.stopTone() }
    fun release() { stop(); tone.release() }
}
