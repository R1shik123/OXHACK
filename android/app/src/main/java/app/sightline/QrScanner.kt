package app.sightline

import com.google.ar.core.Frame
import com.google.zxing.BinaryBitmap
import com.google.zxing.DecodeHintType
import com.google.zxing.PlanarYUVLuminanceSource
import com.google.zxing.common.HybridBinarizer
import com.google.zxing.qrcode.QRCodeReader
import java.util.concurrent.Executors

/** Decodes the pairing QR code from ARCore's camera image, offline (ZXing), a few times a second. */
class QrScanner(private val onCode: (String) -> Unit) {
    private val worker = Executors.newSingleThreadExecutor()
    @Volatile private var busy = false
    private var last = 0L

    fun offer(frame: Frame) {
        val now = System.currentTimeMillis()
        if (busy || now - last < 300) return
        last = now
        val img = try { frame.acquireCameraImage() } catch (e: Exception) { return }
        val w = img.width
        val h = img.height
        val y = ByteArray(w * h)
        try {
            val p = img.planes[0]
            val buf = p.buffer
            for (row in 0 until h) {
                buf.position(row * p.rowStride)
                buf.get(y, row * w, w)
            }
        } finally {
            img.close()
        }
        busy = true
        worker.execute {
            try {
                val src = PlanarYUVLuminanceSource(y, w, h, 0, 0, w, h, false)
                val hints = mapOf(DecodeHintType.TRY_HARDER to true)
                val text = runCatching { QRCodeReader().decode(BinaryBitmap(HybridBinarizer(src)), hints).text }.getOrNull()
                if (text != null) onCode(text)
            } finally {
                busy = false
            }
        }
    }
}
