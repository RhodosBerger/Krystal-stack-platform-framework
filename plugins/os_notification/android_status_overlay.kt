package com.krystalstack.kernel.telemetry

import android.app.*
import android.content.Context
import android.content.Intent
import android.graphics.Color
import android.graphics.PixelFormat
import android.os.Build
import android.os.IBinder
import android.view.Gravity
import android.view.LayoutInflater
import android.view.View
import android.view.WindowManager
import android.widget.TextView
import androidx.core.app.NotificationCompat
import kotlinx.coroutines.*
import java.io.File
import java.io.RandomAccessFile

/**
 * ==============================================================================
 * KRYSTAL-STACK: ANDROID KERNEL NOTIFICATION BAR & STATUS OVERLAY
 * Component: plugins/os_notification/android_status_overlay.kt
 * Description: Real-time Android notification drawer widget & floating HUD
 *              monitoring Linux kernel thread context switches and CPU integrity.
 * Target: Android 12+ (API 31 - 35)
 * System Invariant: VITAL_MAX_HP = 6
 * ==============================================================================
 */
class KrystalKernelNotificationService : Service() {

    companion object {
        const val VITAL_MAX_HP = 6
        const val CHANNEL_ID = "krystal_kernel_telemetry_channel"
        const val NOTIFICATION_ID = 6006
    }

    private val serviceScope = CoroutineScope(Dispatchers.Default + Job())
    private var windowManager: WindowManager? = null
    private var overlayView: View? = null

    private var lastContextSwitches: Long = 0
    private var lastTimestampNs: Long = 0

    override fun onCreate() {
        super.onCreate()
        check(VITAL_MAX_HP == 6) { "Invariant VITAL_MAX_HP must remain 6" }
        createNotificationChannel()
        startForeground(NOTIFICATION_ID, buildNotification("Spúšťanie...", "OPTIMAL", 0.0, 1.0))
        initFloatingOverlay()
        startMonitoringLoop()
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                "Krystal Kernel Telemetry Guard",
                NotificationManager.IMPORTANCE_LOW
            ).apply {
                description = "Priebežné monitorovanie integrity kernelu a prepínania vlákien"
                setShowBadge(false)
            }
            val manager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            manager.createNotificationChannel(channel)
        }
    }

    private fun buildNotification(
        diag: String,
        status: String,
        csRate: Double,
        thrashIdx: Double
    ): Notification {
        val color = when (status) {
            "OPTIMAL" -> Color.parseColor("#00FF88")
            "NOMINAL" -> Color.parseColor("#00E5FF")
            "THRASHING_WARNING" -> Color.parseColor("#FFAA00")
            else -> Color.parseColor("#FF2255")
        }

        val contentTitle = "⚡ Kernel Integrity: $status (HP: $VITAL_MAX_HP/6)"
        val contentText = "CS: %,.0f/s | ThrashIdx: %.2f | %s".format(csRate, thrashIdx, diag)

        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle(contentTitle)
            .setContentText(contentText)
            .setSmallIcon(android.R.drawable.ic_dialog_info)
            .setColor(color)
            .setColorized(true)
            .setOngoing(true)
            .setPriority(if (thrashIdx > 2.5) NotificationCompat.PRIORITY_HIGH else NotificationCompat.PRIORITY_LOW)
            .build()
    }

    private fun initFloatingOverlay() {
        try {
            windowManager = getSystemService(Context.WINDOW_SERVICE) as WindowManager
            val params = WindowManager.LayoutParams(
                WindowManager.LayoutParams.WRAP_CONTENT,
                WindowManager.LayoutParams.WRAP_CONTENT,
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O)
                    WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY
                else
                    WindowManager.LayoutParams.TYPE_PHONE,
                WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or WindowManager.LayoutParams.FLAG_LAYOUT_NO_LIMITS,
                PixelFormat.TRANSLUCENT
            ).apply {
                gravity = Gravity.TOP or Gravity.CENTER_HORIZONTAL
                y = 40
            }

            val tv = TextView(this).apply {
                text = "⚡ KRYSTAL: OPTIMAL (CS: 0/s)"
                setTextColor(Color.WHITE)
                setBackgroundColor(Color.argb(220, 16, 22, 34))
                setPadding(24, 12, 24, 12)
                textSize = 11.5f
            }
            overlayView = tv
            windowManager?.addView(overlayView, params)
        } catch (e: Exception) {
            // Overlay permission might be disabled; notification bar remains primary
        }
    }

    private fun readContextSwitchesFromProc(): Long {
        val statFile = File("/proc/stat")
        if (!statFile.exists()) return 0L
        return try {
            statFile.useLines { lines ->
                for (line in lines) {
                    if (line.startsWith("ctxt ")) {
                        return@useLines line.substring(5).trim().toLongOrNull() ?: 0L
                    }
                }
                0L
            }
        } catch (e: Exception) {
            0L
        }
    }

    private fun startMonitoringLoop() {
        lastContextSwitches = readContextSwitchesFromProc()
        lastTimestampNs = System.nanoTime()

        serviceScope.launch {
            while (isActive) {
                delay(1000)
                val nowNs = System.nanoTime()
                val currentCs = readContextSwitchesFromProc()

                val dtSec = maxOf((nowNs - lastTimestampNs) / 1_000_000_000.0, 0.1)
                val deltaCs = maxOf(0L, currentCs - lastContextSwitches)
                val csRate = deltaCs / dtSec

                lastContextSwitches = currentCs
                lastTimestampNs = nowNs

                // Evaluation against baseline (6,500 CS/s)
                val thrashIdx = csRate / 6500.0
                val (status, diag) = when {
                    thrashIdx <= 1.2 -> "OPTIMAL" to "Kernel beží optimálne."
                    thrashIdx <= 2.2 -> "NOMINAL" to "Bežná aktivita plánovača."
                    thrashIdx <= 3.5 -> "THRASHING_WARNING" to "Varovanie: Detekované nadmerné prepínanie vlákien!"
                    else -> "CRITICAL_INTERFERENCE" to "Kritické zlyhanie: Context-switch storm!"
                }

                // Update Notification Drawer Bar
                val manager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
                manager.notify(NOTIFICATION_ID, buildNotification(diag, status, csRate, thrashIdx))

                // Update Floating Overlay HUD on main thread
                withContext(Dispatchers.Main) {
                    (overlayView as? TextView)?.apply {
                        text = "⚡ KRYSTAL: $status | CS: %,.0f/s | Thrash: %.2f".format(csRate, thrashIdx)
                        setTextColor(when (status) {
                            "OPTIMAL" -> Color.parseColor("#00FF88")
                            "NOMINAL" -> Color.parseColor("#00E5FF")
                            "THRASHING_WARNING" -> Color.parseColor("#FFAA00")
                            else -> Color.parseColor("#FF2255")
                        })
                    }
                }
            }
        }
    }

    override fun onDestroy() {
        serviceScope.cancel()
        overlayView?.let { windowManager?.removeView(it) }
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null
}
