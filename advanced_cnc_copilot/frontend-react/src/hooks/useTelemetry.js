import { useState, useEffect, useCallback, useRef } from 'react';

/**
 * Hook for managing real-time CNC telemetry via FastAPI WebSockets.
 * Handles automatic reconnection and fallback to simulation mode.
 */
export const useTelemetry = () => {
    const [telemetry, setTelemetry] = useState({
        machine_id: "CNC-001",
        rpm: 0,
        feed: 0,
        load: 0,
        vibration: 0,
        temperature_c: 25.0,
        tool_health: 1.0,
        neuro_state: { dopamine: 50, cortisol: 50, serotonin: 50 },
        action: "INITIALIZING",
        reasoning_trace: ["[System] Connecting to Nervous System..."],
        scalpel: { active: false, fro: 1.0, reason: "NOMINAL" },
        is_simulated: true // Default to true until connection is established
    });

    const ws = useRef(null);
    const reconnectTimeout = useRef(null);

    const connect = useCallback(() => {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        // In local dev, standard proxy might not handle WS well, so we target 8000 directly if needed
        const host = window.location.hostname === 'localhost' ? 'localhost:8000' : window.location.host;
        const wsUrl = `${protocol}//${host}/ws/telemetry`;

        console.log(`[Telemetry] Connecting to ${wsUrl}...`);
        ws.current = new WebSocket(wsUrl);

        ws.current.onopen = () => {
            console.log("[Telemetry] Uplink Established.");
            setTelemetry(prev => ({ ...prev, is_simulated: false }));
        };

        ws.current.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                setTelemetry(prev => ({
                    ...prev,
                    ...data,
                    is_simulated: false
                }));
            } catch (err) {
                console.error("[Telemetry] Data Parse Error:", err);
            }
        };

        ws.current.onerror = (err) => {
            console.warn("[Telemetry] Connection Error. Falling back to Simulation.");
            setTelemetry(prev => ({ ...prev, is_simulated: true }));
        };

        ws.current.onclose = () => {
            console.log("[Telemetry] Connection Closed. Retrying in 5s...");
            setTelemetry(prev => ({ ...prev, is_simulated: true }));
            reconnectTimeout.current = setTimeout(connect, 5000);
        };
    }, []);

    useEffect(() => {
        connect();
        return () => {
            if (ws.current) ws.current.close();
            if (reconnectTimeout.current) clearTimeout(reconnectTimeout.current);
        };
    }, [connect]);

    // Mock/Simulated Loop if offline
    useEffect(() => {
        let simInterval;
        if (telemetry.is_simulated) {
            simInterval = setInterval(() => {
                setTelemetry(prev => {
                    if (!prev.is_simulated) return prev; // Avoid racing if connected
                    const nextRpm = 5000 + Math.random() * 2000;
                    const nextFeed = 500 + Math.random() * 50;
                    const nextTemp = prev.temperature_c + (nextRpm ** 1.5 * nextFeed ** 0.5) * 0.0000001 - 0.05;
                    const nextToolHealth = Math.max(0, prev.tool_health - 0.0001);

                    return {
                        ...prev,
                        rpm: nextRpm,
                        feed: nextFeed,
                        load: 30 + (nextRpm / 200) + Math.random() * 5,
                        vibration: 0.1 + Math.random() * 0.05,
                        temperature_c: Math.max(25, nextTemp),
                        tool_health: nextToolHealth,
                        neuro_state: {
                            dopamine: 40 + Math.random() * 20,
                            cortisol: 30 + Math.random() * 10,
                            serotonin: 50 + Math.random() * 10
                        },
                        action: "SIMULATED_RUN",
                        reasoning_trace: [
                            "[System] Simulating physics-based telemetry...",
                            `[Thermal] Spindle Temp: ${nextTemp.toFixed(1)}C`,
                            `[Tool] Health: ${(nextToolHealth * 100).toFixed(1)}%`
                        ],
                        scalpel: {
                            active: nextTemp > 80,
                            fro: nextTemp > 80 ? 0.8 : 1.0,
                            reason: nextTemp > 80 ? "THERMAL_COMPENSATION" : "NOMINAL"
                        }
                    };
                });
            }, 1000);
        }
        return () => clearInterval(simInterval);
    }, [telemetry.is_simulated]);

    return telemetry;
};
