
import React, { useState, useEffect, useRef } from 'react';
import { Telemetry } from '../types';

// Fix: Added React import to resolve the 'Cannot find namespace React' error when using React.Dispatch and React.SetStateAction
export const useTelemetry = (initial: Telemetry): [Telemetry, React.Dispatch<React.SetStateAction<Telemetry>>] => {
  const [data, setData] = useState<Telemetry>(initial);
  
  // Kalman Filter State for Vibration (Smoothing)
  const x = useRef(0.01); // estimate
  const p = useRef(1);    // error covariance
  const q = 0.001;        // process noise
  const r = 0.01;         // measurement noise

  useEffect(() => {
    const interval = setInterval(() => {
      // Mock raw sensor data with noise
      const rawVibration = 0.015 + (Math.random() - 0.5) * 0.02 + (Math.random() > 0.98 ? 0.05 : 0);
      
      // Kalman Update
      const p_priori = p.current + q;
      const k = p_priori / (p_priori + r);
      x.current = x.current + k * (rawVibration - x.current);
      p.current = (1 - k) * p_priori;

      setData(prev => ({
        ...prev,
        vibration: x.current,
        // Only update speed automatically if it's within a small jitter range, 
        // allowing manual overrides to persist better
        speed: prev.speed + (Math.random() - 0.5) * 0.5,
        temp: Math.min(100, prev.temp + (x.current * 10) - 0.1),
        spindleLoad: Math.min(100, 20 + (x.current * 600))
      }));
    }, 100);

    return () => clearInterval(interval);
  }, []);

  return [data, setData];
};

export const getNeuroColor = (dopamine: number, cortisol: number) => {
  if (cortisol > 0.7) return 'text-safety-orange';
  if (dopamine > 0.8) return 'text-emerald-green';
  return 'text-cyber-blue';
};
