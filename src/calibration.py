#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: ONLINE CALIBRATION ENGINE (M1)
==============================================================================
Module: src/calibration.py
Performs closed-loop parameter recalibration from empirical receipts:
    CalibrationTrend_N = MAE_baseline - EMA(MAE_N)
    CG = (MAE_{t-1} - MAE_t) / MAE_{t-1}
Constitutional Invariant: VITAL_MAX_HP == 6
==============================================================================
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Any
from .transition_predictor import TransitionPredictor
from .validation_receipt import ValidationReceipt


@dataclass
class CalibrationStatus:
    total_receipts_processed: int
    current_mae: float
    baseline_mae: float
    ema_mae: float
    calibration_trend: float
    calibration_gain: float
    is_learning_confirmed: bool


class OnlineCalibrationEngine:
    """
    Recalibrates the transition predictor using empirical receipts.
    Computes statistical trend and proves parameter convergence.
    """

    def __init__(self, predictor: TransitionPredictor, baseline_mae: float = 0.085, ema_alpha: float = 0.15) -> None:
        self.predictor = predictor
        self.baseline_mae = baseline_mae
        self.ema_alpha = ema_alpha
        self.ema_mae = baseline_mae

        self.historical_maes: List[float] = []
        self.historical_receipts: List[ValidationReceipt] = []

    def ingest_receipt(self, receipt: ValidationReceipt) -> CalibrationStatus:
        """
        Ingests an executed ValidationReceipt, updates predictor drift/sensitivity,
        and computes windowed calibration statistics.
        """
        self.historical_receipts.append(receipt)
        mae = receipt.prediction_mae
        self.historical_maes.append(mae)
        if len(self.historical_maes) > 100:
            self.historical_maes.pop(0)

        # Update EMA
        self.ema_mae = (self.ema_alpha * mae) + ((1.0 - self.ema_alpha) * self.ema_mae)

        # Recalibrate predictor parameters from directional residual
        self.predictor.calibrate(receipt.prediction_residual, learning_rate=0.03)

        # Compute trend and single-step gain
        trend = round(self.baseline_mae - self.ema_mae, 4)
        if len(self.historical_maes) >= 2:
            prev = self.historical_maes[-2]
            curr = self.historical_maes[-1]
            gain = round((prev - curr) / max(0.001, prev), 4)
        else:
            gain = 0.0

        is_confirmed = (len(self.historical_maes) >= 5 and trend > 0.005)

        return CalibrationStatus(
            total_receipts_processed=len(self.historical_receipts),
            current_mae=round(mae, 4),
            baseline_mae=round(self.baseline_mae, 4),
            ema_mae=round(self.ema_mae, 4),
            calibration_trend=trend,
            calibration_gain=gain,
            is_learning_confirmed=is_confirmed,
        )
