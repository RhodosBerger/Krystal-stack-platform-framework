#!/usr/bin/env python3
"""
THE SENSORY CORTEX
Unified Parser System for Fanuc Rise.

Purpose: To collect data from various endpoints (Fanuc, SW, Sensors)
and normalize it into a unified 'SenseDatum' for the Brain.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import datetime
import json
import logging
from backend.cms.hal_core import GenericController
from backend.cms.hal_fanuc import FanucAdapter
from backend.cms.hal_siemens import SiemensAdapter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [CORTEX] - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class SenseDatum:
    """
    The Atomic Unit of Experience.
    normalized data structure from ANY source.
    """
    source_id: str          # e.g., "FANUC_CNTRL_01", "SW_SIM_02"
    timestamp: float        # UTC Context
    data_type: str          # "TELEMETRY", "METADATA", "EVENT"
    
    # Unified Metrics (Normalized 0.0 - 1.0 where possible)
    load_factor: float      # 0.0 (Idle) - 1.0 (Stall)
    vibration_level: float  # 0.0 (Smooth) - 1.0 (Chatter)
    speed_metric: float     # 0.0 (Slow) - 1.0 (Max RPM)
    
    # Raw Data Dump
    raw_payload: Dict[str, Any]


class BaseParser:
    """
    Abstract Base Class for all Parsers.
    """
    def __init__(self, endpoint_config: Dict):
        self.config = endpoint_config

    def fetch(self) -> SenseDatum:
        raise NotImplementedError

class HALParser(BaseParser):
    """
    Parser for any HAL-compliant Controller (Fanuc, Siemens, etc.).
    """
    def __init__(self, controller: GenericController):
        self.controller = controller

    def fetch(self, mock_data: Dict = None) -> SenseDatum:
        # 1. Connect if needed
        if self.controller.get_status() == "DISCONNECTED":
            self.controller.connect()
            
        # 2. Read Metrics from Hardware
        metrics = self.controller.read_metrics()
        
        # 3. Normalize
        return SenseDatum(
            source_id=f"HAL_{self.controller.get_protocol_name()}",
            timestamp=datetime.datetime.utcnow().timestamp(),
            data_type="TELEMETRY",
            load_factor=metrics.get("load", 0.0),
            vibration_level=metrics.get("vibration", 0.0),
            speed_metric=metrics.get("rpm", 0.0) / 20000.0, # Norm against Max
            raw_payload=metrics
        )

class SolidworksParser(BaseParser):
    """
    Parser for Solidworks Feature Data.
    """
    def fetch(self, mock_features: Dict = None) -> SenseDatum:
        feat = mock_features or {"curvature": 0}
        
        # Curvature maps to 'Vibration Potential' (High curvature = jerky moves)
        vib_potential = min(1.0, feat.get("curvature", 0) * 10) 
        
        return SenseDatum(
            source_id="SOLIDWORKS_CAM",
            timestamp=datetime.datetime.utcnow().timestamp(),
            data_type="METADATA",
            load_factor=0.5, # Static prediction
            vibration_level=vib_potential,
            speed_metric=0.0,
            raw_payload=feat
        )

class SensoryCortex:
    """
    The Unified Collector.
    """
    def __init__(self):
        self.parsers = []
        
    def register_parser(self, parser: BaseParser):
        self.parsers.append(parser)
        
    def collect_all(self, inputs: Dict[str, Any]) -> List[SenseDatum]:
        """
        Aggregates from all registered sources.
        """
        results = []
        for p in self.parsers:
            # In a real system, inputs might be handled differently per parser
            # Here we just pass the matching key
            if isinstance(p, HALParser):
                results.append(p.fetch(inputs.get("hal_override")))
            elif isinstance(p, SolidworksParser):
                results.append(p.fetch(inputs.get("solidworks")))
        return results
