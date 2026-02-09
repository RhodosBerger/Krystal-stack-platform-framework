import json
import logging
from pathlib import Path
from typing import Dict, Any


class JsonLogger:
    def __init__(self, log_path: str = "runtime_log.jsonl"):
        self.path = Path(log_path)
        self.base_dir = self.path.parent
        self.logger = logging.getLogger("ovo")
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)
            self.logger.addHandler(logging.StreamHandler())

    def event(self, event_type: str, payload: Dict[str, Any]) -> None:
        row = {"event": event_type, **payload}
        self.base_dir.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
        delegated_path = self.base_dir / f"{event_type}.jsonl"
        with delegated_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
        self.logger.info("[%s] %s", event_type, payload)
