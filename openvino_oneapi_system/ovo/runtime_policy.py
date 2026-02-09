import os
from dataclasses import dataclass
from typing import Dict


@dataclass
class RuntimePolicy:
    threads: int
    streams: int
    mode: str

    def as_dict(self) -> Dict[str, int]:
        return {"threads": self.threads, "streams": self.streams, "mode": self.mode}


class RuntimePolicyApplier:
    """
    Applies process-level runtime hints for oneAPI/OpenMP/OpenVINO.
    """

    def apply(self, threads: int, streams: int, mode: str) -> RuntimePolicy:
        os.environ["ONEAPI_NUM_THREADS"] = str(threads)
        os.environ["OMP_NUM_THREADS"] = str(threads)
        os.environ["OPENVINO_NUM_STREAMS"] = str(streams)

        if mode == "aggressive":
            os.environ["KMP_BLOCKTIME"] = "0"
            os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"
        elif mode == "defensive":
            os.environ["KMP_BLOCKTIME"] = "100"
            os.environ["KMP_AFFINITY"] = "granularity=fine,scatter"
        else:
            os.environ["KMP_BLOCKTIME"] = "20"
            os.environ["KMP_AFFINITY"] = "granularity=fine,balanced"

        return RuntimePolicy(threads=threads, streams=streams, mode=mode)

