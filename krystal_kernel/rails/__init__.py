"""Configuration rails — declarative JSON graphs that connect node functions.

Stdlib-only. Typical use::

    from krystal_kernel.rails import load_preset, run_rail
    result = run_rail(load_preset("hr1_threshold_switching"))
    print(result["values"]["verdict.verdict"]["status"])

CLI::

    python -m krystal_kernel.rails list
    python -m krystal_kernel.rails run hr1_threshold_switching
    python -m krystal_kernel.rails run path/to/my.rail.json
"""

from .nodes import DEFAULT_REGISTRY, NodeRegistry, NodeSpec, Param, Port, node
from .rail import (
    MAX_LINKS,
    MAX_NODES,
    MAX_RAIL_SOURCE_SECONDS,
    PRESETS_DIR,
    RAIL_SCHEMA,
    RailValidationError,
    check_rail,
    jsonable,
    list_presets,
    load_preset,
    run_rail,
    summarize_result,
)
from .schema_validate import SchemaError, unsupported_keywords, validate

__all__ = [
    "DEFAULT_REGISTRY", "NodeRegistry", "NodeSpec", "Param", "Port", "node",
    "MAX_LINKS", "MAX_NODES", "MAX_RAIL_SOURCE_SECONDS", "PRESETS_DIR", "RAIL_SCHEMA",
    "RailValidationError", "check_rail", "jsonable", "list_presets", "load_preset",
    "run_rail", "summarize_result", "SchemaError", "unsupported_keywords", "validate",
]
