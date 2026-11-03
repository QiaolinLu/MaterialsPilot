"""Scalar outcome normalization and strict JSON evidence serialization.

NumPy booleans must be normalized before identity-based outcome decisions,
not just when writing a journal. Nonfinite evidence is represented as null
with an explicit path/type/value warning rather than nonstandard JSON NaN.
"""
from __future__ import annotations

import math
import numpy as np


def optional_bool(value):
    """Return the three-state outcome as a Python bool or None.

    Reject malformed outcome types instead of accepting truthy strings or
    NaNs as success. Geometric predicates may legitimately yield np.bool_.
    """
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise TypeError(f"Constraint outcome must be bool or None, got {type(value).__name__}")


def strict_json_value(value, warnings, path="$"):
    """Recursively convert NumPy scalars/arrays; annotate nonfinite values."""
    original_type = type(value).__name__
    if isinstance(value, np.ndarray):
        return strict_json_value(value.tolist(), warnings, path)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    elif isinstance(value, np.generic):
        return strict_json_value(value.item(), warnings, path)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        warnings.append({"path": path, "kind": "nonfinite_replaced_with_null",
                         "original_type": original_type, "value": repr(value)})
        return None
    if isinstance(value, dict):
        output = {}
        for key, item in value.items():
            if isinstance(key, np.generic):
                key = key.item()
            if not isinstance(key, (str, int, float, bool)) and key is not None:
                raise TypeError(f"Unsupported JSON key at {path}: {type(key).__name__}")
            if isinstance(key, float) and not math.isfinite(key):
                raise ValueError(f"Nonfinite JSON key at {path}")
            output[key] = strict_json_value(item, warnings, f"{path}.{key}")
        return output
    if isinstance(value, (list, tuple)):
        return [strict_json_value(item, warnings, f"{path}[{index}]") for index,item in enumerate(value)]
    raise TypeError(f"Unsupported JSON evidence at {path}: {type(value).__name__}")


def json_mapping(value):
    """Return a strict JSON object with any replacement warnings attached."""
    warnings = []
    result = strict_json_value(value, warnings)
    if warnings:
        result.setdefault("serialization_warnings", []).extend(warnings)
    return result
