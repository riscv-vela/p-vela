"""Read the shared literal KEY=value file without executing shell code."""

import math
from pathlib import Path
import re
import shlex


CONFIG_PATH = Path(__file__).resolve().with_name("power_model_defaults.env")


def load_defaults(path=CONFIG_PATH):
    values = {}
    with open(path, encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            tokens = shlex.split(line, comments=True)
            if not tokens:
                continue
            if len(tokens) != 1 or "=" not in tokens[0]:
                raise ValueError(f"{path}:{lineno}: expected literal KEY=value")
            key, value = tokens[0].split("=", 1)
            if not re.fullmatch(r"[A-Z][A-Z0-9_]*", key) or not value:
                raise ValueError(f"{path}:{lineno}: invalid setting")
            if key in values:
                raise ValueError(f"{path}:{lineno}: duplicate setting {key}")
            values[key] = value
    return values


DEFAULTS = load_defaults()


def number(key):
    """Return a finite numeric setting; missing/invalid values fail explicitly."""
    value = float(DEFAULTS[key])
    if not math.isfinite(value):
        raise ValueError(f"{CONFIG_PATH}: {key} must be finite")
    return value
