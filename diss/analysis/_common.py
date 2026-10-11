"""Shared helpers for diss.analysis scripts.

Deliberately minimal: only helpers whose implementations were byte-identical
across call sites are lifted here, so importing from this module cannot change
any script's output. Grow it only with equally-verified shared code.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

# Per-(model, locale) selected-neuron file: "<Country>_neurons_blend_mcq.json".
NEURON_FILE_RE = re.compile(r"^(?P<country>[A-Za-z]+)_neurons_blend_mcq\.json$")


def load_neuron_selection(path: Path) -> set[tuple]:
    """Set of (family, logical_module_name, layer_idx, neuron_idx) tuples from a
    selected-neuron JSON's ``top_neurons`` list."""
    payload = json.loads(path.read_text())
    return {
        (n["family"], n["logical_module_name"], n["layer_idx"], n["neuron_idx"])
        for n in payload.get("top_neurons", [])
    }
