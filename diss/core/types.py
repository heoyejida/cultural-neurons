from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class NeuronCoord:
    family: str
    logical_module_name: str
    layer_idx: int
    neuron_idx: int

    def key(self) -> str:
        return f"{self.family}|{self.logical_module_name}|{self.layer_idx}|{self.neuron_idx}"

    @classmethod
    def from_key(cls, key: str) -> NeuronCoord:
        family, logical_module_name, layer_idx, neuron_idx = key.split("|")
        return cls(
            family=family,
            logical_module_name=logical_module_name,
            layer_idx=int(layer_idx),
            neuron_idx=int(neuron_idx),
        )


@dataclass
class PreparedExample:
    example_id: str
    country: str
    locale: str
    labels: list[str]
    gold_label: str
    prompt_prefix: str | None = None
    option_text_by_label: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelSpec:
    model_id: str
    name: str
    architecture_family: str
    role: str
    locales: list[str]
    eval_locales: list[str]
    culture_enabled: bool = True
    reasoning_enabled: bool = True
    neuron_enabled: bool = True
    enabled: bool = True
    revision: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoreRecord:
    coord: NeuronCoord
    scores: dict[str, float]


@dataclass
class SelectionRecord:
    coord: NeuronCoord
    attribute_score: float
    scores: dict[str, float] | None = None
