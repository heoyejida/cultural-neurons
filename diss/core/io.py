from __future__ import annotations

import csv
import json
from pathlib import Path

from diss.core.types import NeuronCoord, ScoreRecord, SelectionRecord

OUTPUT_ROOT = Path("diss/outputs")


def output_dir(
    slug: str,
    subdir: str,
    revision: str | None = None,
    *,
    root: Path | None = None,
) -> Path:
    """revision falsy → <root>/slug/subdir; truthy → <root>/slug/cp_{revision}/subdir."""
    base = (root if root is not None else OUTPUT_ROOT) / slug
    if revision:
        base = base / f"cp_{revision}"
    return base / subdir


def masked_output_dir(
    slug: str,
    neuron_stem: str,
    subdir: str,
    revision: str | None = None,
    *,
    root: Path | None = None,
) -> Path:
    """Masked evals nest under the base model: <root>/<slug>/[cp_<rev>/]masked/<neuron_stem>/<subdir>."""
    return (
        output_dir(slug, "masked", revision=revision, root=root) / neuron_stem / subdir
    )


def model_slug(model_name: str) -> str:
    return model_name.split("/")[-1]


def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def score_records_to_json(records: list[ScoreRecord]) -> list[dict]:
    return [
        {
            "family": r.coord.family,
            "logical_module_name": r.coord.logical_module_name,
            "layer_idx": r.coord.layer_idx,
            "neuron_idx": r.coord.neuron_idx,
            "scores": r.scores,
        }
        for r in records
    ]


def selection_records_to_json(records: list[SelectionRecord]) -> list[dict]:
    return [
        {
            "family": r.coord.family,
            "logical_module_name": r.coord.logical_module_name,
            "layer_idx": r.coord.layer_idx,
            "neuron_idx": r.coord.neuron_idx,
            "attribute_score": r.attribute_score,
            "scores": r.scores,
        }
        for r in records
    ]


def score_records_from_json(payload: list[dict]) -> list[ScoreRecord]:
    return [
        ScoreRecord(
            coord=NeuronCoord(
                family=item["family"],
                logical_module_name=item["logical_module_name"],
                layer_idx=int(item["layer_idx"]),
                neuron_idx=int(item["neuron_idx"]),
            ),
            scores={str(k): float(v) for k, v in item.get("scores", {}).items()},
        )
        for item in payload
    ]


def selection_records_from_json(payload: list[dict]) -> list[SelectionRecord]:
    return [
        SelectionRecord(
            coord=NeuronCoord(
                family=item["family"],
                logical_module_name=item["logical_module_name"],
                layer_idx=int(item["layer_idx"]),
                neuron_idx=int(item["neuron_idx"]),
            ),
            attribute_score=float(item["attribute_score"]),
            scores=item.get("scores"),
        )
        for item in payload
    ]


def read_results_csv(path: Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def results_accuracy(
    rows: list[dict],
    *,
    topics: set[str] | None = None,
    correct_field: str = "is_correct",
) -> tuple[float, int, int] | tuple[None, None, None]:
    """Row-instance accuracy over results.csv. topics=None → whole file; else filter by topic.

    Returns (accuracy, correct, total), or (None, None, None) when empty.
    """
    if topics is not None:
        if rows and "topic" not in rows[0]:
            raise RuntimeError(
                "results.csv has no 'topic' column; re-run the eval "
                "(python -m diss.cli eval) to regenerate it with the topic column"
            )
        rows = [r for r in rows if r["topic"] in topics]
    total = len(rows)
    if total == 0:
        return None, None, None
    correct = sum(1 for r in rows if str(r.get(correct_field, "")).lower() == "true")
    return correct / total, correct, total
