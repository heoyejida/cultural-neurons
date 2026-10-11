"""CULNIG cultural-general neuron selection (Yamamoto et al., 2025, Sec. 4.2),
restricted to the ``mlp.gate_proj`` target.

Candidate pool is gate_proj neurons ranked by target-country attribution score;
take the top ``mlp_threshold``, then exclude the top ``countryrc_threshold``
CountryRC neurons over the same pool. No z-score filter: CULNIG's culture-
specific step is degenerate for single-country models (zero per-country
variance). Selection scope stays per ``target_country`` (culture fixed, regime
varied).
"""

from __future__ import annotations

from collections import defaultdict
import logging
import random
from pathlib import Path

from diss.core.adapters import CULNIG_MLP_MODULES, get_adapter
from diss.core.io import (
    dump_json,
    load_json,
    model_slug,
    output_dir,
    score_records_from_json,
    selection_records_to_json,
)
from diss.core.types import NeuronCoord, ScoreRecord, SelectionRecord

LOGGER = logging.getLogger(__name__)

# Top 1% of MLP gate_proj neurons (CULNIG's published value), top 1% CountryRC
# filtered out. mlp_threshold stays tunable for sparsity sweeps.
DEFAULT_MLP_THRESHOLD = 0.01
DEFAULT_COUNTRYRC_THRESHOLD = 0.01


def _score_map(records: list[ScoreRecord]) -> dict[str, ScoreRecord]:
    return {r.coord.key(): r for r in records}


def _allowed_modules(model_name: str, family: str | None = None) -> set[str]:
    adapter = get_adapter(model_name, family=family)
    return set(adapter.default_score_modules())


def _top_k(
    ranked: list[tuple[NeuronCoord, float]], threshold: float
) -> list[tuple[NeuronCoord, float]]:
    """Top fraction of a score-sorted list; positive fractions select at least one."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Selection threshold must be between 0 and 1")
    if not ranked or threshold == 0:
        return []
    return ranked[: max(1, int(len(ranked) * threshold))]


def select_culture_neurons(
    *,
    model_name: str,
    dataset_name: str,
    target_country: str,
    mlp_threshold: float = DEFAULT_MLP_THRESHOLD,
    countryrc_threshold: float = DEFAULT_COUNTRYRC_THRESHOLD,
    revision: str | None = None,
    score_payload: dict | None = None,
    crc_payload: dict | None = None,
    score_records: list[ScoreRecord] | None = None,
    crc_records: list[ScoreRecord] | None = None,
    family: str | None = None,
) -> dict:
    """CULNIG cultural-general MLP-only selection for ``target_country``.

    Pass pre-parsed payloads/records to skip the load_json /
    score_records_from_json calls; used by run-attribution to amortize one
    parse across the select calls that share a score JSON.
    """
    slug = model_slug(model_name)
    base = output_dir(slug, "cultural_neuron", revision=revision)
    if score_payload is None:
        score_payload = load_json(base / f"{dataset_name}_max_scores.json")
    if crc_payload is None:
        from diss.prep.countryrc import COUNTRY_TO_LOCALE

        crc_locale = COUNTRY_TO_LOCALE[target_country]
        crc_path = base / f"countryrc_localized_{crc_locale}_max_scores.json"
        crc_payload = load_json(crc_path)

    if score_records is None:
        score_records = score_records_from_json(score_payload["neuron_scores"])
    if crc_records is None:
        crc_records = score_records_from_json(crc_payload["neuron_scores"])

    score_map = _score_map(score_records)
    crc_map = _score_map(crc_records)

    n_score = max(1, len(score_payload["dataset_ids"].get(dataset_name, [])))
    allowed = _allowed_modules(model_name, family=family)

    neuron_scores: dict[str, dict[str, float]] = defaultdict(dict)
    mlp_candidates: list[tuple[NeuronCoord, float]] = []
    for key, rec in score_map.items():
        mn = rec.coord.logical_module_name
        if mn not in allowed or mn not in CULNIG_MLP_MODULES:
            continue
        for country, v in rec.scores.items():
            neuron_scores[key][country] = v / n_score
        target_score = neuron_scores[key].get(target_country, 0.0)
        mlp_candidates.append((rec.coord, target_score))

    if not mlp_candidates:
        raise ValueError(f"No MLP candidates for {model_name} / {target_country}")

    mlp_ranked = sorted(mlp_candidates, key=lambda x: x[1], reverse=True)
    top = _top_k(mlp_ranked, mlp_threshold)

    crc_ranked = sorted(
        [
            (c, r.scores.get(target_country, 0.0))
            for r in crc_map.values()
            if (c := r.coord).logical_module_name in CULNIG_MLP_MODULES
            and c.logical_module_name in allowed
        ],
        key=lambda x: x[1],
        reverse=True,
    )
    crc_keys = {c.key() for c, _ in _top_k(crc_ranked, countryrc_threshold)}

    selected: list[SelectionRecord] = []
    for coord, score in top:
        if coord.key() in crc_keys:
            continue
        selected.append(SelectionRecord(coord, score, dict(neuron_scores[coord.key()])))

    layer_dist: dict[int, int] = defaultdict(int)
    for sr in selected:
        layer_dist[sr.coord.layer_idx] += 1
    layer_distribution = {str(li): cnt for li, cnt in sorted(layer_dist.items())}

    LOGGER.info(
        "Cultural-general MLP-only selection [%s]: %d neurons selected — "
        "%d/%d MLP candidates, %d removed by CountryRC",
        target_country,
        len(selected),
        len(top),
        len(mlp_candidates),
        len(top) - len(selected),
    )

    ids = {}
    ids.update(score_payload.get("dataset_ids", {}))
    ids.update(crc_payload.get("dataset_ids", {}))
    return {
        "model_name": model_name,
        "target_country": target_country,
        "revision": revision,
        "dataset_ids": ids,
        "thresholds": {
            "mlp_threshold": mlp_threshold,
            "countryrc_threshold": countryrc_threshold,
        },
        "top_neurons": selection_records_to_json(selected),
        "layer_distribution": layer_distribution,
        "total_selected": len(selected),
        "total_candidates": len(mlp_candidates),
    }


def select_random_neurons(
    *,
    model_name: str,
    dataset_name: str,
    seed: int,
    revision: str | None = None,
    match_neuron_file: str | None = None,
    mlp_threshold: float = DEFAULT_MLP_THRESHOLD,
    score_payload: dict | None = None,
    score_records: list[ScoreRecord] | None = None,
    family: str | None = None,
) -> dict:
    """Random MLP-neuron selection. With ``match_neuron_file``, sample exactly
    its MLP-neuron count; otherwise sample top ``mlp_threshold`` of the pool.
    """
    slug = model_slug(model_name)
    base = output_dir(slug, "cultural_neuron", revision=revision)
    if score_payload is None:
        score_payload = load_json(base / f"{dataset_name}_max_scores.json")
    if score_records is None:
        score_records = score_records_from_json(score_payload["neuron_scores"])
    allowed = _allowed_modules(model_name, family=family)
    mlp_pool = [
        r.coord
        for r in score_records
        if r.coord.logical_module_name in CULNIG_MLP_MODULES
        and r.coord.logical_module_name in allowed
    ]

    if match_neuron_file is not None:
        target = load_json(Path(match_neuron_file))
        top = target.get("top_neurons", [])
        mlp_k = sum(1 for n in top if n["logical_module_name"] in CULNIG_MLP_MODULES)
    else:
        mlp_k = len(_top_k(mlp_pool, mlp_threshold))

    rng = random.Random(seed)
    chosen = rng.sample(mlp_pool, min(mlp_k, len(mlp_pool)))

    LOGGER.info(
        "Random MLP selection: %d/%d neurons sampled (seed %d, match=%s)",
        len(chosen),
        len(mlp_pool),
        seed,
        Path(match_neuron_file).name if match_neuron_file else "none",
    )
    return {
        "model_name": model_name,
        "revision": revision,
        "dataset_ids": score_payload.get("dataset_ids", {}),
        "thresholds": {
            "mlp_threshold": mlp_threshold,
            "random_seed": seed,
            "match_neuron_file": match_neuron_file,
        },
        "top_neurons": selection_records_to_json(
            [SelectionRecord(c, 0.0) for c in chosen]
        ),
    }


def select_same_layer_random_neurons(
    *,
    model_name: str,
    dataset_name: str,
    seed: int,
    match_neuron_file: str,
    revision: str | None = None,
    score_payload: dict | None = None,
    score_records: list[ScoreRecord] | None = None,
    family: str | None = None,
) -> dict:
    """Random MLP selection matched to the per-layer histogram of
    ``match_neuron_file``, controlling for the layer-depth confounder: sample k
    neurons at each layer holding k selected in the reference file.
    """
    slug = model_slug(model_name)
    base = output_dir(slug, "cultural_neuron", revision=revision)
    if score_payload is None:
        score_payload = load_json(base / f"{dataset_name}_max_scores.json")
    if score_records is None:
        score_records = score_records_from_json(score_payload["neuron_scores"])
    allowed = _allowed_modules(model_name, family=family)
    candidates = [
        r.coord
        for r in score_records
        if r.coord.logical_module_name in CULNIG_MLP_MODULES
        and r.coord.logical_module_name in allowed
    ]

    target = load_json(Path(match_neuron_file))
    layer_counts: dict[int, int] = defaultdict(int)
    for n in target.get("top_neurons", []):
        if n["logical_module_name"] not in CULNIG_MLP_MODULES:
            continue
        layer_counts[int(n["layer_idx"])] += 1

    by_layer: dict[int, list] = defaultdict(list)
    for c in candidates:
        by_layer[c.layer_idx].append(c)

    rng = random.Random(seed)
    chosen: list = []
    for layer_idx, k in sorted(layer_counts.items()):
        pool = by_layer.get(layer_idx, [])
        if not pool:
            LOGGER.warning(
                "same_layer_random: no MLP candidates at layer %d (need %d) — skipping",
                layer_idx,
                k,
            )
            continue
        chosen.extend(rng.sample(pool, min(k, len(pool))))

    LOGGER.info(
        "Same-layer-random MLP selection: %d / %d candidates sampled across "
        "%d layers (seed %d, match=%s)",
        len(chosen),
        len(candidates),
        len(layer_counts),
        seed,
        Path(match_neuron_file).name,
    )
    return {
        "model_name": model_name,
        "revision": revision,
        "dataset_ids": score_payload.get("dataset_ids", {}),
        "thresholds": {
            "random_seed": seed,
            "match_neuron_file": match_neuron_file,
            "selection_kind": "same_layer_random",
        },
        "top_neurons": selection_records_to_json(
            [SelectionRecord(c, 0.0) for c in chosen]
        ),
    }


def _random_baseline_prefix(match_neuron_file: str | None) -> str:
    """Country tag prefix from the matched culture-file stem.

    Random baselines are sized by the matched culture file, so countries with
    different neuron counts must not share output filenames. Returns "" when no
    match file is given.
    """
    if not match_neuron_file:
        return ""
    stem = Path(match_neuron_file).stem
    head = stem.split("_neurons_", 1)[0]
    return f"{head}_" if head and head != stem else ""


def run_selection_cli(
    args,
    *,
    score_payload: dict | None = None,
    crc_payload: dict | None = None,
    score_records: list[ScoreRecord] | None = None,
    crc_records: list[ScoreRecord] | None = None,
) -> None:
    """CLI entry for `python -m diss.cli select`.

    Pass pre-parsed score data via the kwargs to skip disk reads.
    ``crc_payload`` / ``crc_records`` are only used by the ``culture`` kind.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    revision = getattr(args, "revision", None)
    output_subdir = getattr(args, "output_subdir", None) or "cultural_neuron"
    if args.selection_kind == "culture":
        payload = select_culture_neurons(
            model_name=args.model,
            dataset_name=args.dataset,
            target_country=args.target_country,
            mlp_threshold=args.mlp_threshold,
            countryrc_threshold=args.countryrc_threshold,
            revision=revision,
            score_payload=score_payload,
            crc_payload=crc_payload,
            score_records=score_records,
            crc_records=crc_records,
            family=getattr(args, "family", None),
        )
        fname = f"{args.target_country.replace(' ', '')}_neurons_{args.dataset}.json"
    elif args.selection_kind == "same_layer_random":
        match_file = getattr(args, "match_neuron_file", None)
        if not match_file:
            raise ValueError(
                "--match-neuron-file is required for selection_kind=same_layer_random"
            )
        payload = select_same_layer_random_neurons(
            model_name=args.model,
            dataset_name=args.dataset,
            seed=args.random_seed,
            match_neuron_file=match_file,
            revision=revision,
            score_payload=score_payload,
            score_records=score_records,
            family=getattr(args, "family", None),
        )
        fname = f"{_random_baseline_prefix(match_file)}all_neurons_random_layer{args.random_seed}.json"
    else:
        match_file = getattr(args, "match_neuron_file", None)
        payload = select_random_neurons(
            model_name=args.model,
            dataset_name=args.dataset,
            seed=args.random_seed,
            mlp_threshold=args.mlp_threshold,
            match_neuron_file=match_file,
            revision=revision,
            score_payload=score_payload,
            score_records=score_records,
            family=getattr(args, "family", None),
        )
        fname = f"{_random_baseline_prefix(match_file)}all_neurons_random{args.random_seed}.json"
    out = output_dir(model_slug(args.model), output_subdir, revision=revision) / fname
    dump_json(out, payload)
    LOGGER.info("Saved %s", out)
