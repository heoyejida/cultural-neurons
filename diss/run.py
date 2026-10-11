"""Manifest-driven orchestrator: run-behavior + summarize."""

from __future__ import annotations

import csv
import logging
from pathlib import Path

from diss.core.io import (
    OUTPUT_ROOT,
    masked_output_dir,
    model_slug,
    output_dir,
    read_results_csv,
    results_accuracy,
)
from diss.core.registry import CULTURE_COUNTRY_BY_LOCALE, load_manifest
from diss.prep.blend import NON_NEURON_TOPICS

LOGGER = logging.getLogger(__name__)


def _locale_enabled(spec, locale: str) -> bool:
    return locale in spec.eval_locales


def _expected_eval_outputs(
    spec,
    *,
    culture_locales: list[str],
    reasoning_locales: list[str],
    output_root: Path | None = None,
    revision: str | None = None,
) -> list[Path]:
    """results.csv paths this spec would produce (mirrors run_behavior filtering)."""
    rev = revision if revision is not None else getattr(spec, "revision", None)
    root_kwarg = {"revision": rev} if rev else {}
    root = Path(output_root) if output_root is not None else OUTPUT_ROOT
    slug = model_slug(spec.name)

    def _path(subdir: str) -> Path:
        if rev:
            return root / slug / f"cp_{rev}" / subdir / "results.csv"
        return root / slug / subdir / "results.csv"

    paths: list[Path] = []
    if spec.culture_enabled:
        for locale in culture_locales:
            if locale not in spec.eval_locales:
                continue
            if CULTURE_COUNTRY_BY_LOCALE.get(locale) is None:
                continue
            paths.append(_path(f"blend_mcq_{locale}"))
    if spec.reasoning_enabled:
        for locale in reasoning_locales:
            if locale not in spec.eval_locales:
                continue
            paths.append(_path(f"belebele_{locale}"))
    return paths


def run_behavior(args) -> None:
    import torch
    from diss.core.gpu import load_model_and_tokenizer, setup_torch_optimizations
    from diss.pipeline.evaluate import evaluation_cache_complete, run_evaluation

    setup_torch_optimizations()
    manifest = load_manifest(args.manifest)
    data_root = Path(args.data_root)
    force = getattr(args, "force", False)

    only_models = getattr(args, "only_models", None)
    if only_models:
        only_set = set(only_models)
        manifest_ids = {s.model_id for s in manifest.models}
        unknown = only_set - manifest_ids
        if unknown:
            raise ValueError(f"Unknown model_ids in --only-models: {sorted(unknown)}")
        models_to_run = [s for s in manifest.models if s.model_id in only_set]
        LOGGER.info(
            "Restricted run to %d model(s): %s",
            len(models_to_run),
            [s.model_id for s in models_to_run],
        )
    else:
        models_to_run = list(manifest.models)

    for spec in models_to_run:
        revision = getattr(spec, "revision", None)
        work = []
        if spec.culture_enabled:
            for locale in manifest.culture_locales:
                country = CULTURE_COUNTRY_BY_LOCALE.get(locale)
                if _locale_enabled(spec, locale) and country is not None:
                    work.append(("blend_mcq", locale, [country], None))
        if spec.reasoning_enabled:
            for locale in manifest.reasoning_locales:
                if _locale_enabled(spec, locale):
                    work.append(("belebele", locale, None, [locale]))

        model = tokenizer = None
        try:
            for dataset, locale, countries, locales in work:
                subdir = f"{dataset}_{locale}"
                result_path = output_dir(model_slug(spec.name), subdir, revision=revision) / "results.csv"
                population = dict(
                    dataset_name=dataset, target_countries=countries,
                    target_locales=locales, data_root=data_root, target_data="all",
                )
                if not force and evaluation_cache_complete(result_path, **population):
                    LOGGER.info("Skip complete evaluation: %s", result_path)
                    continue
                if model is None:
                    model, tokenizer = load_model_and_tokenizer(
                        spec.name, dtype=args.dtype, devices=args.devices,
                        compile_model=getattr(args, "compile", False), revision=revision,
                    )
                _, path, acc = run_evaluation(
                    model_name=spec.name, **population,
                    batch_size=args.batch_size, dtype=args.dtype,
                    output_subdir=subdir, max_seq_len=args.max_seq_len,
                    revision=revision, model=model, tokenizer=tokenizer,
                    force=force, family=spec.architecture_family,
                )
                LOGGER.info("%s %s %s -> %.4f (%s)", dataset, spec.model_id, locale, acc, path)
        finally:
            del model, tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def _build_attribution_mask_comparison_rows(
    *,
    spec,
    target_locales: list[str],
    target_countries: list[str],
    revision: str | None,
    num_seeds: int,
    cross_locale: bool,
) -> list[dict]:
    """Per-cell mask_comparison rows (unmasked / cultural / random / SLR) via pure file I/O.

    eval_locales is diagonal-only by default, or every target_locale under cross_locale.
    """
    import math

    rows: list[dict] = []
    slug = model_slug(spec.name)
    base = output_dir(slug, "", revision=revision)

    for src_country, src_locale in zip(target_countries, target_locales):
        country_tag = src_country.replace(" ", "")
        neuron_stem = f"{country_tag}_neurons_blend_mcq"
        eval_locs = target_locales if cross_locale else [src_locale]

        for eval_loc in eval_locs:
            for ds in ("blend_mcq", "belebele", "countryrc_localized"):
                if ds == "blend_mcq":
                    ds_subdir = f"blend_mcq_{eval_loc}_non_neuron"
                    blend_td = "non_neuron"
                elif ds == "countryrc_localized":
                    ds_subdir = f"countryrc_localized_{eval_loc}_non_neuron"
                    blend_td = "non_neuron"
                else:
                    ds_subdir = f"belebele_{eval_loc}"
                    blend_td = "all"

                unmasked = _unmasked_accuracy(base, ds, eval_loc, blend_td)

                cultural = _read_accuracy_simple(
                    masked_output_dir(slug, neuron_stem, ds_subdir, revision=revision)
                    / "results.csv"
                )

                random_accs: list[float] = []
                slr_accs: list[float] = []
                for seed in range(num_seeds):
                    rp = (
                        masked_output_dir(
                            slug,
                            f"{country_tag}_all_neurons_random{seed}",
                            ds_subdir,
                            revision=revision,
                        )
                        / "results.csv"
                    )
                    acc = _read_accuracy_simple(rp)
                    if acc is not None:
                        random_accs.append(acc)
                    sp = (
                        masked_output_dir(
                            slug,
                            f"{country_tag}_all_neurons_random_layer{seed}",
                            ds_subdir,
                            revision=revision,
                        )
                        / "results.csv"
                    )
                    acc = _read_accuracy_simple(sp)
                    if acc is not None:
                        slr_accs.append(acc)

                def _mean_std(xs: list[float]):
                    if not xs:
                        return None, None
                    m = sum(xs) / len(xs)
                    s = math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))
                    return m, s

                r_mean, r_std = _mean_std(random_accs)
                slr_mean, slr_std = _mean_std(slr_accs)

                rows.append(
                    {
                        "model": spec.model_id,
                        "model_name": spec.name,
                        "regime": spec.role,
                        "revision": revision or "",
                        "src_country": src_country,
                        "src_locale": src_locale,
                        "eval_locale": eval_loc,
                        "is_diagonal": src_locale == eval_loc,
                        "dataset": ds,
                        "unmasked_acc": (
                            f"{unmasked:.4f}" if unmasked is not None else ""
                        ),
                        "cultural_masked_acc": (
                            f"{cultural:.4f}" if cultural is not None else ""
                        ),
                        "cultural_drop": (
                            f"{unmasked - cultural:.4f}"
                            if unmasked is not None and cultural is not None
                            else ""
                        ),
                        "random_masked_mean": (
                            f"{r_mean:.4f}" if r_mean is not None else ""
                        ),
                        "random_masked_std": (
                            f"{r_std:.4f}" if r_std is not None else ""
                        ),
                        "random_drop_mean": (
                            f"{unmasked - r_mean:.4f}"
                            if unmasked is not None and r_mean is not None
                            else ""
                        ),
                        "num_random_seeds": len(random_accs),
                        "slr_masked_mean": (
                            f"{slr_mean:.4f}" if slr_mean is not None else ""
                        ),
                        "slr_masked_std": (
                            f"{slr_std:.4f}" if slr_std is not None else ""
                        ),
                        "slr_drop_mean": (
                            f"{unmasked - slr_mean:.4f}"
                            if unmasked is not None and slr_mean is not None
                            else ""
                        ),
                        "num_slr_seeds": len(slr_accs),
                    }
                )
    return rows


def run_attribution(args) -> None:
    """Manifest-driven attribution: score -> select -> mask-eval -> mask-compare.

    Runs over each neuron_enabled spec's eval_locales (English excluded). Each stage
    skips work whose primary output exists; --force overwrites.
    """
    import argparse
    import torch
    from diss.core.adapters import get_adapter
    from diss.core.gpu import load_model_and_tokenizer, setup_torch_optimizations
    from diss.core.io import load_json, score_records_from_json
    from diss.core.locale_meta import COUNTRY_TO_LOCALE, LOCALE_TO_COUNTRY
    from diss.pipeline.attribute import run_scoring_cli
    from diss.pipeline.evaluate import _precompute_eval, evaluation_cache_complete, run_evaluation
    from diss.pipeline.mask import run_masked_evaluation
    from diss.pipeline.select import run_selection_cli
    from diss.prep import load_examples

    setup_torch_optimizations()
    manifest = load_manifest(args.manifest)

    only_models = getattr(args, "only_models", None)
    if only_models:
        only_set = set(only_models)
        unknown = only_set - {s.model_id for s in manifest.models}
        if unknown:
            raise ValueError(f"Unknown model_ids in --only-models: {sorted(unknown)}")
        models_to_run = [s for s in manifest.models if s.model_id in only_set]
    else:
        models_to_run = list(manifest.models)

    skipped = [s.model_id for s in models_to_run if not s.neuron_enabled]
    if skipped:
        LOGGER.info("Skipping neuron_enabled=False models: %s", skipped)
    models_to_run = [s for s in models_to_run if s.neuron_enabled]

    if not models_to_run:
        LOGGER.warning("No neuron_enabled models matched — nothing to do.")
        return

    LOGGER.info(
        "Attribution run over %d model(s): %s",
        len(models_to_run),
        [s.model_id for s in models_to_run],
    )

    force = getattr(args, "force", False)
    num_seeds = int(getattr(args, "num_seeds", 5))
    mlp_threshold = float(getattr(args, "mlp_threshold", 0.01))
    countryrc_threshold = float(getattr(args, "countryrc_threshold", 0.01))
    empty_cache_interval = int(getattr(args, "empty_cache_interval", 0))
    cross_locale = getattr(args, "cross_locale", False)
    data_root = Path(args.data_root)

    for spec in models_to_run:
        target_locales = [
            loc for loc in spec.eval_locales if loc in LOCALE_TO_COUNTRY and loc != "en"
        ]
        if not target_locales:
            LOGGER.warning(
                "Skip %s: no usable eval_locales (after en-exclusion)", spec.model_id
            )
            continue

        target_countries = [LOCALE_TO_COUNTRY[loc] for loc in target_locales]
        revision = getattr(spec, "revision", None)
        slug = model_slug(spec.name)
        score_dir = output_dir(slug, "cultural_neuron", revision=revision)
        LOGGER.info(
            "=== %s (slug=%s) locales=%s rev=%s ===",
            spec.model_id,
            slug,
            target_locales,
            revision or "default",
        )

        loaded: dict[str, object] = {"model": None, "tokenizer": None}

        def _ensure_loaded():
            if loaded["model"] is None:
                LOGGER.info(
                    "Loading %s (%s) rev=%s",
                    spec.model_id,
                    spec.name,
                    revision or "default",
                )
                m, t = load_model_and_tokenizer(
                    spec.name,
                    dtype=args.dtype,
                    devices=args.devices,
                    revision=revision,
                    compile_model=False,
                )
                loaded["model"] = m
                loaded["tokenizer"] = t
            return loaded["model"], loaded["tokenizer"]

        try:
            # Stage 1a: blend_mcq score (one call, all target countries)
            blend_score_path = score_dir / "blend_mcq_max_scores.json"
            if blend_score_path.exists() and not force:
                LOGGER.info("Skip blend_mcq score (exists): %s", blend_score_path)
            else:
                m, t = _ensure_loaded()
                run_scoring_cli(
                    argparse.Namespace(
                        model=spec.name,
                        family=spec.architecture_family,
                        datasets=["blend_mcq"],
                        target_countries=target_countries,
                        target_locales=None,
                        batch_size=args.batch_size,
                        dtype=args.dtype,
                        data_root=str(data_root),
                        target_data="neuron",
                        max_seq_len=args.max_seq_len,
                        devices=args.devices,
                        no_compile=False,
                        compile_mode=None,
                        revision=revision,
                        limit_examples=None,
                        empty_cache_interval=empty_cache_interval,
                    ),
                    model=m,
                    tokenizer=t,
                )

            # Stage 1b: countryrc_localized score per locale (CRC is locale-pinned).
            for loc in target_locales:
                crc_score_path = (
                    score_dir / f"countryrc_localized_{loc}_max_scores.json"
                )
                if crc_score_path.exists() and not force:
                    LOGGER.info("Skip CRC score (exists): %s", crc_score_path)
                    continue
                m, t = _ensure_loaded()
                run_scoring_cli(
                    argparse.Namespace(
                        model=spec.name,
                        family=spec.architecture_family,
                        datasets=["countryrc_localized"],
                        target_countries=target_countries,
                        target_locales=[loc],
                        batch_size=args.batch_size,
                        dtype=args.dtype,
                        data_root=str(data_root),
                        target_data="neuron",
                        max_seq_len=args.max_seq_len,
                        devices=args.devices,
                        no_compile=False,
                        compile_mode=None,
                        revision=revision,
                        limit_examples=None,
                        empty_cache_interval=empty_cache_interval,
                    ),
                    model=m,
                    tokenizer=t,
                )

            # Stage 1c: CRC non_neuron unmasked baseline per locale. CRC has no topic
            # axis, so the held-out half (idx 25..49) must be evaluated explicitly here.
            for loc in target_locales:
                crc_unm_path = (
                    output_dir(
                        slug,
                        f"countryrc_localized_{loc}_non_neuron",
                        revision=revision,
                    )
                    / "results.csv"
                )
                if not force and evaluation_cache_complete(
                    crc_unm_path, dataset_name="countryrc_localized",
                    target_countries=None, target_locales=[loc],
                    data_root=data_root, target_data="non_neuron",
                ):
                    LOGGER.info(
                        "Skip CRC unmasked non_neuron (exists): %s", crc_unm_path
                    )
                    continue
                m, t = _ensure_loaded()
                run_evaluation(
                    model_name=spec.name,
                    dataset_name="countryrc_localized",
                    target_countries=None,
                    target_locales=[loc],
                    data_root=data_root,
                    batch_size=args.batch_size,
                    dtype=args.dtype,
                    max_seq_len=args.max_seq_len,
                    devices=args.devices,
                    target_data="non_neuron",
                    output_subdir=f"countryrc_localized_{loc}_non_neuron",
                    revision=revision,
                    model=m,
                    tokenizer=t,
                    force=force,
                    family=spec.architecture_family,
                )

            # Stage 2: select (no model). Cache blend/CRC score JSONs lazily; each
            # select would otherwise re-parse a 50 MB+ file ~126 times per spec.
            sel_cache: dict[str, object] = {}

            def _blend_payload() -> dict:
                if "blend_payload" not in sel_cache:
                    sel_cache["blend_payload"] = load_json(
                        score_dir / "blend_mcq_max_scores.json"
                    )
                return sel_cache["blend_payload"]  # type: ignore[return-value]

            def _blend_records():
                if "blend_records" not in sel_cache:
                    sel_cache["blend_records"] = score_records_from_json(
                        _blend_payload()["neuron_scores"]
                    )
                return sel_cache["blend_records"]

            def _crc_payload(loc: str) -> dict:
                key = f"crc_payload_{loc}"
                if key not in sel_cache:
                    sel_cache[key] = load_json(
                        score_dir / f"countryrc_localized_{loc}_max_scores.json"
                    )
                return sel_cache[key]  # type: ignore[return-value]

            def _crc_records(loc: str):
                key = f"crc_records_{loc}"
                if key not in sel_cache:
                    sel_cache[key] = score_records_from_json(
                        _crc_payload(loc)["neuron_scores"]
                    )
                return sel_cache[key]

            for country in target_countries:
                tag = country.replace(" ", "")
                crc_loc = COUNTRY_TO_LOCALE[country]
                culture_path = score_dir / f"{tag}_neurons_blend_mcq.json"
                base_sel = dict(
                    model=spec.name,
                    family=spec.architecture_family,
                    dataset="blend_mcq",
                    target_country=country,
                    mlp_threshold=mlp_threshold,
                    countryrc_threshold=countryrc_threshold,
                    revision=revision,
                    output_subdir=None,
                )

                if culture_path.exists() and not force:
                    LOGGER.info("Skip culture select (exists): %s", culture_path)
                else:
                    run_selection_cli(
                        argparse.Namespace(
                            **base_sel,
                            selection_kind="culture",
                            random_seed=0,
                            match_neuron_file=None,
                        ),
                        score_payload=_blend_payload(),
                        crc_payload=_crc_payload(crc_loc),
                        score_records=_blend_records(),
                        crc_records=_crc_records(crc_loc),
                    )

                for seed in range(num_seeds):
                    rand_path = score_dir / f"{tag}_all_neurons_random{seed}.json"
                    if rand_path.exists() and not force:
                        LOGGER.info("Skip random seed=%d (exists)", seed)
                    else:
                        run_selection_cli(
                            argparse.Namespace(
                                **base_sel,
                                selection_kind="random",
                                random_seed=seed,
                                match_neuron_file=str(culture_path),
                            ),
                            score_payload=_blend_payload(),
                            score_records=_blend_records(),
                        )

                    slr_path = score_dir / f"{tag}_all_neurons_random_layer{seed}.json"
                    if slr_path.exists() and not force:
                        LOGGER.info("Skip same_layer_random seed=%d (exists)", seed)
                    else:
                        run_selection_cli(
                            argparse.Namespace(
                                **base_sel,
                                selection_kind="same_layer_random",
                                random_seed=seed,
                                match_neuron_file=str(culture_path),
                            ),
                            score_payload=_blend_payload(),
                            score_records=_blend_records(),
                        )

            # Stage 3: mask-eval (reuse the loaded model). Culture neuron files are
            # diagonal, plus off-diagonal under --cross-locale; random/SLR baselines
            # stay diagonal only. Lazy _ensure_loaded means a fully-cached resume
            # never touches the GPU.
            def _enqueue(work, neuron_file: Path, eval_loc: str) -> None:
                """Append BLEnD/Belebele/CRC mask-eval cells for (neuron_file, eval_loc), skipping existing."""
                stem = neuron_file.stem
                eval_country = LOCALE_TO_COUNTRY[eval_loc]
                blend_sub = f"blend_mcq_{eval_loc}_non_neuron"
                blend_out = (
                    masked_output_dir(slug, stem, blend_sub, revision=revision)
                    / "results.csv"
                )
                if not force and evaluation_cache_complete(
                    blend_out, dataset_name="blend_mcq", target_countries=[eval_country],
                    target_locales=None, data_root=data_root, target_data="non_neuron",
                ):
                    LOGGER.info("Skip mask blend (exists): %s", blend_out)
                else:
                    work.append(
                        dict(
                            neuron_file=str(neuron_file),
                            dataset="blend_mcq",
                            target_countries=[eval_country],
                            target_locales=None,
                            target_data="non_neuron",
                            output_subdir=blend_sub,
                        )
                    )
                bele_sub = f"belebele_{eval_loc}"
                bele_out = (
                    masked_output_dir(slug, stem, bele_sub, revision=revision)
                    / "results.csv"
                )
                if not force and evaluation_cache_complete(
                    bele_out, dataset_name="belebele", target_countries=None,
                    target_locales=[eval_loc], data_root=data_root, target_data="all",
                ):
                    LOGGER.info("Skip mask belebele (exists): %s", bele_out)
                else:
                    work.append(
                        dict(
                            neuron_file=str(neuron_file),
                            dataset="belebele",
                            target_countries=None,
                            target_locales=[eval_loc],
                            target_data="all",
                            output_subdir=bele_sub,
                        )
                    )
                crc_sub = f"countryrc_localized_{eval_loc}_non_neuron"
                crc_out = (
                    masked_output_dir(slug, stem, crc_sub, revision=revision)
                    / "results.csv"
                )
                if not force and evaluation_cache_complete(
                    crc_out, dataset_name="countryrc_localized", target_countries=None,
                    target_locales=[eval_loc], data_root=data_root, target_data="non_neuron",
                ):
                    LOGGER.info("Skip mask CRC (exists): %s", crc_out)
                else:
                    work.append(
                        dict(
                            neuron_file=str(neuron_file),
                            dataset="countryrc_localized",
                            target_countries=None,
                            target_locales=[eval_loc],
                            target_data="non_neuron",
                            output_subdir=crc_sub,
                        )
                    )

            def _has_neurons(p: Path) -> bool:
                if not p.exists():
                    return False
                try:
                    return len(load_json(p).get("top_neurons", [])) > 0
                except (OSError, ValueError):
                    return False

            mask_work: list[dict] = []
            # 3a. Culture neuron files
            for src_loc, src_country in zip(target_locales, target_countries):
                tag = src_country.replace(" ", "")
                culture_nf = score_dir / f"{tag}_neurons_blend_mcq.json"
                if not _has_neurons(culture_nf):
                    LOGGER.warning(
                        "Skip mask-eval for %s (0 culture neurons in %s)",
                        src_country,
                        culture_nf.name,
                    )
                    continue
                eval_locs = target_locales if cross_locale else [src_loc]
                for eval_loc in eval_locs:
                    _enqueue(mask_work, culture_nf, eval_loc)

            # 3b. Random + same-layer-random baselines (diagonal only)
            for src_loc, src_country in zip(target_locales, target_countries):
                tag = src_country.replace(" ", "")
                culture_nf = score_dir / f"{tag}_neurons_blend_mcq.json"
                if not _has_neurons(culture_nf):
                    continue  # already warned in 3a
                for seed in range(num_seeds):
                    _enqueue(
                        mask_work,
                        score_dir / f"{tag}_all_neurons_random{seed}.json",
                        src_loc,
                    )
                    _enqueue(
                        mask_work,
                        score_dir / f"{tag}_all_neurons_random_layer{seed}.json",
                        src_loc,
                    )

            if not mask_work:
                LOGGER.info(
                    "Skip mask-eval stage (all outputs exist): %s", spec.model_id
                )
            else:
                LOGGER.info(
                    "Mask-eval stage for %s: %d work items (cross_locale=%s)",
                    spec.model_id,
                    len(mask_work),
                    cross_locale,
                )
                model, tokenizer = _ensure_loaded()
                adapter = get_adapter(spec.name, family=spec.architecture_family)
                slice_cache: dict[tuple, tuple[list, dict]] = {}

                def _slice_key(w):
                    tc = tuple(w["target_countries"]) if w["target_countries"] else None
                    tl = tuple(w["target_locales"]) if w["target_locales"] else None
                    return (w["dataset"], tc, tl, w["target_data"])

                def _get_slice(w):
                    key = _slice_key(w)
                    cached = slice_cache.get(key)
                    if cached is not None:
                        return cached
                    examples = load_examples(
                        dataset_name=w["dataset"],
                        target_countries=w["target_countries"],
                        target_locales=w["target_locales"],
                        data_root=data_root,
                        target_data=w["target_data"],
                    )
                    encs = _precompute_eval(
                        examples, tokenizer, adapter, args.max_seq_len
                    )
                    slice_cache[key] = (examples, encs)
                    return examples, encs

                for w in mask_work:
                    examples, encs = _get_slice(w)
                    run_masked_evaluation(
                        model_name=spec.name,
                        dataset_name=w["dataset"],
                        neuron_file=w["neuron_file"],
                        target_countries=w["target_countries"],
                        target_locales=w["target_locales"],
                        data_root=data_root,
                        batch_size=args.batch_size,
                        dtype=args.dtype,
                        operation="suppress",
                        max_seq_len=args.max_seq_len,
                        devices=args.devices,
                        target_data=w["target_data"],
                        output_subdir=w["output_subdir"],
                        revision=revision,
                        model=model,
                        tokenizer=tokenizer,
                        examples=examples,
                        encs=encs,
                        family=spec.architecture_family,
                    )

            # Stage 4: consolidate {slug}/mask_comparison.csv (one row per
            # src_country/eval_locale/dataset cell) for all countries in the spec.
            mask_compare_path = (
                output_dir(slug, "", revision=revision) / "mask_comparison.csv"
            )
            rows = _build_attribution_mask_comparison_rows(
                spec=spec,
                target_locales=target_locales,
                target_countries=target_countries,
                revision=revision,
                num_seeds=num_seeds,
                cross_locale=cross_locale,
            )
            _write_csv(mask_compare_path, rows)
            LOGGER.info(
                "Saved mask_comparison.csv for %s (%d rows): %s",
                spec.model_id,
                len(rows),
                mask_compare_path,
            )
        finally:
            import gc

            loaded["model"] = None
            loaded["tokenizer"] = None
            m = t = model = tokenizer = adapter = None
            for _cache_name in ("sel_cache", "slice_cache"):
                _cache_obj = locals().get(_cache_name)
                if _cache_obj is not None:
                    _cache_obj.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def _read_accuracy(path: Path) -> tuple[float, int, int] | tuple[None, None, None]:
    if not path.exists():
        return None, None, None
    rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
    total = len(rows)
    correct = sum(1 for r in rows if str(r.get("is_correct", "")).lower() == "true")
    return (correct / total if total else 0.0), correct, total


def _read_accuracy_both(path: Path) -> dict:
    """Read results.csv as sum-, byte-, and token-normalized accuracy (default is_correct == byte)."""
    empty = {
        "acc_sum": None,
        "correct_sum": None,
        "acc_byte": None,
        "correct_byte": None,
        "acc_token": None,
        "correct_token": None,
        "total": None,
    }
    if not path.exists():
        return empty
    rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
    total = len(rows)
    if total == 0:
        return empty

    def _count(field: str) -> int | None:
        if not any(field in r for r in rows):
            return None
        return sum(1 for r in rows if str(r.get(field, "")).lower() == "true")

    correct_sum = _count("is_correct_sum")
    correct_byte = _count("is_correct_byte")
    correct_token = _count("is_correct_token")
    if correct_byte is None:
        correct_byte = sum(
            1 for r in rows if str(r.get("is_correct", "")).lower() == "true"
        )

    return {
        "acc_sum": (correct_sum / total) if correct_sum is not None else None,
        "correct_sum": correct_sum,
        "acc_byte": correct_byte / total,
        "correct_byte": correct_byte,
        "acc_token": (correct_token / total) if correct_token is not None else None,
        "correct_token": correct_token,
        "total": total,
    }


def _build_regime_rankings(
    manifest,
    behavior_rows: list[dict],
) -> list[dict]:
    """One row per (locale, size-bucket, task) ranking the regimes by byte-norm row accuracy."""
    from collections import defaultdict
    from diss.core.registry import MODEL_PARAMS

    id_to_role = {s.model_id: s.role for s in manifest.models}
    role_short = {"monolingual": "mono", "bilingual": "bi", "multilingual": "multi"}
    REGIMES = ["mono", "bi", "multi"]

    def _collect(field: str) -> dict[tuple[str, str], float]:
        out: dict[tuple[str, str], float] = {}
        for r in behavior_rows:
            if r[field] == "":
                continue
            try:
                out[(r["label"], r["locale"])] = float(r[field])
            except (TypeError, ValueError):
                continue
        return out

    cultural_acc = _collect("cultural_accuracy")
    reasoning_acc = _collect("reasoning_accuracy")

    # Bucket: (locale, size, task) -> {regime_short: list[(label, acc)]}
    def size_bucket(p: float) -> str:
        return "<1B" if p < 1e9 else "≥1B"

    buckets: dict[tuple[str, str, str], dict[str, list[tuple[str, float]]]] = (
        defaultdict(lambda: defaultdict(list))
    )
    for (label, loc), acc in cultural_acc.items():
        params = MODEL_PARAMS.get(label)
        role = id_to_role.get(label)
        if params is None or role is None:
            continue
        rs = role_short.get(role, role)
        buckets[(loc, size_bucket(params), "cultural")][rs].append((label, acc))
    for (label, loc), acc in reasoning_acc.items():
        params = MODEL_PARAMS.get(label)
        role = id_to_role.get(label)
        if params is None or role is None:
            continue
        rs = role_short.get(role, role)
        buckets[(loc, size_bucket(params), "reasoning")][rs].append((label, acc))

    from diss.core.registry import LOCALE_DISPLAY_ORDER

    locales_present = sorted(
        {key[0] for key in buckets},
        key=lambda l: (
            LOCALE_DISPLAY_ORDER.index(l)
            if l in LOCALE_DISPLAY_ORDER
            else len(LOCALE_DISPLAY_ORDER)
        ),
    )

    rows: list[dict] = []
    for loc in locales_present:
        for size in ["<1B", "≥1B"]:
            tasks_present = [
                t for t in ("cultural", "reasoning") if (loc, size, t) in buckets
            ]
            if not tasks_present:
                rows.append(
                    {
                        "locale": loc,
                        "model_size": "—",
                        "task": "—",
                        "acc_rank": f"(no {size} model has {loc} in eval_locales)",
                    }
                )
                continue
            for task in ("cultural", "reasoning"):
                regime_to_models = buckets.get((loc, size, task))
                if not regime_to_models:
                    continue
                ranking: list[tuple[str, float]] = []
                for r in REGIMES:
                    items = regime_to_models.get(r, [])
                    if not items:
                        continue
                    mean_acc = sum(a for _, a in items) / len(items)
                    ranking.append((r, mean_acc))
                ranking.sort(key=lambda kv: -kv[1])
                rank_str = " > ".join(f"{r}({a * 100:.1f})" for r, a in ranking)
                if len(ranking) == 1:
                    rank_str += " only"
                rows.append(
                    {
                        "locale": loc,
                        "model_size": size,
                        "task": task,
                        "acc_rank": rank_str,
                    }
                )
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def run_summary(args) -> None:
    manifest = load_manifest(args.manifest)
    root = OUTPUT_ROOT

    behavior_rows: list[dict] = []
    for spec in manifest.models:
        slug = model_slug(spec.name)
        rev = getattr(spec, "revision", None)

        cult = {}
        for loc in manifest.culture_locales:
            if loc not in spec.eval_locales or not spec.culture_enabled:
                continue
            cult_dir = output_dir(slug, f"blend_mcq_{loc}", revision=rev, root=root)
            stats = _read_accuracy_both(cult_dir / "results.csv")
            if stats["acc_byte"] is not None:
                cult[loc] = {
                    "ds": f"blend_mcq_{loc}",
                    **stats,
                    "path": str(cult_dir / "results.csv"),
                    "country": CULTURE_COUNTRY_BY_LOCALE[loc],
                }

        reas = {}
        for loc in manifest.reasoning_locales:
            if loc not in spec.eval_locales or not spec.reasoning_enabled:
                continue
            reas_dir = output_dir(slug, f"belebele_{loc}", revision=rev, root=root)
            stats = _read_accuracy_both(reas_dir / "results.csv")
            if stats["acc_byte"] is not None:
                reas[loc] = {
                    "ds": f"belebele_{loc}",
                    **stats,
                    "path": str(reas_dir / "results.csv"),
                }

        for loc in sorted(set(cult) | set(reas)):
            c, r = cult.get(loc), reas.get(loc)
            behavior_rows.append(
                {
                    "locale": loc,
                    "regime": spec.role,
                    "label": spec.model_id,
                    "model_name": spec.name,
                    "culture_country": c["country"] if c else "",
                    "cultural_accuracy": c["acc_byte"] if c else "",
                    "cultural_correct": c["correct_byte"] if c else "",
                    "cultural_total": c["total"] if c else "",
                    "cultural_accuracy_sum": (
                        c["acc_sum"] if c and c["acc_sum"] is not None else ""
                    ),
                    "cultural_correct_sum": (
                        c["correct_sum"] if c and c["correct_sum"] is not None else ""
                    ),
                    "cultural_accuracy_token": (
                        c["acc_token"] if c and c["acc_token"] is not None else ""
                    ),
                    "cultural_correct_token": (
                        c["correct_token"]
                        if c and c["correct_token"] is not None
                        else ""
                    ),
                    "reasoning_accuracy": r["acc_byte"] if r else "",
                    "reasoning_correct": r["correct_byte"] if r else "",
                    "reasoning_total": r["total"] if r else "",
                    "reasoning_accuracy_sum": (
                        r["acc_sum"] if r and r["acc_sum"] is not None else ""
                    ),
                    "reasoning_correct_sum": (
                        r["correct_sum"] if r and r["correct_sum"] is not None else ""
                    ),
                    "reasoning_accuracy_token": (
                        r["acc_token"] if r and r["acc_token"] is not None else ""
                    ),
                    "reasoning_correct_token": (
                        r["correct_token"]
                        if r and r["correct_token"] is not None
                        else ""
                    ),
                }
            )

    _write_csv(OUTPUT_ROOT / "behavior_summary.csv", behavior_rows)

    main_rows = []
    for loc in manifest.culture_locales:
        lr = [
            r
            for r in behavior_rows
            if r["locale"] == loc and r["cultural_accuracy"] != ""
        ]
        if {"monolingual", "bilingual", "multilingual"}.issubset(
            {r["regime"] for r in lr}
        ):
            main_rows.extend(lr)
    _write_csv(OUTPUT_ROOT / "main_tradeoff_summary.csv", main_rows)

    aux_rows = [r for r in behavior_rows if r["reasoning_accuracy"] != ""]
    _write_csv(OUTPUT_ROOT / "aux_reasoning_summary.csv", aux_rows)

    regime_rows = _build_regime_rankings(manifest, behavior_rows)
    _write_csv(OUTPUT_ROOT / "regime_ranking_summary.csv", regime_rows)

    LOGGER.info("Summaries written")


def _read_accuracy_simple(path: Path) -> float | None:
    acc, _, _ = _read_accuracy(path)
    return acc


def _unmasked_accuracy(
    base_dir: Path,
    ds: str,
    locale: str,
    blend_td: str,
) -> float | None:
    """Unmasked baseline accuracy. BLEnD non_neuron is derived by topic-filtering the
    all-topic eval; Belebele and blend `all` read the whole results.csv."""
    if ds == "blend_mcq" and blend_td == "non_neuron":
        path = base_dir / f"blend_mcq_{locale}" / "results.csv"
        if not path.exists():
            return None
        return results_accuracy(read_results_csv(path), topics=NON_NEURON_TOPICS)[0]
    if ds == "countryrc_localized" and blend_td == "non_neuron":
        path = base_dir / f"countryrc_localized_{locale}_non_neuron" / "results.csv"
        if not path.exists():
            return None
        return _read_accuracy_simple(path)
    sub = f"{ds}_{locale}" if locale else ds
    return _read_accuracy_simple(base_dir / sub / "results.csv")
