from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Dissertation experiment CLI")
    sub = p.add_subparsers(dest="command", required=True)

    ev = sub.add_parser("eval", help="Evaluate a model on a dataset")
    ev.add_argument("--model", required=True)
    ev.add_argument(
        "--dataset",
        required=True,
        choices=["blend_mcq", "countryrc_localized", "belebele"],
    )
    ev.add_argument("--target-countries", nargs="+", default=["South Korea", "Spain"])
    ev.add_argument("--target-locales", nargs="+", default=None)
    ev.add_argument(
        "--target-data", default="all", choices=["all", "neuron", "non_neuron"]
    )
    ev.add_argument("--output-subdir", default=None)
    ev.add_argument("--batch-size", type=int, default=8)
    ev.add_argument(
        "--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"]
    )
    ev.add_argument("--data-root", default="diss/data")
    ev.add_argument("--max-seq-len", type=int, default=1024)
    ev.add_argument("--devices", nargs="+", type=int, default=None)
    ev.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable torch.compile (recommended for short debug or equivalence runs).",
    )
    ev.add_argument(
        "--compile-mode",
        choices=["default", "reduce-overhead"],
        default=None,
        help="torch.compile mode. Auto: 'reduce-overhead' for inference, 'default' for grad. "
        "Pass 'default' for the score path to avoid CUDA Graph + autograd issues.",
    )
    ev.add_argument(
        "--revision",
        default=None,
        help="Hugging Face revision; if set, outputs go under cp_{revision}/",
    )
    ev.add_argument(
        "--limit-examples",
        type=int,
        default=None,
        help="Cap the number of examples after load_examples() (debug / Tier 1 baselines).",
    )
    ev.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Re-evaluate even if results.csv already exists (deletes existing output).",
    )

    sc = sub.add_parser("score", help="Compute neuron attribution scores")
    sc.add_argument("--model", required=True)
    sc.add_argument("--datasets", nargs="+", required=True)
    sc.add_argument("--target-countries", nargs="+", default=["South Korea", "Spain"])
    sc.add_argument(
        "--target-locales",
        nargs="+",
        default=None,
        help="Required for countryrc_localized: locale(s) to render CRC stimuli in. "
        "Ignored by country-pinned datasets (blend_mcq, etc.).",
    )
    sc.add_argument("--batch-size", type=int, default=8)
    sc.add_argument(
        "--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"]
    )
    sc.add_argument("--data-root", default="diss/data")
    sc.add_argument(
        "--target-data", default="neuron", choices=["all", "neuron", "non_neuron"]
    )
    sc.add_argument("--max-seq-len", type=int, default=1024)
    sc.add_argument("--devices", nargs="+", type=int, default=None)
    sc.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable torch.compile (recommended for short debug or equivalence runs).",
    )
    sc.add_argument(
        "--compile-mode",
        choices=["default", "reduce-overhead"],
        default=None,
        help="torch.compile mode. Auto: 'reduce-overhead' for inference, 'default' for grad. "
        "Pass 'default' for the score path to avoid CUDA Graph + autograd issues.",
    )
    sc.add_argument("--revision", default=None)
    sc.add_argument(
        "--limit-examples",
        type=int,
        default=None,
        help="Cap the number of examples after load_examples() (debug / Tier 1 baselines).",
    )
    sc.add_argument(
        "--empty-cache-interval",
        type=int,
        default=0,
        help="Run gc.collect+torch.cuda.empty_cache every N grad-scoring batches. "
        "Default 0 leaves it to the CUDA caching allocator. Bump to a small "
        "integer (e.g. 50) only when monitoring shows allocator drift on 7B+.",
    )

    se = sub.add_parser("select", help="Select cultural / random neurons")
    se.add_argument("--model", required=True)
    se.add_argument("--dataset", default="blend_mcq")
    se.add_argument("--target-country", default="South Korea")
    se.add_argument(
        "--mlp-threshold",
        type=float,
        default=0.01,
        help="CULNIG-aligned MLP gate-projection selection proportion "
        "(default: 0.01 = top 1%, matches CULNIG). Tunable for "
        "sparsity sensitivity sweeps; see "
        "docs/culnig_gate_proj_split_decisions.md.",
    )
    se.add_argument(
        "--countryrc-threshold",
        type=float,
        default=0.01,
        help="CountryRC filter proportion, applied to the MLP pool "
        "(default: 0.01 = top 1%, fixed by the methodological decision).",
    )
    se.add_argument(
        "--selection-kind",
        default="culture",
        choices=["culture", "random", "same_layer_random"],
    )
    se.add_argument("--random-seed", type=int, default=0)
    se.add_argument(
        "--match-neuron-file",
        default=None,
        help="For random selection: sample exactly the same count as this selected file. "
        "Required for same_layer_random (per-layer histogram matched).",
    )
    se.add_argument(
        "--output-subdir",
        default=None,
        help="Output subdirectory under the model slug (default: cultural_neuron). "
        "Use e.g. cultural_neuron_tau0001 for sparsity sensitivity.",
    )
    se.add_argument("--revision", default=None)

    mk = sub.add_parser("mask-eval", help="Evaluate with neuron masking")
    mk.add_argument("--model", required=True)
    mk.add_argument(
        "--dataset",
        required=True,
        choices=["blend_mcq", "countryrc_localized", "belebele"],
    )
    mk.add_argument("--neuron-file", required=True)
    mk.add_argument("--target-countries", nargs="+", default=["South Korea", "Spain"])
    mk.add_argument("--target-locales", nargs="+", default=None)
    mk.add_argument(
        "--target-data", default="all", choices=["all", "neuron", "non_neuron"]
    )
    mk.add_argument("--output-subdir", default=None)
    mk.add_argument("--batch-size", type=int, default=8)
    mk.add_argument(
        "--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"]
    )
    mk.add_argument("--data-root", default="diss/data")
    mk.add_argument("--max-seq-len", type=int, default=1024)
    mk.add_argument("--devices", nargs="+", type=int, default=None)
    mk.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable torch.compile (recommended for short debug or equivalence runs).",
    )
    mk.add_argument(
        "--compile-mode",
        choices=["default", "reduce-overhead"],
        default=None,
        help="torch.compile mode. Auto: 'reduce-overhead' for inference, 'default' for grad. "
        "Pass 'default' for the score path to avoid CUDA Graph + autograd issues.",
    )
    mk.add_argument("--revision", default=None)
    mk.add_argument(
        "--limit-examples",
        type=int,
        default=None,
        help="Cap the number of examples after load_examples() (debug / Tier 1 baselines).",
    )

    rb = sub.add_parser(
        "run-behavior", help="Run manifest-driven behavioral evaluation"
    )
    rb.add_argument("--manifest", default="diss/config/model_manifest.yaml")
    rb.add_argument("--data-root", default="diss/data")
    rb.add_argument("--batch-size", type=int, default=8)
    rb.add_argument(
        "--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"]
    )
    rb.add_argument("--max-seq-len", type=int, default=1024)
    rb.add_argument("--devices", nargs="+", type=int, default=None)
    rb.add_argument("--compile", action="store_true", default=False)
    rb.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Re-run evaluations even if results.csv already exists",
    )
    rb.add_argument(
        "--only-models",
        nargs="+",
        default=None,
        help="Restrict the run to these manifest model_ids (default: all)",
    )

    ra = sub.add_parser(
        "run-attribution",
        help="Run manifest-driven attribution pipeline (score + select + mask-eval) "
        "for every neuron_enabled model",
    )
    ra.add_argument("--manifest", default="diss/config/model_manifest.yaml")
    ra.add_argument("--data-root", default="diss/data")
    ra.add_argument("--batch-size", type=int, default=8)
    ra.add_argument(
        "--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"]
    )
    ra.add_argument("--max-seq-len", type=int, default=1024)
    ra.add_argument("--devices", nargs="+", type=int, default=None)
    ra.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Re-run stages even if expected outputs already exist",
    )
    ra.add_argument(
        "--only-models",
        nargs="+",
        default=None,
        help="Restrict the run to these manifest model_ids. "
        "Models with neuron_enabled=False are still skipped.",
    )
    ra.add_argument(
        "--num-seeds",
        type=int,
        default=5,
        help="Number of random / same-layer-random baseline seeds per country",
    )
    ra.add_argument(
        "--mlp-threshold",
        type=float,
        default=0.01,
        help="Top-fraction MLP gate-projection selection (default 0.01 = top 1%%, matches CULNIG)",
    )
    ra.add_argument(
        "--countryrc-threshold",
        type=float,
        default=0.01,
        help="CountryRC filter top-fraction over the MLP pool (default 0.01)",
    )
    ra.add_argument(
        "--empty-cache-interval",
        type=int,
        default=0,
        help="Forwarded to the score stage: gc.collect+empty_cache cadence "
        "(0 = let the CUDA caching allocator manage)",
    )
    ra.add_argument(
        "--cross-locale",
        action="store_true",
        default=False,
        help="In mask-eval, also evaluate each culture neuron file on the "
        "OTHER eval_locales (off-diagonal cells). e.g. ko-specific "
        "neurons masked, drop measured on es / zh / ar / ja / fr datasets. "
        "Random / same-layer-random baselines stay diagonal-only "
        "(matches checkpoint_trace.py convention).",
    )

    ct = sub.add_parser(
        "run-checkpoint-trace", help="Run checkpoint trajectory experiment"
    )
    ct.add_argument(
        "--config",
        default="diss/config/checkpoint_trace.yaml",
        help="Path to checkpoint trace config YAML",
    )
    ct.add_argument(
        "--run-id",
        default=None,
        help="If set, only this run id from the config is executed",
    )
    ct.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Re-run stages even if outputs already exist",
    )

    sm = sub.add_parser(
        "summarize", help="Build behavior summaries from manifest outputs"
    )
    sm.add_argument("--manifest", default="diss/config/model_manifest.yaml")

    return p


def setup_logging() -> Path:
    log_dir = Path("diss/outputs/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    # DISS_LOG_FILE lets fan-out wrappers append all invocations to one file;
    # otherwise one file per day.
    override = os.environ.get("DISS_LOG_FILE")
    log_file = (
        Path(override)
        if override
        else (log_dir / f"run_{datetime.now().strftime('%Y%m%d')}.log")
    )
    log_file.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    root.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    root.addHandler(ch)

    return log_file


def main() -> None:
    log_file = setup_logging()
    logging.getLogger(__name__).info("Log file: %s", log_file)

    args = build_parser().parse_args()
    cmd = args.command

    if cmd == "eval":
        from diss.core.gpu import setup_torch_optimizations
        from diss.pipeline.evaluate import run_evaluation

        setup_torch_optimizations()
        compile_mode_arg = getattr(args, "compile_mode", None)
        no_compile = getattr(args, "no_compile", False)
        compile_model_arg = False if (no_compile or compile_mode_arg is None) else True
        compile_mode_arg = compile_mode_arg or "reduce-overhead"
        results, path, acc = run_evaluation(
            model_name=args.model,
            dataset_name=args.dataset,
            target_countries=args.target_countries,
            target_locales=args.target_locales,
            data_root=Path(args.data_root),
            batch_size=args.batch_size,
            dtype=args.dtype,
            max_seq_len=args.max_seq_len,
            devices=args.devices,
            compile_model=compile_model_arg,
            compile_mode=compile_mode_arg,
            target_data=args.target_data,
            output_subdir=args.output_subdir,
            revision=args.revision,
            limit_examples=getattr(args, "limit_examples", None),
            force=getattr(args, "force", False),
        )
        logging.getLogger(__name__).info(
            "Acc %.4f (%d/%d) -> %s",
            acc,
            sum(1 for r in results if r["is_correct"]),
            len(results),
            path,
        )

    elif cmd == "score":
        from diss.pipeline.attribute import run_scoring_cli

        run_scoring_cli(args)

    elif cmd == "select":
        from diss.pipeline.select import run_selection_cli

        run_selection_cli(args)

    elif cmd == "mask-eval":
        from diss.pipeline.mask import run_mask_eval_cli

        run_mask_eval_cli(args)

    elif cmd == "run-behavior":
        from diss.run import run_behavior

        run_behavior(args)

    elif cmd == "run-attribution":
        from diss.run import run_attribution

        run_attribution(args)

    elif cmd == "summarize":
        from diss.run import run_summary

        run_summary(args)

    elif cmd == "run-checkpoint-trace":
        from diss.pipeline.checkpoint_trace import run_checkpoint_trace

        run_checkpoint_trace(args)


if __name__ == "__main__":
    main()
