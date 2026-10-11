from __future__ import annotations

import contextlib
import json
import logging
import struct
import urllib.request

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import transformers.modeling_utils as _tm

if not hasattr(_tm, "SequenceSummary"):

    class _SequenceSummaryShim:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError(
                "SequenceSummary placeholder (transformers>=4.43); "
                "this build supports CausalLM heads only."
            )

    _tm.SequenceSummary = _SequenceSummaryShim


LOGGER = logging.getLogger(__name__)

_TORCH_DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def _safetensors_vocab_size(model_name: str, revision: str | None) -> int | None:
    """Read embed_tokens row count from the safetensors header, or None on failure.

    Lets callers override a config.json whose vocab_size disagrees with the real
    weight shape (seen on some intermediate-checkpoint repos).
    """
    rev = revision or "main"
    url = f"https://huggingface.co/{model_name}/resolve/{rev}/model.safetensors"
    try:
        req = urllib.request.Request(url, headers={"Range": "bytes=0-65535"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = resp.read()
        n = struct.unpack("<Q", data[:8])[0]
        header = json.loads(data[8 : 8 + n].decode())
        for key, meta in header.items():
            if key.endswith("embed_tokens.weight") or key.endswith("wte.weight"):
                shape = meta.get("shape")
                if shape and len(shape) == 2:
                    return int(shape[0])
    except Exception as exc:
        LOGGER.debug(
            "safetensors vocab probe failed for %s rev=%s: %s", model_name, rev, exc
        )
    return None


def setup_torch_optimizations() -> None:
    """Enable A100-optimized settings. Call once per process."""
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)

    # Cache compiled variants for multiple bucketed shapes.
    try:
        import torch._dynamo as _dynamo

        _dynamo.config.cache_size_limit = 64
    except (ImportError, AttributeError):
        pass

    LOGGER.info(
        "GPU opts: TF32=%s flash_sdp=%s cudnn_bench=%s %d GPU(s)",
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cuda.flash_sdp_enabled(),
        torch.backends.cudnn.benchmark,
        torch.cuda.device_count(),
    )


def get_autocast_ctx(dtype: str = "bfloat16"):
    """Return autocast context for available device."""
    if torch.cuda.is_available():
        return torch.autocast("cuda", dtype=_TORCH_DTYPE_MAP.get(dtype, torch.bfloat16))
    return contextlib.nullcontext()


def _patch_tokenizer_class(model_name: str) -> None:
    """Fix HuggingFace-cached tokenizer_config.json if tokenizer_class is invalid."""
    import json
    from huggingface_hub import hf_hub_download

    try:
        cached = hf_hub_download(model_name, "tokenizer_config.json")
    except Exception:
        return
    try:
        with open(cached, "r", encoding="utf-8") as f:
            config = json.load(f)
        if config.get("tokenizer_class") == "TokenizersBackend":
            LOGGER.warning(
                "Patching tokenizer_class 'TokenizersBackend' -> 'PreTrainedTokenizerFast' for %s",
                model_name,
            )
            config["tokenizer_class"] = "PreTrainedTokenizerFast"
            with open(cached, "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def load_model_and_tokenizer(
    model_name: str,
    *,
    adapter_path: str | None = None,
    dtype: str = "bfloat16",
    devices: list[int] | None = None,
    compile_model: bool | None = None,
    compile_mode: str = "reduce-overhead",
    revision: str | None = None,
):
    """Load model + tokenizer, fastest attention first (flash_attention_2 → sdpa → eager).

    compile_model None auto-enables on single-GPU only. compile_mode "default" suits
    forward+backward paths, "reduce-overhead" inference-only paths.
    """
    _patch_tokenizer_class(model_name)
    tokenizer_kwargs = dict(padding_side="left", trust_remote_code=True)
    if revision:
        tokenizer_kwargs["revision"] = revision
    tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = _TORCH_DTYPE_MAP[dtype]
    multi_gpu = devices is not None and len(devices) > 1 and torch.cuda.is_available()

    load_kwargs = dict(torch_dtype=torch_dtype, trust_remote_code=True)
    if revision:
        load_kwargs["revision"] = revision

    actual_vocab = _safetensors_vocab_size(model_name, revision)
    if actual_vocab is not None:
        cfg_kwargs = {"trust_remote_code": True}
        if revision:
            cfg_kwargs["revision"] = revision
        try:
            cfg = AutoConfig.from_pretrained(model_name, **cfg_kwargs)
            cfg_vocab = getattr(cfg, "vocab_size", None)
            if cfg_vocab is not None and cfg_vocab != actual_vocab:
                LOGGER.info(
                    "Vocab mismatch for %s rev=%s: config=%d, weights=%d — "
                    "overriding config.vocab_size to match weights",
                    model_name,
                    revision or "default",
                    cfg_vocab,
                    actual_vocab,
                )
                cfg.vocab_size = actual_vocab
                load_kwargs["config"] = cfg

                for attr in ("pad_token_id", "eos_token_id", "bos_token_id"):
                    tok_id = getattr(tokenizer, attr, None)
                    if tok_id is not None and tok_id >= actual_vocab:
                        LOGGER.info(
                            "Clamping tokenizer.%s=%d → 0 (model vocab=%d)",
                            attr,
                            tok_id,
                            actual_vocab,
                        )
                        setattr(tokenizer, attr, 0)
        except Exception as exc:
            LOGGER.debug(
                "config probe failed for %s rev=%s: %s",
                model_name,
                revision or "default",
                exc,
            )

    model = None
    for attn_impl in ("flash_attention_2", "sdpa", "eager"):
        try:
            if multi_gpu:
                max_memory = {d: "75GiB" for d in devices}
                max_memory["cpu"] = "32GiB"
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    **load_kwargs,
                    attn_implementation=attn_impl,
                    device_map="auto",
                    max_memory=max_memory,
                )
                LOGGER.info(
                    "Loaded %s rev=%s across GPUs %s (attn=%s)",
                    model_name,
                    revision or "default",
                    devices,
                    attn_impl,
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    **load_kwargs,
                    attn_implementation=attn_impl,
                )
                device_id = devices[0] if devices else 0
                device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
                model.to(device)
                LOGGER.info(
                    "Loaded %s rev=%s on %s (attn=%s)",
                    model_name,
                    revision or "default",
                    device,
                    attn_impl,
                )
            break
        except (ValueError, ImportError, RuntimeError) as exc:
            if attn_impl == "eager":
                raise
            LOGGER.info("attn=%s unsupported (%s), trying next", attn_impl, exc)

    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)

    if compile_model is None:
        compile_model = (not multi_gpu) and torch.cuda.is_available()
    if compile_model and torch.cuda.is_available() and not multi_gpu:
        LOGGER.info(
            "torch.compile(mode=%s, dynamic=True, fullgraph=False)", compile_mode
        )
        model = torch.compile(model, mode=compile_mode, dynamic=True, fullgraph=False)

    return model, tokenizer
