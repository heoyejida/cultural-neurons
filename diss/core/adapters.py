from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable

CULNIG_MLP_MODULES = ("mlp.gate_proj",)
CULNIG_ATTENTION_MODULES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
)
CULNIG_SCORE_MODULES = list(CULNIG_MLP_MODULES)


def _linear_width(module, kind: str) -> int:
    if hasattr(module, "out_features") and kind == "post":
        return int(module.out_features)
    if hasattr(module, "in_features") and kind == "pre":
        return int(module.in_features)
    if hasattr(module, "nf"):
        return int(module.nf if kind == "post" else module.weight.shape[0])
    w = getattr(module, "weight", None)
    if w is None:
        raise ValueError(f"Cannot infer feature width for {module!r}")
    return int(w.shape[0] if kind == "post" else w.shape[1])


@dataclass
class ArchitectureAdapter(ABC):
    family: str

    @abstractmethod
    def get_backbone(self, model):
        raise NotImplementedError

    def get_num_layers(self, model) -> int:
        return len(self.get_layers(model))

    @abstractmethod
    def get_layers(self, model):
        raise NotImplementedError

    @abstractmethod
    def resolve_module(self, model, layer_idx: int, logical_module_name: str):
        raise NotImplementedError

    def prepend_token_id(self, tokenizer) -> int | None:
        return (
            getattr(tokenizer, "bos_token_id", None)
            or getattr(tokenizer, "eos_token_id", None)
            or getattr(tokenizer, "cls_token_id", None)
        )

    @abstractmethod
    def default_score_modules(self) -> list[str]:
        raise NotImplementedError

    def module_hook_kind(self, logical_module_name: str) -> str:
        """Forward-hook kind: "post" (output) by default, "pre" when the logical neuron is a projection's input."""
        return "post"

    def extract_logical_tensor(
        self, model, layer_idx: int, logical_module_name: str, tensor
    ):
        return tensor

    def logical_module_size(
        self, model, layer_idx: int, logical_module_name: str
    ) -> int:
        module = self.resolve_module(model, layer_idx, logical_module_name)
        return _linear_width(module, self.module_hook_kind(logical_module_name))

    def prepare_for_scoring(self, model, max_seq_len: int) -> None:
        """Called once at the top of attribution scoring; no-op unless a lazy cache (e.g. RoPE freqs_cis) must be built outside inference_mode."""
        return None

    def translate_output_indices(
        self,
        model,
        layer_idx: int,
        logical_module_name: str,
        neuron_indices: Iterable[int],
    ) -> list[int]:
        return list(neuron_indices)


class GPT2LikeAdapter(ArchitectureAdapter):
    def __init__(self) -> None:
        super().__init__(family="gpt2_like")

    def get_backbone(self, model):
        if hasattr(model, "transformer"):
            return model.transformer
        raise ValueError("Unsupported gpt2-like model structure")

    def get_layers(self, model):
        return self.get_backbone(model).h

    def resolve_module(self, model, layer_idx: int, logical_module_name: str):
        layer = self.get_layers(model)[layer_idx]
        if logical_module_name in {"mlp.gate_proj", "mlp.c_fc"}:
            return layer.mlp.c_fc
        if logical_module_name == "mlp.c_proj":
            return layer.mlp.c_proj
        if logical_module_name in {
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "attn.c_attn",
        }:
            return layer.attn.c_attn
        if logical_module_name == "attn.c_proj":
            return layer.attn.c_proj
        raise ValueError(f"Unsupported logical module {logical_module_name}")

    def default_score_modules(self) -> list[str]:
        return list(CULNIG_SCORE_MODULES)

    def _attn_split_size(self, model, layer_idx: int) -> int:
        module = self.get_layers(model)[layer_idx].attn.c_attn
        out_width = int(module.nf if hasattr(module, "nf") else module.weight.shape[-1])
        if out_width % 3 != 0:
            raise ValueError(
                f"GPT-2 c_attn output width is not divisible by 3: {out_width}"
            )
        return out_width // 3

    def logical_module_size(
        self, model, layer_idx: int, logical_module_name: str
    ) -> int:
        if logical_module_name in CULNIG_ATTENTION_MODULES:
            return self._attn_split_size(model, layer_idx)
        return super().logical_module_size(model, layer_idx, logical_module_name)

    def extract_logical_tensor(
        self, model, layer_idx: int, logical_module_name: str, tensor
    ):
        if logical_module_name not in CULNIG_ATTENTION_MODULES:
            return tensor
        split = self._attn_split_size(model, layer_idx)
        offsets = {
            "self_attn.q_proj": 0,
            "self_attn.k_proj": split,
            "self_attn.v_proj": 2 * split,
        }
        start = offsets[logical_module_name]
        return tensor[..., start : start + split]

    def translate_output_indices(
        self,
        model,
        layer_idx: int,
        logical_module_name: str,
        neuron_indices: Iterable[int],
    ) -> list[int]:
        if logical_module_name not in CULNIG_ATTENTION_MODULES:
            return list(neuron_indices)
        split = self._attn_split_size(model, layer_idx)
        offsets = {
            "self_attn.q_proj": 0,
            "self_attn.k_proj": split,
            "self_attn.v_proj": 2 * split,
        }
        offset = offsets[logical_module_name]
        return [offset + int(i) for i in neuron_indices]

    def prepend_token_id(self, tokenizer) -> int | None:
        return (
            getattr(tokenizer, "cls_token_id", None)
            or getattr(tokenizer, "bos_token_id", None)
            or getattr(tokenizer, "eos_token_id", None)
        )


class XGLMLikeAdapter(ArchitectureAdapter):
    def __init__(self) -> None:
        super().__init__(family="xglm_like")

    def get_backbone(self, model):
        if hasattr(model, "model"):
            return model.model
        raise ValueError("Unsupported xglm-like model structure")

    def get_layers(self, model):
        return self.get_backbone(model).layers

    def resolve_module(self, model, layer_idx: int, logical_module_name: str):
        layer = self.get_layers(model)[layer_idx]
        if logical_module_name in {"mlp.gate_proj", "fc1"}:
            return layer.fc1
        if logical_module_name == "fc2":
            return layer.fc2
        if logical_module_name == "self_attn.q_proj":
            return layer.self_attn.q_proj
        if logical_module_name == "self_attn.k_proj":
            return layer.self_attn.k_proj
        if logical_module_name == "self_attn.v_proj":
            return layer.self_attn.v_proj
        raise ValueError(f"Unsupported logical module {logical_module_name}")

    def default_score_modules(self) -> list[str]:
        return list(CULNIG_SCORE_MODULES)

    def prepend_token_id(self, tokenizer) -> int | None:
        return getattr(tokenizer, "eos_token_id", None) or getattr(
            tokenizer, "bos_token_id", None
        )


class LlamaAdapter(ArchitectureAdapter):
    """Llama-style architecture. Legacy ``mlp.down_proj`` captures post-SwiGLU hidden states (pre-hook)."""

    def __init__(self) -> None:
        super().__init__(family="llama_like")

    def get_backbone(self, model):
        if hasattr(model, "model"):
            return model.model
        raise ValueError("Unsupported llama-like model structure")

    def get_layers(self, model):
        return self.get_backbone(model).layers

    def resolve_module(self, model, layer_idx: int, logical_module_name: str):
        layer = self.get_layers(model)[layer_idx]
        if logical_module_name == "mlp.down_proj":
            return layer.mlp.down_proj
        if logical_module_name == "mlp.gate_proj":
            return layer.mlp.gate_proj
        if logical_module_name == "mlp.up_proj":
            return layer.mlp.up_proj
        if logical_module_name == "self_attn.q_proj":
            return layer.self_attn.q_proj
        if logical_module_name == "self_attn.k_proj":
            return layer.self_attn.k_proj
        if logical_module_name == "self_attn.v_proj":
            return layer.self_attn.v_proj
        raise ValueError(f"Unsupported logical module {logical_module_name}")

    def default_score_modules(self) -> list[str]:
        return list(CULNIG_SCORE_MODULES)

    def module_hook_kind(self, logical_module_name: str) -> str:
        if logical_module_name == "mlp.down_proj":
            return "pre"
        return "post"

    def prepend_token_id(self, tokenizer) -> int | None:
        for attr in ("bos_token_id", "eos_token_id"):
            value = getattr(tokenizer, attr, None)
            if value is not None:
                return value
        return None


class GPTNeoXAdapter(ArchitectureAdapter):
    """GPT-NeoX: h→4h→h GeLU MLP; logical ``mlp.gate_proj`` maps to ``dense_h_to_4h`` output.

    Attention is a fused ``query_key_value`` linear, so per-projection q/k/v
    slicing is unsupported — fine since the primary score targets MLP only.
    """

    def __init__(self) -> None:
        super().__init__(family="gpt_neox_like")

    def get_backbone(self, model):
        if hasattr(model, "gpt_neox"):
            return model.gpt_neox
        raise ValueError("Unsupported GPT-NeoX model structure")

    def get_layers(self, model):
        return self.get_backbone(model).layers

    def resolve_module(self, model, layer_idx: int, logical_module_name: str):
        layer = self.get_layers(model)[layer_idx]
        if logical_module_name == "mlp.gate_proj":
            return layer.mlp.dense_h_to_4h
        if logical_module_name == "mlp.down_proj":
            return layer.mlp.dense_4h_to_h
        if logical_module_name in CULNIG_ATTENTION_MODULES:
            raise ValueError(
                f"GPT-NeoX uses a fused query_key_value linear; "
                f"per-projection attention scoring ({logical_module_name!r}) "
                f"is not supported by this adapter"
            )
        raise ValueError(f"Unsupported logical module {logical_module_name}")

    def default_score_modules(self) -> list[str]:
        return list(CULNIG_SCORE_MODULES)

    def module_hook_kind(self, logical_module_name: str) -> str:
        if logical_module_name == "mlp.down_proj":
            return "pre"
        return "post"

    def prepend_token_id(self, tokenizer) -> int | None:
        return getattr(tokenizer, "bos_token_id", None) or getattr(
            tokenizer, "eos_token_id", None
        )


def get_adapter(model_name: str, family: str | None = None) -> ArchitectureAdapter:
    """Return the architecture adapter; ``family`` (when given) bypasses manifest lookup for off-manifest models."""
    if family is None:
        from diss.core.registry import infer_architecture_family

        family = infer_architecture_family(model_name)
    if family == "gpt2_like":
        return GPT2LikeAdapter()
    if family == "xglm_like":
        return XGLMLikeAdapter()
    if family == "llama_like":
        return LlamaAdapter()
    if family == "gpt_neox_like":
        return GPTNeoXAdapter()
    raise ValueError(f"Unsupported architecture family: {family}")
