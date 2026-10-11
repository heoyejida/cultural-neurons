"""Length-bucket sort and left-pad utilities for the cultural-neuron pipeline."""

from __future__ import annotations

from typing import Callable, Sequence, TypeVar

import torch

T = TypeVar("T")


def length_bucket_sort(
    items: Sequence[T],
    key_fn: Callable[[T], int],
    batch_size: int,
) -> tuple[list[list[T]], list[int]]:
    """Sort ``items`` by ``key_fn`` ascending, then chunk into batches of ``batch_size``.

    Returns (batches, restore), where ``restore[i]`` is the original index of
    flat position ``i`` — callers use it to put results back in input order.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive (got {batch_size})")

    indexed = sorted(enumerate(items), key=lambda p: key_fn(p[1]))
    sorted_items = [it for _, it in indexed]
    restore = [orig for orig, _ in indexed]

    batches: list[list[T]] = []
    for start in range(0, len(sorted_items), batch_size):
        batches.append(sorted_items[start : start + batch_size])
    return batches, restore


def left_pad_batch(
    seqs: Sequence[Sequence[int]],
    pad_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Left-pad token-id sequences to uniform length.

    Returns (B, S) tensors: padded ids, attention mask, and position ids
    (``mask.cumsum(-1) - 1`` with pad positions clamped to 0).
    """
    if not seqs:
        raise ValueError("seqs must be non-empty")
    max_len = max(len(s) for s in seqs)
    padded_ids: list[list[int]] = []
    padded_mask: list[list[int]] = []
    for s in seqs:
        pad_n = max_len - len(s)
        padded_ids.append([pad_id] * pad_n + list(s))
        padded_mask.append([0] * pad_n + [1] * len(s))

    ids = torch.tensor(padded_ids, dtype=torch.long, device=device)
    mask = torch.tensor(padded_mask, dtype=torch.long, device=device)
    pos = mask.long().cumsum(-1) - 1
    pos.masked_fill_(pos < 0, 0)
    return ids, mask, pos
