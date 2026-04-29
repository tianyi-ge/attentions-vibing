from __future__ import annotations

import torch

from attentions.config import OperatorConfig
from attentions.operators.placeholders import unavailable_operator
from attentions.operators.reference import naive_sdpa_reference, torch_sdpa_reference
from attentions.registry import OperatorRegistry, RuntimeOperator


IMPLEMENTED_IDS = {"torch_sdpa_ref", "naive_sdpa", "online_softmax_fwd_v1", "online_softmax_fwd_v2"}


def _resolve_fn(operator_id: str):
    if operator_id == "torch_sdpa_ref":
        return torch_sdpa_reference, True, ""
    if operator_id == "naive_sdpa":
        return naive_sdpa_reference, True, "running eager math baseline until Triton kernel is added"
    if operator_id == "online_softmax_fwd_v1":
        try:
            from attentions.operators.online_softmax_fwd_v1 import online_softmax_attention_forward_v1
        except Exception as exc:
            return unavailable_operator, False, f"failed to import Triton kernel: {exc}"
        if not torch.cuda.is_available():
            return unavailable_operator, False, "online_softmax_fwd_v1 requires a CUDA runtime"
        return online_softmax_attention_forward_v1, True, "archived v1 Triton online-softmax attention forward kernel"
    if operator_id == "online_softmax_fwd_v2":
        try:
            from attentions.operators.online_softmax_fwd_v2 import online_softmax_attention_forward_v2
        except Exception as exc:
            return unavailable_operator, False, f"failed to import Triton kernel: {exc}"
        if not torch.cuda.is_available():
            return unavailable_operator, False, "online_softmax_fwd_v2 requires a CUDA runtime"
        return online_softmax_attention_forward_v2, True, "forked v2 Triton online-softmax attention forward kernel"
    return unavailable_operator, False, "placeholder only; implementation will be added in a later stage"


def build_registry(configs: list[OperatorConfig]) -> OperatorRegistry:
    registry = OperatorRegistry()
    for config in configs:
        fn, is_available, note = _resolve_fn(config.operator_id)
        registry.register(
            RuntimeOperator(
                config=config,
                fn=fn,
                is_available=is_available,
                availability_note=note,
            )
        )
    return registry
