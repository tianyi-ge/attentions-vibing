from __future__ import annotations

from attentions.config import OperatorConfig
from attentions.operators.placeholders import unavailable_operator
from attentions.operators.reference import naive_sdpa_reference, torch_sdpa_reference
from attentions.registry import OperatorRegistry, RuntimeOperator


IMPLEMENTED_IDS = {"torch_sdpa_ref", "naive_sdpa"}


def _resolve_fn(operator_id: str):
    if operator_id == "torch_sdpa_ref":
        return torch_sdpa_reference, True, ""
    if operator_id == "naive_sdpa":
        return naive_sdpa_reference, True, "running eager math baseline until Triton kernel is added"
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
