from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from attentions.config import BenchmarkCase, OperatorConfig


AttentionCallable = Callable[..., object]


@dataclass(frozen=True)
class RuntimeOperator:
    config: OperatorConfig
    fn: AttentionCallable | None
    is_available: bool
    availability_note: str = ""

    def supports_case(self, case: BenchmarkCase) -> tuple[bool, str]:
        if case.mode not in self.config.mode:
            return False, f"mode {case.mode} not in supported modes {self.config.mode}"
        if case.causal and not self.config.supports_causal:
            return False, "causal mask not supported"
        if case.h_q != case.h_kv and not self.config.supports_gqa:
            return False, "GQA layout not supported"
        if case.mask == "sliding_window" and not self.config.supports_sliding_window:
            return False, "sliding window mask not supported"
        if case.page_size > 0 and not self.config.supports_paged_kv:
            return False, "paged KV cache not supported"
        if case.mode in {"bwd", "fwd_bwd"} and not self.config.backward_support:
            return False, "backward path not supported"
        if not self.is_available:
            return False, self.availability_note or "operator implementation not available"
        return True, ""


class OperatorRegistry:
    def __init__(self) -> None:
        self._operators: dict[str, RuntimeOperator] = {}

    def register(self, operator: RuntimeOperator) -> None:
        self._operators[operator.config.operator_id] = operator

    def get(self, operator_id: str) -> RuntimeOperator:
        return self._operators[operator_id]

    def all(self) -> list[RuntimeOperator]:
        return list(self._operators.values())
