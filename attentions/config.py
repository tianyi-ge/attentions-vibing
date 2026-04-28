from __future__ import annotations

from dataclasses import dataclass
import csv
from pathlib import Path
from typing import Iterable


ROOT_DIR = Path(__file__).resolve().parent.parent
BENCHMARKS_DIR = ROOT_DIR / "benchmarks"
OPERATORS_CSV = BENCHMARKS_DIR / "operators.csv"
CASES_CSV = BENCHMARKS_DIR / "cases.csv"
RESULTS_TEMPLATE_CSV = BENCHMARKS_DIR / "results-template.csv"


def _to_bool(raw: str) -> bool:
    return raw.strip() in {"1", "true", "True", "yes", "y"}


def _split_modes(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in raw.split("|") if part.strip())


@dataclass(frozen=True)
class OperatorConfig:
    operator_id: str
    enabled: bool
    priority: str
    implementation_stage: str
    backend: str
    category: str
    mode: tuple[str, ...]
    exactness: str
    supports_causal: bool
    supports_gqa: bool
    supports_sliding_window: bool
    supports_paged_kv: bool
    backward_support: bool
    notes: str


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    enabled: bool
    family: str
    mode: str
    dtype: str
    mask: str
    batch_size: int
    s_q: int
    s_k: int
    h_q: int
    h_kv: int
    head_dim: int
    window_size: int
    page_size: int
    block_size: int
    causal: bool
    notes: str


def load_operator_configs(path: Path = OPERATORS_CSV) -> list[OperatorConfig]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        return [
            OperatorConfig(
                operator_id=row["operator_id"],
                enabled=_to_bool(row["enabled"]),
                priority=row["priority"],
                implementation_stage=row["implementation_stage"],
                backend=row["backend"],
                category=row["category"],
                mode=_split_modes(row["mode"]),
                exactness=row["exactness"],
                supports_causal=_to_bool(row["supports_causal"]),
                supports_gqa=_to_bool(row["supports_gqa"]),
                supports_sliding_window=_to_bool(row["supports_sliding_window"]),
                supports_paged_kv=_to_bool(row["supports_paged_kv"]),
                backward_support=_to_bool(row["backward_support"]),
                notes=row["notes"],
            )
            for row in rows
        ]


def load_benchmark_cases(path: Path = CASES_CSV) -> list[BenchmarkCase]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        return [
            BenchmarkCase(
                case_id=row["case_id"],
                enabled=_to_bool(row["enabled"]),
                family=row["family"],
                mode=row["mode"],
                dtype=row["dtype"],
                mask=row["mask"],
                batch_size=int(row["batch_size"]),
                s_q=int(row["s_q"]),
                s_k=int(row["s_k"]),
                h_q=int(row["h_q"]),
                h_kv=int(row["h_kv"]),
                head_dim=int(row["head_dim"]),
                window_size=int(row["window_size"]),
                page_size=int(row["page_size"]),
                block_size=int(row["block_size"]),
                causal=_to_bool(row["causal"]),
                notes=row["notes"],
            )
            for row in rows
        ]


def filter_enabled(items: Iterable[OperatorConfig | BenchmarkCase]) -> list[OperatorConfig | BenchmarkCase]:
    return [item for item in items if item.enabled]
