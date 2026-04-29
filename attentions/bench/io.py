from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable


RESULT_FIELDNAMES = [
    "run_id",
    "operator_id",
    "case_id",
    "backend",
    "mode",
    "dtype",
    "is_correct",
    "max_abs_err",
    "max_rel_err",
    "runtime_ms",
    "throughput_tflops",
    "estimated_flops",
    "estimated_hbm_bytes",
    "arithmetic_intensity",
    "math_mfu",
    "tensorcore_mfu",
    "max_memory_mb",
    "occupancy_estimate",
]


def write_results_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row_list = list(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDNAMES)
        writer.writeheader()
        for row in row_list:
            writer.writerow(row)
