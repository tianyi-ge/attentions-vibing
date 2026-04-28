from __future__ import annotations

import torch


def unavailable_operator(*args: torch.Tensor, **kwargs: object) -> torch.Tensor:
    raise NotImplementedError("This operator is a scaffold placeholder and is not implemented yet.")
