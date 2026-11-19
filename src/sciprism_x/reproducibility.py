from __future__ import annotations

import hashlib
import os
import random


def derive_seed(base_seed: int, namespace: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{namespace}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


def seed_everything(seed: int, deterministic: bool) -> None:
    import numpy as np
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
