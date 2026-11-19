def test_seed_streams_repeat() -> None:
    from sciprism_x.reproducibility import seed_everything

    import numpy as np
    import torch
    import random

    seed_everything(42, deterministic=True)
    first = (random.random(), np.random.random(), torch.rand(1).item())

    seed_everything(42, deterministic=True)
    second = (random.random(), np.random.random(), torch.rand(1).item())

    assert first == second


def test_derived_seed_is_namespace_stable() -> None:
    from sciprism_x.reproducibility import derive_seed

    assert derive_seed(42, "sampling") == derive_seed(42, "sampling")
    assert derive_seed(42, "sampling") != derive_seed(42, "bootstrap")
