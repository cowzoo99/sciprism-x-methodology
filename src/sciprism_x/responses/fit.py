from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.optimize import differential_evolution

from sciprism_x.reproducibility import derive_seed

from .schema import ReconstructionProfile

PROFILE_NAMES = tuple(ReconstructionProfile.model_fields)
PROFILE_BOUNDS: dict[str, tuple[float, float]] = {
    **{name: (0.0, 1.0) for name in PROFILE_NAMES if name != "length_ratio"},
    "length_ratio": (0.25, 2.0),
}


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ProfileGroup(StrictModel):
    model_id: str = Field(min_length=1)
    regime: str = Field(min_length=1)
    domain: str = Field(min_length=1)
    competency: str = Field(min_length=1)

    def namespace(self) -> str:
        return ":".join((self.model_id, self.regime, self.domain, self.competency))


class ProfileFitRequest(StrictModel):
    group: ProfileGroup
    targets: dict[str, float]
    weights: dict[str, float]

    @model_validator(mode="after")
    def validate_targets(self) -> "ProfileFitRequest":
        if not self.targets:
            raise ValueError("at least one fitting target is required")
        if set(self.targets) != set(self.weights):
            raise ValueError("target and weight names must match")
        if any(weight <= 0.0 for weight in self.weights.values()):
            raise ValueError("all fitting weights must be positive")
        return self


class OptimizerControls(StrictModel):
    strategy: str = "best1bin"
    maxiter: int = 500
    popsize: int = 15
    tol: float = 1.0e-9
    mutation: tuple[float, float] = (0.5, 1.0)
    recombination: float = 0.7
    polish: bool = False
    init: str = "latinhypercube"
    updating: str = "immediate"
    workers: int = 1
    atol: float = 0.0


class ProfileFitResult(StrictModel):
    group: ProfileGroup
    profile: ReconstructionProfile
    target_vector: dict[str, float]
    weights: dict[str, float]
    bounds: dict[str, tuple[float, float]]
    convergence_trace: tuple[float, ...]
    loss: float
    iterations: int
    evaluations: int
    seed: int
    entropy_weight: float = Field(ge=0.0)
    optimizer: OptimizerControls


Predictor = Callable[[ReconstructionProfile], Mapping[str, float]]


def profile_observables(profile: ReconstructionProfile) -> dict[str, float]:
    """Identity observation model used for direct profile-rate targets."""

    return {name: float(value) for name, value in profile.model_dump().items()}


def profile_from_vector(values: Sequence[float]) -> ReconstructionProfile:
    return ReconstructionProfile(**dict(zip(PROFILE_NAMES, values, strict=True)))


def maximum_entropy_objective(
    vector: np.ndarray,
    target_names: tuple[str, ...],
    targets: np.ndarray,
    weights: np.ndarray,
    entropy_weight: float,
) -> float:
    if entropy_weight < 0.0:
        raise ValueError("entropy_weight must be nonnegative")
    profile = profile_from_vector(vector)
    observed = np.asarray(
        [profile_observables(profile)[name] for name in target_names],
        dtype=np.float64,
    )
    discrepancy = float(np.sum(weights * np.square(observed - targets)))
    probabilities = np.clip(vector[:9], 1.0e-12, 1.0 - 1.0e-12)
    negative_entropy = float(
        np.sum(
            probabilities * np.log(probabilities)
            + (1.0 - probabilities) * np.log(1.0 - probabilities)
        )
    )
    return discrepancy + entropy_weight * negative_entropy


def fit_profiles(
    requests: Sequence[ProfileFitRequest],
    *,
    seed: int,
    predictor: Predictor = profile_observables,
    entropy_weight: float = 0.0,
) -> tuple[ProfileFitResult, ...]:
    if entropy_weight < 0.0:
        raise ValueError("entropy_weight must be nonnegative")
    controls = OptimizerControls()
    results: list[ProfileFitResult] = []
    ordered_bounds = tuple(PROFILE_BOUNDS[name] for name in PROFILE_NAMES)

    for request in requests:
        target_names = tuple(sorted(request.targets))
        fit_seed = derive_seed(seed, f"profile-fit:{request.group.namespace()}")
        trace: list[float] = []

        def objective(values: np.ndarray) -> float:
            predicted = predictor(profile_from_vector(values))
            missing = set(target_names).difference(predicted)
            if missing:
                raise ValueError(f"predictor omitted targets: {sorted(missing)}")
            weighted_error = float(sum(
                request.weights[name]
                * (float(predicted[name]) - request.targets[name]) ** 2
                for name in target_names
            ))
            probabilities = np.clip(values[:9], 1.0e-12, 1.0 - 1.0e-12)
            negative_entropy = float(
                np.sum(
                    probabilities * np.log(probabilities)
                    + (1.0 - probabilities) * np.log(1.0 - probabilities)
                )
            )
            return weighted_error + entropy_weight * negative_entropy

        def callback(values: np.ndarray, convergence: float) -> bool:
            del convergence
            trace.append(objective(values))
            return False

        optimized = differential_evolution(
            objective,
            ordered_bounds,
            strategy=controls.strategy,
            maxiter=controls.maxiter,
            popsize=controls.popsize,
            tol=controls.tol,
            mutation=controls.mutation,
            recombination=controls.recombination,
            polish=controls.polish,
            init=controls.init,
            updating=controls.updating,
            workers=controls.workers,
            atol=controls.atol,
            rng=np.random.default_rng(fit_seed),
            callback=callback,
        )
        profile = profile_from_vector(optimized.x)
        results.append(
            ProfileFitResult(
                group=request.group,
                profile=profile,
                target_vector=dict(request.targets),
                weights=dict(request.weights),
                bounds={name: PROFILE_BOUNDS[name] for name in PROFILE_NAMES},
                convergence_trace=tuple(trace),
                loss=float(optimized.fun),
                iterations=int(optimized.nit),
                evaluations=int(optimized.nfev),
                seed=fit_seed,
                entropy_weight=entropy_weight,
                optimizer=controls,
            )
        )
    return tuple(results)
