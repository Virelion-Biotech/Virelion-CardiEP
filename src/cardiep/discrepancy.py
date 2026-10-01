from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from .ecg import ECGResult
from .models import EPObservation
from .provenance import uri_to_path


@dataclass(frozen=True)
class DiscrepancyTerm:
    term_id: str
    observation_id: str
    model_output: str
    metric: str
    value: float
    weight: float

    @property
    def weighted_value(self) -> float:
        return self.value * self.weight

    def to_dict(self) -> dict[str, Any]:
        return {
            "term_id": self.term_id,
            "observation_id": self.observation_id,
            "model_output": self.model_output,
            "metric": self.metric,
            "value": self.value,
            "weight": self.weight,
            "weighted_value": self.weighted_value,
        }


@dataclass(frozen=True)
class DiscrepancyReport:
    objective: float
    terms: tuple[DiscrepancyTerm, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "terms": [item.to_dict() for item in self.terms],
        }


def _load_json(observation: EPObservation) -> dict[str, Any]:
    path = uri_to_path(observation.artifact.uri)
    if path.suffix.lower() != ".json":
        raise ValueError(
            f"Native discrepancy evaluation currently expects JSON observation artifacts: {path}"
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Observation artifact JSON must contain an object")
    return raw


def _resample(values: np.ndarray, n: int) -> np.ndarray:
    x = np.asarray(values, dtype=float).reshape(-1)
    if len(x) == n:
        return x
    if len(x) < 2 or n < 2:
        raise ValueError("Signals require at least two samples")
    source = np.linspace(0.0, 1.0, len(x))
    target = np.linspace(0.0, 1.0, n)
    return np.interp(target, source, x)


def _standardize(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    x = x - float(np.mean(x))
    scale = float(np.sqrt(np.mean(x**2)))
    return x / max(scale, 1e-12)


def correlation_distance(observed: np.ndarray, simulated: np.ndarray) -> float:
    n = max(len(observed), len(simulated))
    a = _standardize(_resample(observed, n))
    b = _standardize(_resample(simulated, n))
    corr = float(np.clip(np.mean(a * b), -1.0, 1.0))
    return 1.0 - corr


def normalized_rmse(observed: np.ndarray, simulated: np.ndarray) -> float:
    n = max(len(observed), len(simulated))
    a = _resample(observed, n)
    b = _resample(simulated, n)
    scale = max(float(np.ptp(a)), float(np.sqrt(np.mean((a - np.mean(a)) ** 2))), 1e-12)
    return float(np.sqrt(np.mean((a - b) ** 2)) / scale)


def huber_rmse(observed: np.ndarray, simulated: np.ndarray, delta: float = 1.5) -> float:
    n = max(len(observed), len(simulated))
    a = _standardize(_resample(observed, n))
    b = _standardize(_resample(simulated, n))
    residual = np.abs(a - b)
    loss = np.where(residual <= delta, 0.5 * residual**2, delta * (residual - 0.5 * delta))
    return float(np.sqrt(2.0 * np.mean(loss)))


def _ecg_leads(raw: dict[str, Any]) -> tuple[list[str], dict[str, np.ndarray]]:
    if "beat_template" in raw:
        mapping = raw["beat_template"]
        names = list(raw.get("lead_names") or mapping)
        return names, {str(name): np.asarray(mapping[name], dtype=float) for name in names if name in mapping}
    names = [str(x) for x in raw["lead_names"]]
    values = np.asarray(raw["values"], dtype=float)
    if values.ndim != 2 or values.shape[0] != len(names):
        raise ValueError("ECG artifact lead_names/values shape mismatch")
    return names, {name: values[i] for i, name in enumerate(names)}


def ecg_discrepancy(raw: dict[str, Any], simulated: ECGResult, metric: str) -> float:
    _, observed = _ecg_leads(raw)
    simulated_by_lead = {
        name: simulated.values[i] for i, name in enumerate(simulated.lead_names)
    }
    common = sorted(set(observed) & set(simulated_by_lead))
    if not common:
        raise ValueError("Observed and simulated ECGs have no common lead names")
    metrics = {
        "correlation": correlation_distance,
        "rmse": normalized_rmse,
        "nrmse": normalized_rmse,
        "huber": huber_rmse,
        "gaussian": normalized_rmse,
        "student_t": huber_rmse,
    }
    if metric not in metrics:
        raise ValueError(f"Unsupported ECG discrepancy metric: {metric}")
    return float(np.mean([metrics[metric](observed[name], simulated_by_lead[name]) for name in common]))


def _field_values(raw: dict[str, Any], *keys: str) -> np.ndarray:
    for key in keys:
        if key in raw:
            return np.asarray(raw[key], dtype=float).reshape(-1)
    raise ValueError(f"Observation artifact is missing one of fields: {', '.join(keys)}")


def field_discrepancy(observed: np.ndarray, simulated: np.ndarray, metric: str) -> float:
    a = np.asarray(observed, dtype=float).reshape(-1)
    b = np.asarray(simulated, dtype=float).reshape(-1)
    if a.shape != b.shape:
        raise ValueError("Observed and simulated field maps must have the same number of nodes")
    if metric in {"rmse", "gaussian"}:
        return float(np.sqrt(np.mean((a - b) ** 2)))
    if metric in {"student_t", "huber"}:
        scale = max(float(np.std(a)), 1.0)
        residual = np.abs((a - b) / scale)
        delta = 1.5
        loss = np.where(residual <= delta, 0.5 * residual**2, delta * (residual - 0.5 * delta))
        return float(np.sqrt(2 * np.mean(loss)))
    if metric == "correlation":
        return correlation_distance(a, b)
    raise ValueError(f"Unsupported field discrepancy metric: {metric}")


def evaluate_observations(
    observations: list[EPObservation],
    *,
    activation_ms: np.ndarray,
    repolarization_ms: np.ndarray,
    ecg: ECGResult | None,
    hints: list[dict[str, Any]] | None = None,
) -> DiscrepancyReport:
    observation_by_id = {item.observation_id: item for item in observations}
    term_specs: list[dict[str, Any]] = []
    if hints:
        term_specs = [dict(item) for item in hints]
    else:
        for obs in observations:
            if obs.kind == "ecg":
                term_specs.append({
                    "term_id": f"{obs.observation_id}:ecg",
                    "observation_id": obs.observation_id,
                    "model_output": "ecg",
                    "discrepancy": "correlation",
                    "weight": 1.0,
                })
            elif obs.kind in {"eam_activation", "activation_map"}:
                term_specs.append({
                    "term_id": f"{obs.observation_id}:activation",
                    "observation_id": obs.observation_id,
                    "model_output": "activation_map",
                    "discrepancy": "huber",
                    "weight": 1.0,
                })
            elif obs.kind == "repolarization_map":
                term_specs.append({
                    "term_id": f"{obs.observation_id}:repolarization",
                    "observation_id": obs.observation_id,
                    "model_output": "repolarization_map",
                    "discrepancy": "huber",
                    "weight": 1.0,
                })

    terms: list[DiscrepancyTerm] = []
    for spec in term_specs:
        observation_id = spec.get("observation_id")
        if observation_id is None and len(observations) == 1:
            observation_id = observations[0].observation_id
        observation_id = str(observation_id or "")
        observation = observation_by_id.get(observation_id)
        if observation is None:
            raise ValueError(f"Likelihood term does not identify a known observation: {spec}")
        raw = _load_json(observation)
        output = str(spec.get("model_output") or "ecg")
        metric = str(spec.get("discrepancy") or "gaussian")
        weight = float(spec.get("weight", 1.0))
        metadata = dict(spec.get("metadata") or {})

        if output == "ecg":
            if ecg is None:
                raise ValueError("ECG discrepancy requested but pseudo-ECG generation is unavailable")
            value = ecg_discrepancy(raw, ecg, metric)
        elif output == "qrs_duration_ms":
            observed_value = metadata.get("observed_value_ms")
            if observed_value is None:
                observed_value = dict(raw.get("qrs") or {}).get("median_duration_ms")
            if observed_value is None:
                raise ValueError("QRS-duration term is missing the observed duration")
            predicted = float(np.max(activation_ms) - np.min(activation_ms))
            sigma = max(float(dict(spec.get("noise_parameters") or {}).get("sigma_ms", 1.0)), 1e-6)
            value = abs(predicted - float(observed_value)) / sigma
        elif output == "activation_map":
            observed = _field_values(raw, "activation_ms", "values_ms", "values")
            value = field_discrepancy(observed, activation_ms, metric)
        elif output == "repolarization_map":
            observed = _field_values(raw, "repolarization_ms", "values_ms", "values")
            value = field_discrepancy(observed, repolarization_ms, metric)
        else:
            raise ValueError(f"Unsupported CardiEP model_output in discrepancy term: {output}")
        terms.append(
            DiscrepancyTerm(
                term_id=str(spec.get("term_id") or f"{observation_id}:{output}"),
                observation_id=observation_id,
                model_output=output,
                metric=metric,
                value=float(value),
                weight=weight,
            )
        )
    objective = float(sum(item.weighted_value for item in terms))
    return DiscrepancyReport(objective=objective, terms=tuple(terms))
