from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from .ecg import ECGResult
from .models import EPObservation
from .provenance import uri_to_path, verify_file_sha256


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
    verify_file_sha256(path, observation.artifact.sha256)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("Observation artifact JSON must contain an object")
    return raw


def _signal(values: np.ndarray, name: str) -> np.ndarray:
    x = np.asarray(values, dtype=float).reshape(-1)
    if len(x) < 2:
        raise ValueError(f"{name} requires at least two samples")
    if not np.isfinite(x).all():
        raise ValueError(f"{name} must contain only finite samples")
    return x


def _resample(values: np.ndarray, n: int) -> np.ndarray:
    x = _signal(values, "Signal")
    if len(x) == n:
        return x
    if n < 2:
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
    observed_resampled = _resample(observed, n)
    simulated_resampled = _resample(simulated, n)
    if np.ptp(observed_resampled) <= 1e-12:
        raise ValueError("Correlation discrepancy is undefined for a constant observed signal")
    if np.ptp(simulated_resampled) <= 1e-12:
        raise ValueError("Correlation discrepancy is undefined for a constant simulated signal")
    a = _standardize(observed_resampled)
    b = _standardize(simulated_resampled)
    corr = float(np.clip(np.mean(a * b), -1.0, 1.0))
    return 1.0 - corr


def normalized_rmse(observed: np.ndarray, simulated: np.ndarray) -> float:
    n = max(len(observed), len(simulated))
    a = _resample(observed, n)
    b = _resample(simulated, n)
    scale = max(float(np.ptp(a)), float(np.sqrt(np.mean((a - np.mean(a)) ** 2))), 1e-12)
    return float(np.sqrt(np.mean((a - b) ** 2)) / scale)


def normalized_mae(observed: np.ndarray, simulated: np.ndarray) -> float:
    n = max(len(observed), len(simulated))
    a = _resample(observed, n)
    b = _resample(simulated, n)
    scale = max(
        float(np.ptp(a)),
        float(np.sqrt(np.mean((a - np.mean(a)) ** 2))),
        1e-12,
    )
    return float(np.mean(np.abs(a - b)) / scale)


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
        if not isinstance(mapping, dict) or not mapping:
            raise TypeError("ECG beat_template must be a non-empty mapping")
        names = [str(name) for name in (raw.get("lead_names") or mapping)]
        missing = [name for name in names if name not in mapping]
        if missing:
            raise ValueError(f"ECG beat_template is missing declared leads: {missing}")
        leads = {name: _signal(mapping[name], f"ECG lead {name}") for name in names}
    else:
        if "lead_names" not in raw or "values" not in raw:
            raise ValueError("ECG artifact requires lead_names and values")
        names = [str(x) for x in raw["lead_names"]]
        values = np.asarray(raw["values"], dtype=float)
        if values.ndim != 2 or values.shape[0] != len(names):
            raise ValueError("ECG artifact lead_names/values shape mismatch")
        leads = {name: _signal(values[i], f"ECG lead {name}") for i, name in enumerate(names)}
    if len(names) != len(set(names)):
        raise ValueError("ECG lead_names must be unique")
    return names, leads


def _observed_ecg_time_ms(raw: dict[str, Any], n_samples: int) -> tuple[np.ndarray | None, bool]:
    if "relative_time_s" in raw:
        time = np.asarray(raw["relative_time_s"], dtype=float).reshape(-1) * 1000.0
        is_relative = True
    elif "time_ms" in raw:
        time = np.asarray(raw["time_ms"], dtype=float).reshape(-1)
        is_relative = False
    else:
        return None, False
    if len(time) != n_samples:
        raise ValueError("ECG time axis length does not match waveform samples")
    if not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
        raise ValueError("ECG time axis must be finite and strictly increasing")
    return time, is_relative


def _align_simulated_ecg(
    raw: dict[str, Any],
    observed_signal: np.ndarray,
    simulated: ECGResult,
    simulated_signal: np.ndarray,
) -> np.ndarray:
    observed_time, is_relative = _observed_ecg_time_ms(raw, len(observed_signal))
    if observed_time is None:
        return _resample(simulated_signal, len(observed_signal))

    simulated_time = np.asarray(simulated.time_ms, dtype=float).reshape(-1)
    if len(simulated_time) != len(simulated_signal):
        raise ValueError("Simulated ECG time axis length does not match waveform samples")
    if not np.isfinite(simulated_time).all() or np.any(np.diff(simulated_time) <= 0):
        raise ValueError("Simulated ECG time axis must be finite and strictly increasing")

    if is_relative:
        if simulated.reference_time_ms is None or not np.isfinite(simulated.reference_time_ms):
            raise ValueError(
                "R-relative ECG comparison requires a finite simulated reference_time_ms"
            )
        simulated_time = simulated_time - float(simulated.reference_time_ms)

    return np.interp(
        observed_time,
        simulated_time,
        simulated_signal,
        left=0.0,
        right=0.0,
    )


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
        "mae": normalized_mae,
        "huber": huber_rmse,
        "gaussian": normalized_rmse,
        "student_t": huber_rmse,
    }
    if metric not in metrics:
        raise ValueError(f"Unsupported ECG discrepancy metric: {metric}")

    values = []
    for name in common:
        aligned = _align_simulated_ecg(
            raw,
            observed[name],
            simulated,
            simulated_by_lead[name],
        )
        values.append(metrics[metric](observed[name], aligned))
    return float(np.mean(values))


def _field_values(raw: dict[str, Any], *keys: str) -> np.ndarray:
    for key in keys:
        if key in raw:
            return np.asarray(raw[key], dtype=float).reshape(-1)
    raise ValueError(f"Observation artifact is missing one of fields: {', '.join(keys)}")


def field_discrepancy(observed: np.ndarray, simulated: np.ndarray, metric: str) -> float:
    a = np.asarray(observed, dtype=float).reshape(-1)
    b = np.asarray(simulated, dtype=float).reshape(-1)
    if a.size == 0 or b.size == 0:
        raise ValueError("Observed and simulated field maps must be non-empty")
    if a.shape != b.shape:
        raise ValueError("Observed and simulated field maps must have the same number of nodes")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Observed and simulated field maps must contain finite values")
    if metric in {"rmse", "gaussian"}:
        return float(np.sqrt(np.mean((a - b) ** 2)))
    if metric == "mae":
        return float(np.mean(np.abs(a - b)))
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
    if len(observation_by_id) != len(observations):
        raise ValueError("EP observation IDs must be unique")
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

    if not term_specs:
        raise ValueError("No supported discrepancy terms were defined for the observations")
    term_ids = [
        str(spec.get("term_id") or "")
        for spec in term_specs
        if spec.get("term_id") is not None
    ]
    if len(term_ids) != len(set(term_ids)):
        raise ValueError("Likelihood term IDs must be unique")

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
        if not np.isfinite(weight) or weight <= 0:
            raise ValueError("Discrepancy weights must be positive and finite")
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
            sigma = float(dict(spec.get("noise_parameters") or {}).get("sigma_ms", 1.0))
            observed_value = float(observed_value)
            if not np.isfinite(sigma) or sigma <= 0:
                raise ValueError("QRS sigma_ms must be positive and finite")
            if not np.isfinite(observed_value):
                raise ValueError("Observed QRS duration must be finite")
            value = abs(predicted - observed_value) / sigma
        elif output == "activation_map":
            observed = _field_values(raw, "activation_ms", "values_ms", "values")
            value = field_discrepancy(observed, activation_ms, metric)
        elif output == "repolarization_map":
            observed = _field_values(raw, "repolarization_ms", "values_ms", "values")
            value = field_discrepancy(observed, repolarization_ms, metric)
        else:
            raise ValueError(f"Unsupported CardiEP model_output in discrepancy term: {output}")
        if not np.isfinite(value):
            raise ValueError(f"Discrepancy term produced a non-finite value: {spec}")
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
