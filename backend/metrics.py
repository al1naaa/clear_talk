"""Reference-based audio metrics. Only meaningful when clean sources are known."""

from itertools import permutations

import numpy as np


def si_sdr(estimate, reference) -> float:
    """Scale-invariant SDR in dB; both signals must represent the same timeline."""
    estimate = np.asarray(estimate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if estimate.ndim != 1 or reference.ndim != 1 or estimate.shape != reference.shape:
        raise ValueError("SI-SDR requires equal-length mono signals.")
    estimate = estimate - estimate.mean()
    reference = reference - reference.mean()
    energy = float(np.dot(reference, reference))
    if energy < 1e-10:
        raise ValueError("SI-SDR cannot be measured against a silent reference.")
    target = (float(np.dot(estimate, reference)) / energy) * reference
    residual = estimate - target
    return float(10 * np.log10((np.dot(target, target) + 1e-10) / (np.dot(residual, residual) + 1e-10)))


def matched_si_sdr_improvement(estimates, references, mixture) -> dict:
    """Resolve arbitrary source order, then compare with the same noisy input."""
    if len(estimates) != 2 or len(references) != 2:
        raise ValueError("This evaluation requires exactly two sources.")
    length = min(*(len(signal) for signal in estimates), *(len(signal) for signal in references), len(mixture))
    if length < 10:
        raise ValueError("The signals are too short for evaluation.")
    estimates = [signal[:length] for signal in estimates]
    references = [signal[:length] for signal in references]
    mixture = mixture[:length]
    best_order = max(permutations(range(2)), key=lambda order: sum(si_sdr(estimates[i], references[order[i]]) for i in range(2)))
    separated = [si_sdr(estimates[i], references[best_order[i]]) for i in range(2)]
    baseline = [si_sdr(mixture, references[best_order[i]]) for i in range(2)]
    return {
        "mean_improvement_db": float(np.mean(np.subtract(separated, baseline))),
        "separated_mean_db": float(np.mean(separated)),
        "mixture_mean_db": float(np.mean(baseline)),
        "reference_order": list(best_order),
    }
