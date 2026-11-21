"""Speaker-stream text and waveform metrics for multilingual two-voice recordings."""

from collections import defaultdict

import numpy as np
from rapidfuzz.distance import Levenshtein
from scipy.optimize import linear_sum_assignment

from cleartalk_asr.mixtures import si_sdr
from cleartalk_ml.text import normalize


def stream_errors(references, turns):
    """Concatenate chronologically per speaker, pad missing streams, then minimize word edits."""
    by_speaker = defaultdict(list)
    for turn in sorted(turns, key=lambda t: t["start"]):
        by_speaker[turn["speaker"]].append(turn["text"])
    refs = [normalize(text).split() for text in references]
    hyps = [normalize(" ".join(parts)).split() for parts in by_speaker.values()]
    n = max(len(refs), len(hyps))
    refs.extend([[] for _ in range(n - len(refs))])
    hyps.extend([[] for _ in range(n - len(hyps))])
    distances = np.array(
        [[Levenshtein.distance(r, h) for h in hyps] for r in refs], dtype=int
    )
    assignment = linear_sum_assignment(distances)
    return int(distances[assignment].sum()), sum(map(len, refs))


def signal_scores(mixture, references, estimates):
    """Return mixture and best matched source-mean SI-SDR in dB, both on the same sample grid."""
    mix = np.asarray(mixture, dtype=np.float64)
    ref, est = (np.asarray(s, dtype=np.float64) for s in (references, estimates))
    if mix.ndim != 1 or ref.shape != est.shape or ref.shape != (2, len(mix)):
        raise ValueError(
            "Expected a mono mixture and two aligned reference and estimate tracks"
        )
    if (
        not np.isfinite(ref).all()
        or not np.isfinite(est).all()
        or not np.isfinite(mix).all()
    ):
        raise ValueError("Non-finite waveform")
    if (ref.var(axis=1) < 1e-12).any():
        raise ValueError("SI-SDR needs two non-silent reference stems")
    before = float(np.mean([si_sdr(mix, target) for target in ref]))
    matrix = np.array([[si_sdr(output, target) for output in est] for target in ref])
    after = float(matrix[linear_sum_assignment(-matrix)].mean())
    return before, after
