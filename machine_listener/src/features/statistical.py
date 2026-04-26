"""
Global statistical feature extractor — one scalar per clip.

Original 5 (Phase 3):
  rms       — overall signal energy
  zcr       — how often the waveform crosses zero (noise vs tonal content)
  centroid  — frequency "center of mass" (brightness) — mostly redundant with mel CNN
  rolloff   — freq below which 85% of energy sits (low vs high frequency content)
  bandwidth — spread of spectrum around centroid (tonal vs broadband)

Added in Phase 4b:
  kurtosis  — tailedness of amplitude distribution. Faulty machines produce
              impulsive bursts → high kurtosis. Normal operation is smoother → low kurtosis.
              This is the standard vibration-analysis fault indicator (ISO 13373).

The full v2 ordering (used for .npy caching) is:
  [rms(0), zcr(1), centroid(2), rolloff(3), bandwidth(4), kurtosis(5)]

All values are RAW — you must StandardScaler-normalize before feeding to the model.
Scale differences are huge (RMS ≈ 0.1, spectral centroid ≈ 3000).
"""

import librosa
import numpy as np
from scipy.stats import kurtosis as scipy_kurtosis


# Original 5 — kept for backward compatibility with Phase 3
ALL_FEATURE_NAMES = ["rms", "zcr", "centroid", "rolloff", "bandwidth"]

# v2 set: centroid dropped (redundant with mel CNN), kurtosis added (fault indicator)
# Used by Phase 2b and Phase 4b onwards
ALL_FEATURE_NAMES_V2 = ["rms", "zcr", "rolloff", "bandwidth", "kurtosis"]

# Column index lookup for v2 — use this instead of hardcoding integers
STAT_COL_V2 = {name: i for i, name in enumerate(ALL_FEATURE_NAMES_V2)}


def compute_statistical_features(
    waveform: np.ndarray,
    sr: int = 16000,
    feature_names: list = None,
) -> np.ndarray:
    """Compute a subset of statistical features for one waveform clip.

    Args:
        waveform      : 1-D float32 array (from AudioPreprocessor)
        sr            : sample rate
        feature_names : which features to compute — defaults to the original 5.
                        Pass ALL_FEATURE_NAMES_V2 to include kurtosis.
    Returns:
        float32 array of shape (len(feature_names),), raw unscaled values.
    """
    if feature_names is None:
        feature_names = ALL_FEATURE_NAMES

    result = []
    for name in feature_names:
        if name == "rms":
            val = float(np.sqrt(np.mean(waveform ** 2)))

        elif name == "zcr":
            val = float(librosa.feature.zero_crossing_rate(waveform).mean())

        elif name == "centroid":
            val = float(librosa.feature.spectral_centroid(y=waveform, sr=sr).mean())

        elif name == "rolloff":
            val = float(librosa.feature.spectral_rolloff(y=waveform, sr=sr).mean())

        elif name == "bandwidth":
            val = float(librosa.feature.spectral_bandwidth(y=waveform, sr=sr).mean())

        elif name == "kurtosis":
            # Fisher's excess kurtosis: normal distribution = 0, impulsive faults > 0
            val = float(scipy_kurtosis(waveform, fisher=True))

        else:
            raise ValueError(f"Unknown feature '{name}'. Valid: {ALL_FEATURE_NAMES_V2}")

        result.append(val)

    return np.array(result, dtype=np.float32)


def compute_stat_features_v2(waveform: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Compute all 5 v2 features for .npy caching.
    Returns float32 array of shape (5,): [rms, zcr, rolloff, bandwidth, kurtosis]
    """
    return compute_statistical_features(waveform, sr, ALL_FEATURE_NAMES_V2)