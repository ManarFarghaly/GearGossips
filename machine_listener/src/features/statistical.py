"""
Global statistical feature extractor — one scalar per clip.

v1 features (Phase 3):
  rms, zcr, centroid, rolloff, bandwidth

v2 features (Phase 2b/4b) — centroid dropped (redundant with mel CNN), kurtosis added:
  rms, zcr, rolloff, bandwidth, kurtosis

v3 features (ablation) — full set including spectral_flux:
  rms, zcr, rolloff, bandwidth, kurtosis, spectral_flux
  spectral_flux: mean frame-to-frame spectral change; catches irregular fault patterns
  without the "already impulsive baseline" problem kurtosis has for Machine2.

All values are RAW — normalise before passing to the model.
"""

import librosa
import numpy as np
from scipy.stats import kurtosis as scipy_kurtosis


# v1 — kept for backward compatibility
ALL_FEATURE_NAMES = ["rms", "zcr", "centroid", "rolloff", "bandwidth"]

# v2 — centroid dropped, kurtosis added
ALL_FEATURE_NAMES_V2 = ["rms", "zcr", "rolloff", "bandwidth", "kurtosis"]

# v3 — full 6-feature set for ablation (superset, slice to get any subset)
ALL_FEATURE_NAMES_V3 = ["rms", "zcr", "rolloff", "bandwidth", "kurtosis", "spectral_flux"]

STAT_COL_V2 = {name: i for i, name in enumerate(ALL_FEATURE_NAMES_V2)}
STAT_COL_V3 = {name: i for i, name in enumerate(ALL_FEATURE_NAMES_V3)}


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
            val = float(scipy_kurtosis(waveform, fisher=True))

        elif name == "spectral_flux":
            S = np.abs(librosa.stft(waveform, n_fft=1024, hop_length=512))
            val = float(np.mean(np.sum(np.diff(S, axis=1) ** 2, axis=0)))

        else:
            raise ValueError(f"Unknown feature '{name}'. Valid: {ALL_FEATURE_NAMES_V3}")

        result.append(val)

    return np.array(result, dtype=np.float32)


def compute_stat_features_v2(waveform: np.ndarray, sr: int = 16000) -> np.ndarray:
    """All 5 v2 features: [rms, zcr, rolloff, bandwidth, kurtosis]"""
    return compute_statistical_features(waveform, sr, ALL_FEATURE_NAMES_V2)


def compute_stat_features_v3(waveform: np.ndarray, sr: int = 16000) -> np.ndarray:
    """All 6 v3 features: [rms, zcr, rolloff, bandwidth, kurtosis, spectral_flux]
    Use this for the ablation cache — precompute once, then slice whichever subset you need.
    """
    return compute_statistical_features(waveform, sr, ALL_FEATURE_NAMES_V3)