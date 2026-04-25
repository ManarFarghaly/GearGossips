"""
Phase 3 — Global statistical feature extractor.

Input : waveform  np.ndarray shape (44000,)
Output: np.ndarray shape (5,)   — one scalar per descriptor

The 5 descriptors:
  RMS energy        — how loud / energetic the signal is overall
  ZCR               — how often the signal crosses zero  (relates to pitch/noise)
  Spectral Centroid — "centre of mass" of the spectrum    (brightness)
  Spectral Rolloff  — frequency below which 85% of energy sits (high vs low content)
  Spectral Bandwidth— spread of the spectrum around its centroid (tone vs noise)

These are GLOBAL (one number per clip), so they feed into a small FC branch
that runs in parallel with the CNN, not replacing it.

IMPORTANT: these 5 values live on very different scales (e.g. RMS ≈ 0.1, Centroid ≈ 3000).
You MUST fit a StandardScaler on the training set and apply it before passing these
to the model.  The scaler is handled in the training script, not here.
"""

import librosa
import numpy as np


ALL_FEATURE_NAMES = ["rms", "zcr", "centroid", "rolloff", "bandwidth"]


def compute_statistical_features(
    waveform: np.ndarray,
    sr: int = 16000,
    feature_names: list = None,
) -> np.ndarray:
    """
    Args:
        waveform      : 1-D float32 array from AudioPreprocessor
        sr            : sample rate
        feature_names : subset of ALL_FEATURE_NAMES to compute.
                        None → compute all 5.

    Returns:
        np.ndarray of shape (len(feature_names),), dtype float32
        Values are RAW (not normalized) — normalize with StandardScaler in train script.
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
            val = float(
                librosa.feature.spectral_centroid(y=waveform, sr=sr).mean()
            )

        elif name == "rolloff":
            val = float(
                librosa.feature.spectral_rolloff(y=waveform, sr=sr).mean()
            )

        elif name == "bandwidth":
            val = float(
                librosa.feature.spectral_bandwidth(y=waveform, sr=sr).mean()
            )

        else:
            raise ValueError(
                f"Unknown feature '{name}'. Choose from {ALL_FEATURE_NAMES}"
            )

        result.append(val)

    return np.array(result, dtype=np.float32)