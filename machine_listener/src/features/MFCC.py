"""
Phase 2 — MFCC feature extractor (+ delta and delta-delta).

Input : waveform  np.ndarray shape (44000,)  — already preprocessed
Output: np.ndarray shape (3, 40, 84)

Channel breakdown:
  channel 0 → raw MFCCs           — WHERE energy is in each frame
  channel 1 → delta (1st order)   — HOW FAST the MFCCs are changing   (velocity)
  channel 2 → delta-delta (2nd)   — HOW FAST the velocity is changing  (acceleration)

Why deltas matter for fault detection:
  Normal machines have steady, repetitive MFCC patterns → small deltas.
  Abnormal machines have irregular changes        → large deltas.
  Without deltas the model only sees a static snapshot; with them it sees motion.
"""

import librosa
import numpy as np


def compute_mfcc(
    waveform: np.ndarray,
    sr: int = 16000,
    n_mfcc: int = 40,
    n_fft: int = 1024,
    hop_length: int = 512,
) -> np.ndarray:
    """
    Args:
        waveform  : 1-D float32 array from AudioPreprocessor
        sr        : sample rate
        n_mfcc    : number of MFCC coefficients (height of output)
        n_fft     : FFT window size — same as mel-spec so time axis matches (84 frames)
        hop_length: step size      — same as mel-spec so time axis matches (84 frames)

    Returns:
        np.ndarray of shape (3, n_mfcc, time_frames), dtype float32, values in [0, 1]
        Same number of time frames as compute_mel_spectrogram (both use same hop_length).
    """
    mfcc   = librosa.feature.mfcc(
        y=waveform, sr=sr,
        n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length,
    )                                                   # (40, 84)

    delta  = librosa.feature.delta(mfcc)               # (40, 84)
    delta2 = librosa.feature.delta(mfcc, order=2)      # (40, 84)

    features = np.stack([mfcc, delta, delta2], axis=0)  # (3, 40, 84)

    # Normalize each channel independently to [0, 1]
    for i in range(features.shape[0]):
        features[i] = _min_max_normalize(features[i])

    return features.astype(np.float32)


# ---------------------------------------------------------------------------
def _min_max_normalize(S: np.ndarray) -> np.ndarray:
    lo, hi = S.min(), S.max()
    if hi - lo < 1e-8:
        return np.zeros_like(S)
    return (S - lo) / (hi - lo)
