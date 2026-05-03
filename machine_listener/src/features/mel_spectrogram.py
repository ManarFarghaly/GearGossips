"""
Input : waveform  np.ndarray shape (44000,)  — already preprocessed by AudioPreprocessor
Output: np.ndarray shape (1, 128, 84)         — ready for the 2-D CNN

    - power_to_db: Convert a power spectrogram (amplitude squared) to decibel (dB) units
        It shrinks big values and boosts small ones, so you can see everything, not just loud parts
    - As SR is 16000, the max frequency is 8000 (half of SR), so we can ignore frequencies above that
    - why n_mels or image length = 128 -> Balance: too small (e.g., 32) → lose detail , too large (e.g., 512) → noisy + heavy model
        128 is a standard compromise in audio ML
    - why image width or time frames = 84 -> This is number of time frames (image width) 
    It depends on: [1 + (len(y) - n_fft) // hop_length]
    so samples = 2.75 sec × 16 kHz ≈ 44000 samples ,so frames = 1+(44000 - 1024) // 512 ≈ 84 frames
    - Shape transformation → (1, 128, 84) 
    Why:
        CNNs expect channels so 
        1 = grayscale channel
        128 = frequency axis
        84 = time axis
        
Shape explanation:
  1   → grayscale channel  (CNNs need an explicit channel dimension)
  128 → mel frequency bins (n_mels)
  84  → time frames        = 1 + (44000 - 1024) // 512  ≈ 84
"""

import librosa
import numpy as np


def compute_mel_spectrogram(
    waveform: np.ndarray,
    sr: int = 16000,
    n_mels: int = 128,
    n_fft: int = 1024,
    hop_length: int = 512,
    fmin: float = 50.0,
    fmax: float = 8000.0,
) -> np.ndarray:
    """
    Args:
        waveform   : 1-D float32 array, output of AudioPreprocessor (do NOT pass a path)
        sr         : sample rate (must match what the preprocessor used — 16 000 Hz)
        n_mels     : height of the output image (frequency axis)
        n_fft      : FFT window size  — controls frequency resolution
        hop_length : step between windows — controls time resolution
        fmin/fmax  : ignore frequencies outside this range

    Returns:
        np.ndarray of shape (1, n_mels, time_frames), dtype float32, values in [0, 1]
    """
    mel = librosa.feature.melspectrogram(
        y=waveform, sr=sr,
        n_mels=n_mels, n_fft=n_fft, hop_length=hop_length,
        fmin=fmin, fmax=fmax,
        center=False,        # no padding at edges → exact 84 frames
    )                        # shape: (128, 84)

    mel_db = librosa.power_to_db(mel, ref=np.max)   # convert to dB scale

    mel_db = _min_max_normalize(mel_db)             # scale to [0, 1]

    mel_db = mel_db.astype(np.float32)

    mel_db = np.expand_dims(mel_db, axis=0)         # (128,84) → (1,128,84)

    return mel_db

def _min_max_normalize(S: np.ndarray) -> np.ndarray:
    lo, hi = S.min(), S.max()
    if hi - lo < 1e-8:
        return np.zeros_like(S)
    return (S - lo) / (hi - lo)
