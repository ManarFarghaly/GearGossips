# # Phase 2: MFCC extractor
# compute_mfcc(waveform, sr=16000) -> np.ndarray

# Parameters:
#   n_mfcc     = 40
#   n_fft      = 1024
#   hop_length = 512
# Steps:
#   1. librosa.feature.mfcc(y, sr, n_mfcc=40, ...)  → shape (40, 84)
#   2. Also compute delta and delta-delta → stack → shape (3, 40, 84)
#      (deltas capture temporal dynamics — important for abnormal detection)
#   3. Normalize each of the 3 channels independently to [0,1]

import librosa
import numpy as np
import train_utils as utilis

def compute_MFCC(waveform, sr=16000, n_mfcc=40, n_fft=1024, hop_length=512):
    mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    features = np.stack([mfcc, delta, delta2], axis=0)
    # Normalize each channel independently to [0,1]
    for i in range(features.shape[0]):
        features[i] = utilis.min_max_normalize(features[i])
    features = features.astype(np.float32)
    return features