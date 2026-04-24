# Phase 1: mel-spec extractor
import librosa
import numpy as np

"""
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
"""
import train_utils as utilis

def calculate_mel_spectrogram(audio_path, n_mels=128, hop_length=512, n_fft=1024, fmin=50, fmax=8000):
    """
    Args:
        audio_path (_type_): the path to the audio file
        n_fft = 1024 → how detailed frequencies are
        hop_length = 512 → how often you “take a snapshot”
        n_mels = 128 → image height
        fmin/fmax → ignore useless frequencies
        
    Returns:
        mel_spectrogram_db (np.2darray): the mel-spectrogram in decibel units, normalized to [0, 1], and shaped (1, n_mels, time_frames)
    """
    # sr is none to keep original sample rate, y is the audio time series (array of sound points)
    y, sr = librosa.load(audio_path, sr = None)
    
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length, n_fft=n_fft, fmin=fmin, fmax=fmax, center=False)
    
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    
    mel_spectrogram_db = utilis.min_max_normalize(mel_spectrogram_db)
    
    mel_spectrogram_db = mel_spectrogram_db.astype(np.float32)
    
    mel_spectrogram_db = np.expand_dims(mel_spectrogram_db, axis=0)
    
    return mel_spectrogram_db