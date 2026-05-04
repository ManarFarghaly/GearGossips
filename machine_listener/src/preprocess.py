from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional
import logging
import math
import re

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

LOGGER = logging.getLogger(__name__)
EPSILON = 1e-8
Mode = Literal["inference", "train"]


@dataclass(slots=True)
class AugmentationConfig:
    """Data setting class for augmenting the data set."""

    enabled: bool = True # enable aug or no

    noise_prob: float = 0.35 # probablity el sample has noise or no
    noise_snr_db_min: float = 15.0 # min db for noise
    noise_snr_db_max: float = 35.0 # max db for noise

    time_shift_prob: float = 0.30  # Prob of applying time shift
    time_shift_max_sec: float = 0.20 

    pitch_shift_prob: float = 0.20
    pitch_shift_min_semitones: float = -1.0
    pitch_shift_max_semitones: float = 1.0

    random_crop_train: bool = True # randomise section to analyse 

    def __post_init__(self) -> None:
        _validate_probability(self.noise_prob, "noise_prob")
        _validate_probability(self.time_shift_prob, "time_shift_prob")
        _validate_probability(self.pitch_shift_prob, "pitch_shift_prob")
        if self.noise_snr_db_min > self.noise_snr_db_max:
            raise ValueError("noise_snr_db_min must be <= noise_snr_db_max")
        if self.pitch_shift_min_semitones > self.pitch_shift_max_semitones:
            raise ValueError(
                "pitch_shift_min_semitones must be <= pitch_shift_max_semitones"
            )
        if self.time_shift_max_sec < 0:
            raise ValueError("time_shift_max_sec must be >= 0")


@dataclass(slots=True)
class PreprocessConfig:
    """Config class for preprocessing"""

    target_sr: int = 16000 # sr = sample rate
    default_duration_sec: float = 3.0 # duration the model can read.

    trim_silence: bool = True # if the first 3 secs are silence better to remove and take next 3
    silence_threshold_ratio: float = 0.02
    # ta5yal el audio windows and the window size is:
    trim_frame_ms: int = 20 # this
    trim_hop_ms: int = 10 # how much you move it is this
    min_retained_sec: float = 0.25  # If this number is less than the remaining duration just keep the OG

    denoise: bool = False # Hal toreed el denoising
    denoise_prop_decrease: float = 0.80 

    normalize_mode: Literal["peak", "rms", "none"] = "peak" # normalise audio? default normalise to peak
    peak_target: float = 0.95 # max amp to normlaise to
    rms_target: float = 0.10 # theoretical rms to normalise to

    clip_value: float = 1.0 # clip after this

    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig) 

    def __post_init__(self) -> None:
        if self.target_sr <= 0:
            raise ValueError("target_sr must be > 0")
        if self.default_duration_sec <= 0:
            raise ValueError("default_duration_sec must be > 0")
        if self.silence_threshold_ratio < 0:
            raise ValueError("silence_threshold_ratio must be >= 0")
        if self.trim_frame_ms <= 0:
            raise ValueError("trim_frame_ms must be > 0")
        if self.trim_hop_ms <= 0:
            raise ValueError("trim_hop_ms must be > 0")
        if self.min_retained_sec < 0:
            raise ValueError("min_retained_sec must be >= 0")
        if self.denoise_prop_decrease < 0 or self.denoise_prop_decrease > 1:
            raise ValueError("denoise_prop_decrease must be in [0, 1]")
        if self.normalize_mode not in {"peak", "rms", "none"}:
            raise ValueError("normalize_mode must be one of: peak, rms, none")
        if self.peak_target <= 0:
            raise ValueError("peak_target must be > 0")
        if self.rms_target <= 0:
            raise ValueError("rms_target must be > 0")
        if self.clip_value <= 0:
            raise ValueError("clip_value must be > 0")


class AudioPreprocessor:
    """
    End-to-end audio preprocessing pipeline.

    Design principles:
    1. Deterministic inference behavior.
    2. Configurable fixed-length output via duration_sec.
    3. Training-only augmentation hooks.
    4. Safe fallback behavior for malformed/corrupted inputs.
    """

    def __init__(self, config: Optional[PreprocessConfig] = None) -> None:
        self.config = config or PreprocessConfig()

    def preprocess(
        self,
        audio_path: str | Path,
        duration_sec: Optional[float] = None,
        target_sr: Optional[int] = None,
        mode: Mode = "inference",
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Preprocess one audio file and return a fixed-length clean waveform.

        Args:
            audio_path: Path to a .wav file.
            duration_sec: Desired output duration in seconds.
            target_sr: Desired output sample rate.
            mode: "inference" or "train".
            seed: Random seed used only when mode="train".

        Returns:
            np.ndarray: 1D float32 waveform with length int(target_sr * duration_sec).
        """
        if mode not in {"inference", "train"}:
            raise ValueError("mode must be either 'inference' or 'train'")

        effective_sr = int(target_sr if target_sr is not None else self.config.target_sr)
        effective_duration = float(
            duration_sec
            if duration_sec is not None
            else self.config.default_duration_sec
        )

        if effective_sr <= 0:
            raise ValueError("target_sr must be > 0")
        if effective_duration <= 0:
            raise ValueError("duration_sec must be > 0")

        target_len = int(round(effective_sr * effective_duration))
        if target_len <= 0:
            raise ValueError("int(target_sr * duration_sec) must be > 0")

        try:
            waveform, original_sr = self._load_audio(audio_path)
        except Exception as exc:
            LOGGER.warning("Failed to read '%s': %s", audio_path, exc)
            return np.zeros(target_len, dtype=np.float32)

        waveform = self._ensure_mono(waveform)
        if original_sr != effective_sr:
            waveform = self._resample(waveform, original_sr, effective_sr)

        if self.config.denoise:
            waveform = self._denoise(waveform, effective_sr)

        if self.config.trim_silence:
            waveform = self._trim_leading_trailing_silence(waveform, effective_sr)

        waveform = self._normalize(waveform)

        rng = np.random.default_rng(seed) if mode == "train" else None
        if mode == "train":
            waveform = self._apply_training_augmentations(waveform, effective_sr, rng)

        waveform = self._fix_length(
            waveform,
            target_len=target_len,
            training=(mode == "train"),
            rng=rng,
        )
        waveform = self._sanitize(waveform)
        return waveform

    def _load_audio(self, audio_path: str | Path) -> tuple[np.ndarray, int]:
        """Load audio while preserving source sample rate."""
        data, sr = sf.read(str(audio_path), always_2d=False, dtype="float32")
        if sr <= 0:
            raise ValueError("Invalid sample rate read from audio file")

        waveform = np.asarray(data, dtype=np.float32)
        if waveform.size == 0:
            raise ValueError("Audio file is empty")
        return waveform, int(sr)

    def _ensure_mono(self, waveform: np.ndarray) -> np.ndarray:
        """Collapse multi-channel audio to mono by channel averaging."""
        if waveform.ndim == 1:
            return waveform

        if waveform.ndim > 2:
            waveform = waveform.reshape(waveform.shape[0], -1)

        mono = waveform.mean(axis=1)
        return mono.astype(np.float32, copy=False)

    def _resample(
        self,
        waveform: np.ndarray,
        source_sr: int,
        target_sr: int,
    ) -> np.ndarray:
        """Resample using polyphase filtering for speed and quality."""
        if source_sr == target_sr:
            return waveform

        divisor = math.gcd(source_sr, target_sr)
        up = target_sr // divisor
        down = source_sr // divisor
        resampled = resample_poly(waveform, up=up, down=down)
        return np.asarray(resampled, dtype=np.float32)

    def _denoise(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        """Optional denoising via noisereduce; skipped if dependency is unavailable."""
        if waveform.size == 0:
            return waveform

        try:
            import noisereduce as nr  # type: ignore
        except ImportError:
            LOGGER.debug("noisereduce is not installed; denoising is skipped")
            return waveform

        try:
            reduced = nr.reduce_noise(
                y=waveform,
                sr=sample_rate,
                stationary=True,
                prop_decrease=self.config.denoise_prop_decrease,
            )
            return np.asarray(reduced, dtype=np.float32)
        except Exception as exc:
            LOGGER.warning("Denoising failed and was skipped: %s", exc)
            return waveform

    def _trim_leading_trailing_silence(
        self,
        waveform: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:
        """
        Trim leading/trailing silence using frame-wise max energy.

        This intentionally avoids aggressive internal silence removal.
        """
        if waveform.size == 0:
            return waveform

        abs_wave = np.abs(waveform)
        peak = float(np.max(abs_wave))
        if peak <= EPSILON:
            return waveform

        threshold = peak * self.config.silence_threshold_ratio
        frame_len = max(1, int(sample_rate * self.config.trim_frame_ms / 1000))
        hop_len = max(1, int(sample_rate * self.config.trim_hop_ms / 1000))

        if waveform.size <= frame_len:
            return waveform

        active_starts: list[int] = []
        last_start = waveform.size - frame_len
        for start in range(0, last_start + 1, hop_len):
            frame_peak = float(np.max(abs_wave[start : start + frame_len]))
            if frame_peak >= threshold:
                active_starts.append(start)

        if not active_starts:
            return waveform

        start_idx = active_starts[0]
        end_idx = min(waveform.size, active_starts[-1] + frame_len)
        trimmed = waveform[start_idx:end_idx]

        min_retained_samples = int(round(self.config.min_retained_sec * sample_rate))
        if min_retained_samples > 0 and trimmed.size < min_retained_samples:
            return waveform

        return np.asarray(trimmed, dtype=np.float32)

    def _normalize(self, waveform: np.ndarray) -> np.ndarray:
        """Normalize waveform by peak or RMS based on config."""
        if waveform.size == 0:
            return waveform

        mode = self.config.normalize_mode
        if mode == "none":
            return waveform

        if mode == "peak":
            peak = float(np.max(np.abs(waveform)))
            if peak > EPSILON:
                waveform = waveform * (self.config.peak_target / peak)
            return np.asarray(waveform, dtype=np.float32)

        rms = float(np.sqrt(np.mean(np.square(waveform), dtype=np.float64)))
        if rms > EPSILON:
            waveform = waveform * (self.config.rms_target / rms)
        return np.asarray(waveform, dtype=np.float32)

    def _apply_training_augmentations(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Apply stochastic training augmentations under bounded ranges."""
        aug = self.config.augmentation
        if not aug.enabled or waveform.size == 0:
            return waveform

        augmented = waveform
        if rng.random() < aug.noise_prob:
            augmented = self._augment_add_noise(augmented, rng)
        if rng.random() < aug.time_shift_prob:
            augmented = self._augment_time_shift(augmented, sample_rate, rng)
        if rng.random() < aug.pitch_shift_prob:
            augmented = self._augment_pitch_shift(augmented, sample_rate, rng)

        return np.asarray(augmented, dtype=np.float32)

    def _augment_add_noise(
        self,
        waveform: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Add noise at a random SNR for robustness."""
        signal_rms = float(np.sqrt(np.mean(np.square(waveform), dtype=np.float64)))
        if signal_rms <= EPSILON:
            return waveform

        aug = self.config.augmentation
        snr_db = float(rng.uniform(aug.noise_snr_db_min, aug.noise_snr_db_max))

        noise = rng.normal(0.0, 1.0, waveform.shape).astype(np.float32)
        noise_rms = float(np.sqrt(np.mean(np.square(noise), dtype=np.float64)))
        if noise_rms <= EPSILON:
            return waveform

        desired_noise_rms = signal_rms / (10.0 ** (snr_db / 20.0))
        scaled_noise = noise * (desired_noise_rms / (noise_rms + EPSILON))
        return waveform + scaled_noise

    def _augment_time_shift(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Circularly shift the waveform by a bounded random amount."""
        max_shift = int(round(self.config.augmentation.time_shift_max_sec * sample_rate))
        if max_shift <= 0:
            return waveform

        shift = int(rng.integers(-max_shift, max_shift + 1))
        if shift == 0:
            return waveform

        return np.roll(waveform, shift)

    def _augment_pitch_shift(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Pitch-shift via librosa when available; skipped otherwise."""
        aug = self.config.augmentation
        n_steps = float(
            rng.uniform(aug.pitch_shift_min_semitones, aug.pitch_shift_max_semitones)
        )
        if abs(n_steps) <= EPSILON:
            return waveform

        try:
            import librosa  # type: ignore
        except ImportError:
            LOGGER.debug("librosa is not installed; pitch shift is skipped")
            return waveform

        try:
            shifted = librosa.effects.pitch_shift(
                y=waveform,
                sr=sample_rate,
                n_steps=n_steps,
            )
            return np.asarray(shifted, dtype=np.float32)
        except Exception as exc:
            LOGGER.warning("Pitch shifting failed and was skipped: %s", exc)
            return waveform

    def _fix_length(
        self,
        waveform: np.ndarray,
        target_len: int,
        training: bool,
        rng: Optional[np.random.Generator],
    ) -> np.ndarray:
        """
        Enforce output length.

        In inference: deterministic center crop for long clips and right padding for short clips.
        In training: optional random crop and random left/right pad split.
        """
        current_len = waveform.size
        if current_len == target_len:
            return waveform

        if current_len > target_len:
            if (
                training
                and self.config.augmentation.random_crop_train
                and rng is not None
            ):
                max_start = current_len - target_len
                start_idx = int(rng.integers(0, max_start + 1))
            else:
                start_idx = (current_len - target_len) // 2

            return waveform[start_idx : start_idx + target_len]

        pad_total = target_len - current_len
        if (
            training
            and self.config.augmentation.random_crop_train
            and rng is not None
        ):
            left_pad = int(rng.integers(0, pad_total + 1))
        else:
            left_pad = 0
        right_pad = pad_total - left_pad

        return np.pad(waveform, (left_pad, right_pad), mode="constant")

    def _sanitize(self, waveform: np.ndarray) -> np.ndarray:
        """Guarantee numeric safety and bounded amplitude."""
        cleaned = np.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0)
        np.clip(cleaned, -self.config.clip_value, self.config.clip_value, out=cleaned)
        return np.ascontiguousarray(cleaned, dtype=np.float32)


def preprocess(
    audio_path: str | Path,
    duration_sec: float = 3.0,
    target_sr: int = 16000,
    mode: Mode = "inference",
    seed: Optional[int] = None,
    preprocessor: Optional[AudioPreprocessor] = None,
    config: Optional[PreprocessConfig] = None,
) -> np.ndarray:
    """
    Convenience wrapper with requested signature shape.

    This returns a clean waveform with length int(target_sr * duration_sec).
    """
    if preprocessor is not None and config is not None:
        raise ValueError("Pass either preprocessor or config, not both")

    engine = preprocessor or AudioPreprocessor(config=config)
    return engine.preprocess(
        audio_path=audio_path,
        duration_sec=duration_sec,
        target_sr=target_sr,
        mode=mode,
        seed=seed,
    )


def iter_wav_files_numeric(data_dir: str | Path) -> list[Path]:
    """Return .wav files sorted by numeric filename (1.wav, 2.wav, ...)."""
    root = Path(data_dir)
    if not root.exists() or not root.is_dir():
        raise ValueError(f"Invalid directory: {data_dir}")

    files = [path for path in root.iterdir() if path.is_file() and path.suffix.lower() == ".wav"]
    return sorted(files, key=_numeric_stem_key)


def preprocess_directory(
    data_dir: str | Path,
    duration_sec: float = 3.0,
    target_sr: int = 16000,
    mode: Mode = "inference",
    seed: Optional[int] = None,
    preprocessor: Optional[AudioPreprocessor] = None,
    config: Optional[PreprocessConfig] = None,
) -> dict[str, np.ndarray]:
    """
    Preprocess all WAV files in numeric order and return a filename->waveform mapping.
    """
    engine = preprocessor or AudioPreprocessor(config=config)
    outputs: dict[str, np.ndarray] = {}
    for wav_file in iter_wav_files_numeric(data_dir):
        outputs[wav_file.name] = engine.preprocess(
            audio_path=wav_file,
            duration_sec=duration_sec,
            target_sr=target_sr,
            mode=mode,
            seed=seed,
        )
    return outputs


def _numeric_stem_key(path: Path) -> tuple[int, int, str]:
    match = re.search(r"\d+", path.stem)
    if match:
        return 0, int(match.group()), path.stem.lower()
    return 1, 0, path.stem.lower()


def _validate_probability(value: float, name: str) -> None:
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be in [0, 1]")

 
