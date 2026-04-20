from pathlib import Path

import soundfile as sf

from preprocess import AudioPreprocessor, AugmentationConfig, PreprocessConfig


# 1) Find your project folder and audio file
project_root = Path(__file__).resolve().parent
audio_path = project_root / "Students" / "Machine 1" / "machine_data" / "Abnormal" / "1.wav"

if not audio_path.exists():
    print("Could not find audio file:", audio_path)
    print("Tip: make sure Students is inside GearGossips.")
    raise SystemExit(1)

# 2) Make a folder for output files
out_dir = project_root / "demo_outputs"
out_dir.mkdir(parents=True, exist_ok=True)

# 3) Basic settings
# target_sr means sample rate (sr):
# how many sound points are stored per second.
# 16000 means 16,000 points each second.
target_sr = 16000

# duration_sec means final clip length in seconds.
# 2.75 means every output clip is exactly 2.75 seconds.
duration_sec = 2.75

# 4) Inference setup (real-world use, no random augmentation)
inference_config = PreprocessConfig(
    target_sr=target_sr,
    default_duration_sec=duration_sec,
    trim_silence=True,                # remove quiet start/end when possible
    denoise=False,                    # keep denoise off unless needed
    normalize_mode="peak",            # normalize volume
    augmentation=AugmentationConfig(enabled=False),  # no random changes
)

inference_preprocessor = AudioPreprocessor(inference_config)

y_inference = inference_preprocessor.preprocess(
    audio_path=audio_path,
    duration_sec=duration_sec,
    target_sr=target_sr,
    mode="inference",                 # deterministic mode
)

expected_len = int(round(target_sr * duration_sec))
print("Inference shape:", y_inference.shape)
print("Expected length:", expected_len)
print("Inference dtype:", y_inference.dtype)

# 5) Training setup (random augmentation on purpose)
train_config = PreprocessConfig(
    target_sr=target_sr,
    default_duration_sec=duration_sec,
    trim_silence=True,
    denoise=False,
    normalize_mode="peak",
    augmentation=AugmentationConfig(
        enabled=True,
        noise_prob=0.7,               # add noise sometimes
        time_shift_prob=0.4,          # shift audio in time sometimes
        pitch_shift_prob=0.2,         # change pitch sometimes
    ),
)

train_preprocessor = AudioPreprocessor(train_config)

y_train = train_preprocessor.preprocess(
    audio_path=audio_path,
    duration_sec=duration_sec,
    target_sr=target_sr,
    mode="train",                     # stochastic mode
    seed=42,                          # controls random behavior
)

print("Train shape:", y_train.shape)
print("Train dtype:", y_train.dtype)

# 6) Save outputs so you can listen
sf.write(out_dir / "inference.wav", y_inference, target_sr)
sf.write(out_dir / "train_augmented.wav", y_train, target_sr)

print("Saved files:")
print(out_dir / "inference.wav")
print(out_dir / "train_augmented.wav")