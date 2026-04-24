import torch
from torch.utils.data import DataLoader

from machine_listener.src.dataset import MachineDataset
from machine_listener.src.preprocessor import AudioPreprocessor,PreprocessorConfig
from machine_listener.src.features import compute_mel_spectrogram
from machine_listener.src.models import MelCNN
import machine_listener.src.train_utils as utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root_dir = "path_to_Students_folder"

preprocessor = AudioPreprocessor(PreprocessorConfig())

train_ds = MachineDataset(
    root_dir=root_dir,
    split="train",
    preprocessor=preprocessor,
    feature_fn=compute_mel_spectrogram
)

val_ds = MachineDataset(
    root_dir=root_dir,
    split="val",
    preprocessor=preprocessor,
    feature_fn=compute_mel_spectrogram
)

test_ds = MachineDataset(
    root_dir=root_dir,
    split="test",
    preprocessor=preprocessor,
    feature_fn=compute_mel_spectrogram
)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

model = MelCNN(num_classes=6).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4
)

criterion = torch.nn.CrossEntropyLoss()

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=50
)

best_acc = 0

for epoch in range(50):
    train_loss, train_acc = utils.train_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc, preds, labels = utils.eval_epoch(model, val_loader, criterion, device)

    scheduler.step()

    print(f"Epoch {epoch}")
    print(f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    if val_acc > best_acc:
        best_acc = val_acc
        utils.save_checkpoint(model, optimizer, epoch, val_acc, "phase1_best.pth")
        