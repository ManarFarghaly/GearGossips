.PHONY: infer train-phase1 train-phase2b train-best clean help

PYTHON    ?= python
DATA_DIR  ?= data

# ---------------------------------------------------------------------------
# Inference — runs the best model on a folder of wav files
# Usage: make infer DATA_DIR=/path/to/wav/folder
# Outputs: results.txt and time.txt in the current directory
# ---------------------------------------------------------------------------
infer:
	$(PYTHON) infer.py $(DATA_DIR)

# ---------------------------------------------------------------------------
# Training — run in order: Phase 1 first, then any Phase 2b variant
# ---------------------------------------------------------------------------

train-phase1:
	$(PYTHON) train_phase1V4.py

train-phase2b-v3: train-phase1
	$(PYTHON) train_phase2bV3.py

train-phase2b-v4: train-phase1
	$(PYTHON) train_phase2bV4.py

# Train best model end-to-end (Phase 1 V4 → Phase 2b V4)
train-best: train-phase1 train-phase2b-v4

# ---------------------------------------------------------------------------
# Install dependencies
# ---------------------------------------------------------------------------
install:
	pip install -r requirements.txt

# ---------------------------------------------------------------------------
# Clean generated outputs (cached features and saved models)
# ---------------------------------------------------------------------------
clean:
	rm -rf machine_listener/outputs/features/mel
	rm -rf machine_listener/outputs/features/stat_v4
	rm -f  machine_listener/outputs/saved_models/phase1_best.pth
	rm -f  machine_listener/outputs/saved_models/phase2b_v3_best.pth
	rm -f  machine_listener/outputs/saved_models/phase2b_v4_best.pth

help:
	@echo "Available targets:"
	@echo "  make infer DATA_DIR=<path>  — run inference on a folder of wav files"
	@echo "  make train-phase1           — train Phase 1 V4 backbone"
	@echo "  make train-phase2b-v3       — train Phase 2b V3 (requires phase1_best.pth)"
	@echo "  make train-phase2b-v4       — train Phase 2b V4 (best model)"
	@echo "  make train-best             — train Phase 1 then Phase 2b V4 end-to-end"
	@echo "  make install                — install Python dependencies"
	@echo "  make clean                  — remove cached features and checkpoints"
