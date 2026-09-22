"""
Mamba Dehazer Training Script
==============================
Entry point for training the End-to-End Vision Mamba dehazer.
Uses domain-randomized augmentations and the new MambaDehazeTrainer.
"""
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from PIL import Image
import os
import sys
import csv
import time
import random

# Windows: redirected stdout defaults to cp1252, which cannot encode the status
# glyphs printed below -> UnicodeEncodeError kills the run at the first 'new best'.
# Pin UTF-8 so `python -m training.train > train.log` behaves like the console.
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from training.trainer import MambaDehazeTrainer
from training.augmentations import HazeDomainRandomization
from torchvision import transforms

# ========================
# Hyperparameters
# ========================
EPOCHS = 50
LEARNING_RATE = 1e-4     # Lowered from 2e-4: 2e-4 peak drove the epoch-3-4 blow-up
BATCH_SIZE = 14          # Compute-bound, not VRAM-bound: bigger batch buys no speed
IMAGE_SIZE = 256
USE_MIXED_PRECISION = True
GRAD_ACCUM_STEPS = 2     # Effective batch = 14 * 2 = 28
NUM_WORKERS = 6
SEED = 42
REAL_FRACTION = 0.20     # share of each epoch drawn from real-haze datasets
WARMUP_EPOCHS = 2        # Short warmup — arch is now stable (bounded K + GroupNorm)
EMBED_DIM = 192          # SSM embedding dimension
N_LAYERS = 8             # Number of Vim blocks
D_STATE = 32             # SSM hidden state size
GRAD_CLIP_NORM = 0.5     # Extreme clipping for SSM stability
K_MAX = 5.0              # Bound on |K| in the AOD equation (stability knob)

random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MambaDehazeDataset(Dataset):
    """
    Dataset that uses HazeDomainRandomization for training
    and simple resize+ToTensor for validation.
    """
    def __init__(self, root_dir, split="train", image_size=256, augment=True):
        self.hazy_dir = os.path.join(root_dir, split, "hazy")
        self.clear_dir = os.path.join(root_dir, split, "clear")
        self.images = sorted([
            f for f in os.listdir(self.hazy_dir)
            if f.endswith(('.png', '.jpg', '.jpeg'))
        ])
        self.augment = augment

        # Per-image physics labels written by process_data.py.
        #   beta -> SOTS-outdoor only (genuine scattering coefficient)
        #   A    -> archive + SOTS-outdoor (atmospheric light)
        # real_map drives the WeightedRandomSampler that rebalances real vs synthetic.
        self.beta_map, self.a_map, self.real_map = {}, {}, {}
        meta_path = os.path.join(root_dir, split, "meta.csv")
        if os.path.exists(meta_path):
            with open(meta_path, newline="") as f:
                for row in csv.DictReader(f):
                    idx = str(row["idx"])
                    for col, dest in (("beta", self.beta_map), ("A", self.a_map)):
                        v = row.get(col, "")
                        if v not in ("", None):
                            dest[idx] = float(v)
                    self.real_map[idx] = (row.get("real_synth") == "real")

        if augment:
            self.augmentor = HazeDomainRandomization(
                image_size=image_size, apply_noise=True, p=0.5
            )
        else:
            # Centre crop, not resize: images are stored short-side=256 with their
            # true aspect ratio, so a square resize would re-introduce the distortion.
            self.transform = transforms.Compose([
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        fname = self.images[idx]
        hazy_path = os.path.join(self.hazy_dir, fname)
        clear_path = os.path.join(self.clear_dir, fname)

        hazy = Image.open(hazy_path).convert("RGB")
        clear = Image.open(clear_path).convert("RGB")

        stem = os.path.splitext(fname)[0]
        beta, A = self.beta_map.get(stem), self.a_map.get(stem)

        if self.augment:
            # jitter alters effective beta/A → invalidates the stored beta label
            hazy_tensor, clear_tensor, jittered = self.augmentor(hazy, clear)
            # jitter changes effective beta AND A -> both labels invalid for this sample
            has_beta = (beta is not None) and (not jittered)
            has_A = (A is not None) and (not jittered)
        else:
            hazy_tensor = self.transform(hazy)
            clear_tensor = self.transform(clear)
            has_beta, has_A = beta is not None, A is not None

        return (hazy_tensor, clear_tensor,
                torch.tensor(float(beta) if beta is not None else 0.0), torch.tensor(has_beta),
                torch.tensor(float(A) if A is not None else 0.0), torch.tensor(has_A))


def run_training():
    config = {
        'lr': LEARNING_RATE,
        'batch_size': BATCH_SIZE,
        'epochs': EPOCHS,
        'image_size': IMAGE_SIZE,
        'use_mixed_precision': USE_MIXED_PRECISION,
        'grad_accum_steps': GRAD_ACCUM_STEPS,
        'warmup_epochs': WARMUP_EPOCHS,
        'embed_dim': EMBED_DIM,
        'n_layers': N_LAYERS,
        'd_state': D_STATE,
        'grad_clip_norm': GRAD_CLIP_NORM,
        'k_max': K_MAX,
        'weight_decay': 1e-4,
        'dropout': 0.1,
        'w_l1': 1.0,
        'w_ssim': 0.5,
        'w_cr': 0.1,
        # --- Physics-explicit model (disentangled t/A/beta + FiLM) ---
        'model': 'physics_mamba',   # 'aod' for the static baseline
        'use_film': True,           # False = FiLM-off ablation arm (physics_mamba only)
        'w_phys': 0.2,              # ASM reconstruction consistency
        'w_beta': 0.1,             # density supervision (SOTS-outdoor only)
        'w_A': 0.1,                # airlight supervision (archive + SOTS-outdoor)
        'w_tv': 0.01,              # transmission smoothness (t is unsupervised)
        't_eps': 0.1,              # min transmission (guards the division)
    }

    # --- Datasets ---
    train_ds = MambaDehazeDataset(
        "data/processed", split="train",
        image_size=IMAGE_SIZE, augment=True
    )
    val_ds = MambaDehazeDataset(
        "data/processed", split="val",
        image_size=IMAGE_SIZE, augment=False
    )

    # Real images are only ~2.5% of the set, so the loss is dominated by synthetic
    # `archive` and the model never fits real haze (9-13 dB on real TRAINING scenes).
    # Oversample them to REAL_FRACTION of each epoch. Epoch length is unchanged.
    is_real = [train_ds.real_map.get(os.path.splitext(f)[0], False) for f in train_ds.images]
    n_real = sum(is_real)
    w_real = (REAL_FRACTION / (1 - REAL_FRACTION)) * (len(is_real) - n_real) / max(n_real, 1)
    sampler = WeightedRandomSampler(
        [w_real if r else 1.0 for r in is_real], len(train_ds), replacement=True
    )
    print(f"  Sampler:   {n_real} real / {len(is_real)} total -> weight {w_real:.1f}x "
          f"(target {REAL_FRACTION:.0%} real per epoch)")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        # 118 batches once per epoch. 6 resident pinned workers alongside the 6 train
        # workers exhausted the Windows commit limit (err 1455) around epoch 29.
        num_workers=2, pin_memory=False
    )

    print(f"=" * 60)
    print(f" End-to-End Vision Mamba Dehazer — Training")
    print(f"=" * 60)
    print(f"  Dataset:   Train={len(train_ds)} | Val={len(val_ds)}")
    print(f"  Model:     embed_dim={EMBED_DIM}, layers={N_LAYERS}, d_state={D_STATE}")
    print(f"  Training:  epochs={EPOCHS}, bs={BATCH_SIZE}, accum={GRAD_ACCUM_STEPS}")
    print(f"  Schedule:  warmup={WARMUP_EPOCHS} epochs, then cosine decay")
    print(f"  Clip:      grad_norm={GRAD_CLIP_NORM}")
    print(f"=" * 60)

    # --- Trainer ---
    trainer = MambaDehazeTrainer(config)

    # Resume from checkpoint if available
    start_epoch = 0
    best_psnr = 0.0
    checkpoint_path = "outputs/checkpoints/mamba_last.pth"

    if os.path.exists(checkpoint_path):
        print(f"\n[Resume] Loading checkpoint: {checkpoint_path}")
        start_epoch, best_psnr = trainer.load_checkpoint(checkpoint_path)
        start_epoch += 1
        print(f"[Resume] Continuing from epoch {start_epoch}, best PSNR={best_psnr:.2f}")

    # --- Training Loop ---
    for epoch in range(start_epoch, EPOCHS):
        start_time = time.time()

        # Update learning rate
        trainer.scheduler.step(epoch)
        lr = trainer.optimizer.param_groups[0]['lr']

        # Train
        train_loss = trainer.train_epoch(train_loader, epoch)

        # Validate
        val_psnr, val_ssim = trainer.validate(val_loader, epoch)

        epoch_time = time.time() - start_time

        # Log
        trainer.history['train_loss'].append(train_loss)
        trainer.history['val_psnr'].append(val_psnr)
        trainer.history['val_ssim'].append(val_ssim)
        trainer.history['lr'].append(lr)

        print(f"\n  Epoch {epoch+1}/{EPOCHS} | Loss: {train_loss:.4f} | "
              f"PSNR: {val_psnr:.2f} dB | SSIM: {val_ssim:.4f} | "
              f"LR: {lr:.2e} | Time: {epoch_time:.1f}s")

        # Save best
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            trainer.save_checkpoint("outputs/checkpoints/mamba_best.pth", epoch, best_psnr)
            print(f"  ★ New best model saved — PSNR: {best_psnr:.2f} dB")

        # Save last
        trainer.save_checkpoint(checkpoint_path, epoch, best_psnr)
        trainer.plot_history()

        # GPU cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n{'=' * 60}")
    print(f" Training Complete — Best PSNR: {best_psnr:.2f} dB")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    if not os.path.exists("data/processed/train/hazy"):
        print("ERROR: Data not processed. Run `python process_data.py` first.")
    else:
        run_training()
