"""
Epoch-time / VRAM probe — size a run before committing hours to it.
===================================================================
Times a handful of real training steps at the CURRENT config in training/train.py,
extrapolates to a full epoch, and prints the EPOCHS value that fits a wall-clock
budget. EPOCHS cannot be changed after launch without invalidating the cosine
schedule (WarmupCosineScheduler.total_epochs), so it has to be chosen up front.

Reuses MambaDehazeTrainer / MambaDehazeDataset directly, so what it measures is the
real step (forward -> PhysicsDehazeLoss incl. ConvNeXt -> backward -> clip -> step),
not an approximation of it.

Run:  PYTHONPATH=. python scripts/probe_epoch_time.py [--budget-hours 8] [--steps 20]
"""
import argparse
import time

import torch
from torch.utils.data import DataLoader

import training.train as T
from training.train import MambaDehazeDataset
from training.trainer import MambaDehazeTrainer

WARMUP_STEPS = 3      # discard: cudnn autotune, allocator growth, ConvNeXt load
SAFETY = 0.9          # leave 10% of the budget for validation drift / checkpointing


def build_config():
    """Mirror the config dict run_training() builds, from the same module globals."""
    return {
        'lr': T.LEARNING_RATE, 'batch_size': T.BATCH_SIZE, 'epochs': T.EPOCHS,
        'image_size': T.IMAGE_SIZE, 'use_mixed_precision': T.USE_MIXED_PRECISION,
        'grad_accum_steps': T.GRAD_ACCUM_STEPS, 'warmup_epochs': T.WARMUP_EPOCHS,
        'embed_dim': T.EMBED_DIM, 'n_layers': T.N_LAYERS, 'd_state': T.D_STATE,
        'grad_clip_norm': T.GRAD_CLIP_NORM, 'k_max': T.K_MAX,
        'weight_decay': 1e-4, 'dropout': 0.1,
        'w_l1': 1.0, 'w_ssim': 0.5, 'w_cr': 0.1,
        'model': 'physics_mamba',
        'w_phys': 0.2, 'w_beta': 0.1, 'w_tv': 0.01, 't_eps': 0.1,
    }


def time_steps(trainer, loader, n_steps):
    """Median-free mean over n_steps real optimizer steps, after WARMUP_STEPS."""
    cfg = trainer.config
    accum = cfg.get('grad_accum_steps', 1)
    trainer.model.train()
    t0 = None
    done = 0

    for i, (hazy, clear, beta, has_beta) in enumerate(loader):
        hazy = hazy.to(trainer.device, non_blocking=True)
        clear = clear.to(trainer.device, non_blocking=True)
        beta = beta.to(trainer.device, non_blocking=True)
        has_beta = has_beta.to(trainer.device, non_blocking=True).bool()

        with torch.autocast('cuda', enabled=cfg['use_mixed_precision']):
            out = trainer.model(hazy)
            loss, _ = trainer.criterion(out, clear, hazy, beta, has_beta)
            loss = loss / accum

        trainer.scaler.scale(loss).backward()
        if (i + 1) % accum == 0:
            trainer.scaler.unscale_(trainer.optimizer)
            torch.nn.utils.clip_grad_norm_(trainer.model.parameters(),
                                           trainer.grad_clip_norm)
            trainer.scaler.step(trainer.optimizer)
            trainer.scaler.update()
            trainer.optimizer.zero_grad(set_to_none=True)

        if i + 1 == WARMUP_STEPS:                 # start the clock after warmup
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        elif t0 is not None:
            done += 1
            if done >= n_steps:
                break

    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-hours", type=float, default=8.0)
    ap.add_argument("--steps", type=int, default=20)
    # Optional capacity overrides, so sizes can be compared without editing train.py
    ap.add_argument("--embed", type=int)
    ap.add_argument("--layers", type=int)
    ap.add_argument("--dstate", type=int)
    ap.add_argument("--batch", type=int, help="BATCH_SIZE override")
    ap.add_argument("--accum", type=int, help="GRAD_ACCUM_STEPS override")
    args = ap.parse_args()

    for attr, val in (("EMBED_DIM", args.embed), ("N_LAYERS", args.layers),
                      ("D_STATE", args.dstate), ("BATCH_SIZE", args.batch),
                      ("GRAD_ACCUM_STEPS", args.accum)):
        if val is not None:
            setattr(T, attr, val)

    cfg = build_config()
    train_ds = MambaDehazeDataset("data/processed", split="train",
                                  image_size=T.IMAGE_SIZE, augment=True)
    val_ds = MambaDehazeDataset("data/processed", split="val",
                                image_size=T.IMAGE_SIZE, augment=False)
    loader = DataLoader(train_ds, batch_size=T.BATCH_SIZE, shuffle=True,
                        num_workers=T.NUM_WORKERS, pin_memory=True, drop_last=True,
                        persistent_workers=True)

    torch.cuda.reset_peak_memory_stats()
    trainer = MambaDehazeTrainer(cfg)
    params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)

    sec_per_step = time_steps(trainer, loader, args.steps)
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3

    steps_per_epoch = len(train_ds) // T.BATCH_SIZE
    train_min = sec_per_step * steps_per_epoch / 60
    # Validation: no backward, no ConvNeXt on 3 sets -> ~35% of a train step, and
    # 1645 val images vs 13137 train. Measured share is small; approximate it.
    val_min = sec_per_step * 0.35 * (len(val_ds) / T.BATCH_SIZE) / 60
    epoch_min = train_min + val_min
    fit = int(args.budget_hours * 60 * SAFETY // epoch_min)

    print("\n" + "=" * 64)
    print(f"  config      embed={T.EMBED_DIM} layers={T.N_LAYERS} d_state={T.D_STATE}"
          f"  bs={T.BATCH_SIZE} accum={T.GRAD_ACCUM_STEPS}")
    print(f"  params      {params:,}")
    print(f"  peak VRAM   {peak_gb:.2f} / {total_gb:.1f} GB")
    print(f"  per step    {sec_per_step:.3f} s   ({steps_per_epoch} steps/epoch)")
    print(f"  per epoch   {epoch_min:.1f} min   (train {train_min:.1f} + val ~{val_min:.1f})")
    print(f"  50 epochs   {epoch_min * 50 / 60:.1f} h")
    print("-" * 64)
    print(f"  EPOCHS that fit {args.budget_hours:g} h (with {1-SAFETY:.0%} headroom): {fit}")
    if peak_gb > total_gb * 0.85:
        print(f"  WARNING: VRAM at {peak_gb/total_gb:.0%} of card — halve BATCH_SIZE "
              f"and set GRAD_ACCUM_STEPS=4 to hold effective batch at 28.")
    print("=" * 64 + "\n")


if __name__ == "__main__":
    main()
