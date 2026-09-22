# CLAUDE.md — Project Handoff & Context

> Read this first. It is the complete brief for this repo. It was written to hand the
> project from one machine to a new **RTX 5070 Ti (12 GB, Blackwell)** laptop.

---

## 1. What this project is

A **physics-informed, condition-adaptive single-image dehazer** built on a **Vision Mamba (bidirectional S4D SSM)** backbone. Capstone / thesis project ("atmos").

The novelty is NOT the Mamba backbone (Mamba-for-dehazing is already published). The thesis claim is the **dynamic / adaptive** part:

> The network senses a continuous **atmospheric state** (atmospheric light `A`, haze density `β`) from the input and **modulates itself (FiLM)** per image, while estimating the **explicit physics parameters** `(t, A)` and reconstructing through the atmospheric-scattering model (ASM). "Dynamic" = behaviour varies continuously with the sensed environment, not a switch over discrete classes.

**Committed scope (do not expand without asking the user):** FiLM conditioning + physics-explicit `(t, A, β)`, **focused & provable** (one strong, fully-ablated claim). **Deferred / out of scope:** test-time adaptation (TTA), Mixture-of-Experts, night / underwater / dust / rain / remote-sensing (no data for those — `rshaze` raw dir is empty).

---

## 2. ⚠ CRITICAL environment note (read before installing torch)

This laptop's **RTX 5070 Ti is Blackwell (sm_120)**. The default PyPI torch wheel (cu124) will fail at runtime with *"no kernel image is available for execution on the device."*

**Install the cu128+ build:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```
Then the rest:
```bash
pip install -r requirements.txt   # fastapi, uvicorn, opencv-python, tqdm, optuna, matplotlib, numpy, pillow, kaggle
```
Verify: `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"` → must print `True` and a cu128 build.

---

## 3. First-time setup on this machine (in order)

Data and outputs are **git-ignored** (too big for GitHub) — you must regenerate them:

```bash
# 1. Datasets (needs kaggle.json in ~/.kaggle/ — user provides it)
python download_datasets.py

# 2. Preprocess → leakage-free, metadata-preserving data/processed/{train,val,test}/
python process_data.py --dry-run   # sanity: prints per-dataset counts + "leakage-free OK"
python process_data.py             # real run (~16k images, a few minutes)

# 3. Train the physics-explicit model
python -m training.train
```
If `data/raw` can't be downloaded (Kaggle), the user has the raw zips; extract under `data/raw/thesis/` and `data/raw/haze1k/`.

---

## 4. Architecture

Two models live here. **`PhysicsMambaDehaze` is the primary/thesis model.** The AOD model is the **baseline for ablation**.

### Primary — `models/physics_mamba.py :: PhysicsMambaDehaze`
```
Hazy I ─► AtmosphericStateEncoder ─► s (FiLM code), A (light, 3), β (density, scalar)
      │                                  │
      └─► PatchEmbed ─► [ BiDirVimBlock ─► FiLM(s) ] × N ─► t-head ─► t(x) ∈ (0,1)
                                                                 │
   Physics (ASM):  J = (I − A) / max(t, ε) + A      (ε = t_eps, guards the division)
   Consistency:    Î = J·t + A·(1 − t)  ≈  I
   Returns dict: {'J', 't', 'A', 'beta'}
```
- **FiLM is zero-initialised** → starts as identity, so training begins from the plain (known-stable) backbone.
- Params ≈ 367 K. Fits far under 12 GB.

### Baseline — `models/mamba_arch.py :: MambaDehaze`
AOD formulation `J = K·(I−1) + 1`, single unified `K` (bounded via `k_max·tanh`). Static, no conditioning. Keep for ablation (`config['model']='aod'`).

### Shared pieces (reused by both)
`PatchEmbedding`, `BiDirectionalVimBlock`, `S4Block` (diagonal SSM via **FFT causal conv**, no Python scan, no Triton), `SpatialReconstruction` (GroupNorm decoder). Physics math in `models/physics.py :: PhysicsReconstruction`.

### Loss — `training/losses.py :: PhysicsDehazeLoss`
`L1 + SSIM + ConvNeXt-contrastive` (on J) `+ w_phys·‖Î−I‖ + w_beta·|β̂−β_gt|` (masked to where β is known) `+ w_tv·TV(t)`. Runs in **fp32** (fp16 SSIM/ConvNeXt/cosine produce NaNs).

---

## 5. ⚠ Stability fixes — DO NOT regress these

The original AOD model diverged at **epoch 3–4 every run** with a **colour explosion**. Root causes, all fixed — if you touch these, keep them:

1. **Bound `K`** in AOD: `K = k_max·tanh(K)` (unbounded per-channel K → channels saturate independently at the clamp = colour explosion).
2. **GroupNorm, never BatchNorm**, in the decoder (BatchNorm running-stats drift with small/accumulated batches → eval-mode inference blows up).
3. **LR = 1e-4, warmup = 2 epochs** (old 2e-4 peak hit the danger zone at epoch 3–4). Removing warmup makes it crash *sooner*, not fix it. Note: grad-clip is weak under Adam (step ≈ lr regardless of clip).
4. **SSM = single FFT causal conv** (was a 1024-step Python loop = launch-bound). Verified numerically identical to the recurrence.
5. **fp32 loss** + **NaN-skip guard** in the train loop.

---

## 6. Data pipeline — `process_data.py`

- **Fresh scene-grouped split** over ALL 8 datasets (group = `dataset:scene_id`), stratified per dataset → **zero cross-split scene leakage** (asserted at runtime). The OLD pipeline random-merged + shuffled → leaked the same scene into train/val/test and inflated metrics. Do not go back to that.
- Datasets: `archive` (~14k synth, β in filename), `SOTS` (synth), `Haze1k` (thin/mod/thick), `NH-HAZE` (real, non-homogeneous), `Dense_Haze`, `I-HAZE`, `O-HAZE`, `BeDDE` (real, grouped by city).
- Writes `data/processed/{split}/{hazy,clear}/{i}.png` + **`meta.csv`** per split (`idx, dataset, scene_id, beta, condition, real_synth`). The loader reads `meta.csv` for β + condition.
- **β labels** come from `archive` filenames (`id_v_beta.png`) and SOTS-outdoor. Used only where present; **jitter augmentation invalidates β** for that sample (it changes effective β/A).
- **The archive `trans/` ground-truth transmission maps are intentionally NOT used** for supervision (user distrusts them). `t` is constrained only by physics-consistency + TV smoothness.

---

## 7. Run & verify

```bash
python -m training.train              # train (physics_mamba by default)
python -m models.physics_mamba        # self-check: shapes/ranges/FiLM-identity
python -m models.mamba_arch           # self-check: FFT-SSM == recurrence
python process_data.py --dry-run      # parsers + leakage-free split, no writes
```
Scripts that import `models.*` / `training.*` **must be run as modules** (`python -m ...`) or from repo root with `PYTHONPATH=.` — running `python models/physics_mamba.py` directly fails (`No module named 'models'`).

Config / hyperparameters live at the top of `training/train.py` (LR, batch, epochs, `k_max`, loss weights `w_phys/w_beta/w_tv`, `t_eps`, `model`).

---

## 8. Gotchas

- **Old checkpoints are incompatible.** The architecture changed twice; a strict `load_state_dict` on any old `outputs/checkpoints/*.pth` will crash on resume. Delete/move them before training fresh. (`outputs/` is git-ignored anyway.)
- **Don't commit `data/`, `outputs/`, `node_modules/`, or `kaggle.json`** — all git-ignored.
- App layer (not the research focus): `backend/` (FastAPI, serves inference via `inference/inference_engine.py`), `frontend/` (React+Vite). `training/tune.py` = Optuna HPO.

---

## 9. Roadmap (next steps, in order)

1. **Ablation harness** — `physics_mamba` vs FiLM-off vs AOD-baseline, reporting **per-condition** PSNR/SSIM by reading `meta.csv`. This is what turns "it trains" into "the dynamic mechanism provably helps" (the thesis proof).
2. Cross-domain / real no-reference eval (BeDDE, NH/O-HAZE), + downstream detection mAP.
3. **Known limitation to address if `A` must be physically meaningful:** `A` is only constrained by physics-consistency (weak signal, ASM `t`↔`A` ambiguity) → add a bright-pixel prior on `A`.

---

## 10. Working style for this repo

The user prefers **minimal, non-over-engineered** changes (ponytail): stdlib/reuse before new code, shortest diff that's correct, no speculative abstractions. Be precise and honest about what's verified vs assumed — never claim a fix works without running a check. Non-trivial logic ships with one runnable self-check (see the `__main__` blocks).
