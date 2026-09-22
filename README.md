# Atmos — Physics-Informed, Condition-Adaptive Vision Mamba Dehazer

A single-image dehazer built on a bidirectional **State Space Model (S4D/Vision-Mamba)**
backbone that estimates the **explicit physics parameters** of the atmospheric scattering
model and reconstructs through them, while **modulating its own features per image** from a
sensed atmospheric state.

| | |
|---|---|
| Parameters | **3,521,877** (`embed_dim=192, n_layers=8, d_state=32`) |
| Best validation | **23.455 dB PSNR / 0.9216 SSIM** (epoch 49 of 50) |
| Training set | 16,418 pairs from 8 datasets, scene-grouped leakage-free split |
| Hardware | RTX 5070 Ti Laptop (Blackwell, sm_120), peak **4.86 GB** VRAM |
| Epoch time | ~3.0 min → ~2.5 h for 50 epochs |

> ⚠️ **Read the headline honestly.** That 23.455 dB is dominated by synthetic data — real
> hazy photographs are only **2.5% of the validation set**. Pooled real-image PSNR is
> **17.49 dB**, a **6.79 dB gap**. See [Results](#7-results).

---

## 1. The claim

Mamba-for-dehazing is already published; the backbone is not the contribution. The thesis
claim is the **dynamic / condition-adaptive** part:

> The network senses a continuous **atmospheric state** (atmospheric light `A`, haze density
> `β`) from the input and **modulates itself (FiLM) per image**, while estimating the explicit
> physics parameters `(t, A)` and reconstructing through the atmospheric scattering model.
> "Dynamic" means behaviour varies *continuously* with the sensed environment — not a switch
> over discrete classes.

---

## 2. Architecture

```
Hazy I ─► AtmosphericStateEncoder ─► s (FiLM code), A (3,), β (scalar)
      │                                   │
      └─► PatchEmbed ─► [ BiDirVimBlock ─► FiLM(s) ] × 8 ─► t-head ─► t(x) ∈ (0,1)
                                                                   │
   Physics (ASM):  J = (I − A) / max(t, t_ε) + A
   Consistency:    Î = J·t + A·(1 − t)  ≈  I
   Returns:        {'J', 't', 'A', 'beta'}
```

`models/physics_mamba.py :: PhysicsMambaDehaze`

### 2.1 Patch embedding

A 256×256 image becomes a sequence of 8×8 patches via a strided convolution:

```
L = (256 / 8)² = 1024 tokens,  D = 192
x = Conv2d(3, D, kernel=8, stride=8)(I).flatten(2).transpose(1,2) + pos_embed
```

### 2.2 The S4D state space block

Each block implements a discretised continuous-time linear system, diagonal in the state
(S4D / DSS style), per channel:

```
continuous:   dx/dt = A·x + B·u ,   y = C·x + D·u
ZOH discrete: Ā = exp(A·Δ) ,        B̄ ≈ Δ·B
recurrence:   x_k = Ā ⊙ x_{k-1} + B̄·u_k ,   y_k = ⟨C, x_k⟩ + D·u_k
```

with `A = −exp(A_log) < 0` (guaranteeing stability), `Δ = exp(log_dt)` clamped to
`[1e-3, 0.1]`, and `A_log` initialised HiPPO-lite as `log(1..N)` to spread the decay rates.

Because the system is **LTI** (`Ā` does not depend on the input), the recurrence is exactly a
causal convolution, so the whole scan collapses to one kernel:

```
K[d, l] = Σ_n C[d,n] · B̄[d,n] · Ā[d,n]^l
y = irfft( rfft(u, 2L) · rfft(K, 2L) )[:L] + D ⊙ u
```

The 2L zero-pad turns the circular FFT product into a *linear* (causal) convolution. This
replaces a 1024-step Python loop that was launch-bound. Numerically verified against the
explicit recurrence — **max error 1.49e-08** (`python -m models.mamba_arch`).

Each `BiDirectionalVimBlock` runs two such blocks (forward and reversed sequence), then fuses:
`y = LayerNorm(Linear([y_fwd ; y_bwd]))`.

### 2.3 FiLM conditioning — the "dynamic" mechanism

After every block, tokens are modulated by the sensed atmospheric state `s`:

```
x ← x · (1 + γ(s)) + β(s)        γ, β : Linear(state_dim → D), ZERO-initialised
```

Zero-init makes FiLM an exact identity at step 0, so training begins from the plain
(known-stable) backbone and learns to modulate from there. After training, `‖γ‖` is measurably
non-zero, i.e. the mechanism is actually used rather than collapsing back to identity.

---

## 3. The physics

The **atmospheric scattering model (ASM)**:

```
I(x) = J(x)·t(x) + A·(1 − t(x))          t(x) = exp(−β·d(x))
```

* `I` — observed hazy image `J` — haze-free radiance
* `t` — transmission (fraction of light surviving to the camera), depth-dependent
* `A` — atmospheric light (airlight) `β` — scattering coefficient, `d` — scene depth

**Recovery** (`models/physics.py :: PhysicsReconstruction`):

```
J = (I − A) / max(t, t_ε) + A ,  clamped to [0,1] ,  t_ε = 0.1
```

`t_ε` guards the division; it also caps dehazing strength at `1/t_ε = 10×`.

**Consistency** — re-synthesising the input from the predicted parameters:

```
Î = J·t + A·(1 − t)          loss pulls Î → I
```

---

## 4. Loss

`training/losses.py :: PhysicsDehazeLoss`

```
L = w_l1 ·‖J − J*‖₁
  + w_ssim·(1 − SSIM(J, J*))
  + w_cr  · CR(J, J*, I)
  + w_phys·‖J·t + A(1−t) − I‖₁          physics consistency
  + w_A   ·|mean_c(Â) − A*| ⊙ m_A        airlight supervision
  + w_β   ·|β̂ − β*| ⊙ m_β                density supervision
  + w_tv  · TV(t)                        transmission smoothness
```

| weight | value | purpose |
|---|---|---|
| `w_l1` | 1.0 | pixel fidelity |
| `w_ssim` | 0.5 | structural fidelity |
| `w_cr` | 0.1 | ConvNeXt contrastive regulariser |
| `w_phys` | 0.2 | ASM self-consistency |
| `w_A` | 0.1 | airlight, where labelled |
| `w_beta` | 0.1 | density, where labelled |
| `w_tv` | 0.01 | `t` smoothness (`t` is otherwise unsupervised) |

**Contrastive regulariser** (AECR-style, ConvNeXt-Tiny features, frozen), over 3 scales:

```
CR = (1/3) Σ_k [ d(f_k(J), f_k(J*)) + relu(margin − d(f_k(J), f_k(I))) ]
d(a,b) = 1 − cosine_similarity(a, b)      margin = 1.0
```

It pulls the output toward the clear image and pushes it away from the hazy input, which
suppresses the "do nothing" solution.

**Total variation** on the transmission map:

```
TV(t) = mean|t[:, :, 1:, :] − t[:, :, :-1, :]| + mean|t[:, :, :, 1:] − t[:, :, :, :-1]|
```

**The entire loss runs in fp32** (`@torch.amp.autocast('cuda', enabled=False)`). SSIM
convolutions, ConvNeXt features and cosine similarities all under/overflow in fp16 and
produce NaNs. The masks `m_A`, `m_β` are zeroed whenever colour/density jitter was applied,
because jitter changes the *effective* `A` and `β` and invalidates the stored label.

---

## 5. Data

`process_data.py` — 8 datasets, **16,418 pairs**.

| dataset | kind | pairs | condition |
|---|---|---|---|
| archive | synthetic | 13,990 | homogeneous, `A` in filename |
| Haze1k | synthetic | 1,035 | thin / moderate / thick |
| SOTS | synthetic | 1,000 | indoor + outdoor, `A` and `β` in filename |
| BeDDE | real | 208 | urban fog, 23 cities, fixed cameras |
| NH-HAZE | real | 55 | non-homogeneous |
| Dense_Haze | real | 55 | extreme density |
| O-HAZE | real | 45 | outdoor |
| I-HAZE | real | 30 | indoor |

### 5.1 Leakage-free splitting

Splits are assigned **by scene group** (`dataset:scene_id`), stratified per dataset, so no
scene's variants can straddle a split. An assertion enforces it at runtime.

```
train 13,137   |   val 1,645   |   test 1,636
```

A naive merge-then-shuffle split leaks the same scene into train *and* test — with `archive`
supplying 10 hazy variants per clear image, that inflates metrics badly.

### 5.2 Aspect ratio

Source aspect ratios span **0.80–1.53**. The pipeline resizes the **shorter side** to 256 and
crops (random for train, centre for eval) rather than squashing everything to 256², which
distorted geometry by a different amount per dataset.

### 5.3 Class balance

Real images are only **311 of 13,137 training pairs (2.4%)**. A `WeightedRandomSampler`
raises them to **20% of every epoch** (see [§10](#10-calculations) for the weight derivation).

---

## 6. Label semantics — `A` vs `β`

**The single most consequential bug found in this project.**

`archive` filenames look like `1_10_0.98796.png`. The pipeline read the third token as `β`
and trained the density head on it — **93% of all β supervision**. It is not `β`; it is `A`.

**Proof 1 — correlation, holding the clear image constant.** Across 250 images / 25 scenes,
comparing the filename value against image statistics *within* each scene:

| within-scene correlation of the filename value against… | |
|---|---|
| airlight estimate (99.9th percentile of `I`) | **+0.904** |
| brightness lift (`mean I − mean J`) | +0.708 |
| contrast destruction (`1 − σI/σJ`) | **−0.087** |

A scattering coefficient destroys contrast. This value does *nothing* to contrast (−0.087)
while tracking airlight at +0.904.

**Proof 2 — independent least-squares recovery.** For a paired dataset the ASM is heavily
over-determined: 3 channel equations per pixel, 1 unknown per pixel (`t`) plus 3 shared
(`A`). Fitting `A` per image and solving `t` in closed form gives a recovered `A` that
correlates **+0.997** with the filename label, mean |difference| **0.003**.

Two independent methods, same answer. The fix separates the columns:

```
archive       →  A = toks[2]                      β = (none)
SOTS-outdoor  →  A = toks[1],  β = toks[-1]       (the only genuine β in the corpus)
```

Post-fix label counts: **A = 14,490**, **β = 500**. Previously `β` claimed 14,490 labels, of
which 13,990 were actually airlight.

---

## 7. Results

### 7.1 Per-dataset validation PSNR

| dataset | kind | n | run 1 | run 2 | Δ |
|---|---|---|---|---|---|
| archive | synth | 1400 | 24.43 | 24.46 | +0.03 |
| SOTS | synth | 101 | 25.56 | 25.27 | −0.29 |
| Haze1k | synth | 103 | 21.51 | 20.99 | −0.52 |
| BeDDE | real | 24 | 17.84 | 18.20 | +0.36 |
| NH-HAZE | real | 5 | 14.50 | 16.06 | **+1.56** |
| Dense_Haze | real | 5 | 12.30 | 14.06 | **+1.76** |
| I-HAZE | real | 3 | 13.68 | 15.67 | **+1.99** |
| O-HAZE | real | 4 | 16.38 | 20.69 | **+4.31** |
| **POOLED** | **synth** | 1604 | 24.31 | 24.29 | −0.02 |
| **POOLED** | **real** | 41 | 16.31 | **17.49** | **+1.18** |
| | **gap** | | 8.00 | **6.79** | **−1.21** |

Run 1 = squashed images, no rebalancing, no `A` supervision.
Run 2 = aspect-preserved, 20% real, `w_A = 0.1`.

> ⚠️ **These two runs are not a clean A/B.** Run 2's validation *images* changed
> (aspect-correct centre crops vs squashed squares), so part of every delta is the crop. The
> real-set gains (+1.2 to +4.3 dB) are far larger than crop noise plausibly explains and are
> in the predicted direction, but they are not a controlled measurement. The aggregate
> 23.455 vs 23.233 dB is **meaningless** as a comparison for the same reason.

### 7.2 Is the atmospheric state *sensed* or *memorised*?

| | predicted `A` | per-image std | error vs DCP |
|---|---|---|---|
| run 1, synth | [0.948 0.955 0.964] | **0.03** | 0.080 |
| run 2, synth | [0.867 0.871 0.873] | **0.069** | 0.047 |
| run 2, real | [0.669 0.694 0.732] | 0.112–0.163 | 0.149 |

The true `A` labels have a spread of ≈0.09. Run 1's encoder emitted a near-constant 0.95 — a
third of the variance it should have — i.e. it had memorised the training mean rather than
sensing anything. After supervision the std more than doubles to 0.069 and the error against
an independent dark-channel estimate nearly halves. **This is the condition-adaptive claim
becoming measurable.**

### 7.3 PSNR against atmospheric light

| `A` range (quantile bins) | n | PSNR |
|---|---|---|
| [0.700, 0.776] | 363 | 25.49 |
| [0.776, 0.850] | 365 | 24.94 |
| [0.850, 0.924] | 371 | 24.27 |
| [0.924, 1.000] | 363 | 23.33 |

Monotonic degradation with airlight — brighter, denser haze is harder, as expected.
Binning by `β` is not possible: only 51 validation images carry a genuine `β`.

### 7.4 The transmission map — the dominant remaining limitation

True `t` derived from `(I, J, A)` versus what the model predicts:

| dataset | TRUE `t` mean | TRUE 1st-pct | PRED `t` mean | PRED min |
|---|---|---|---|---|
| archive (synth) | 0.542 | **0.345** | 0.577 | 0.379 |
| BeDDE | 0.764 | 0.240 | 0.844 | 0.496 |
| NH-HAZE | 0.551 | **0.023** | 0.646 | 0.301 |
| Dense_Haze | 0.327 | **0.044** | 0.387 | 0.184 |
| O-HAZE | 0.485 | 0.163 | 0.571 | 0.275 |

The *mean* is close. The **lower tail is collapsed**: in the densest regions of real images
true transmission reaches 0.02–0.16, but the model floors at 0.18–0.50 — on NH-HAZE, 13× too
high exactly where haze is worst.

The cause is in the same table: **`archive`'s own 1st percentile is 0.345.** The synthetic set
that is 85% of training *contains no severe haze at all*, so the model never saw `t < 0.35`
and cannot extrapolate. It is not mis-trained; it was never shown the regime.

This one fact explains the observed failure modes — distant haze surviving, near regions
restored better than far, and weak colour recovery (chroma scales with `1/t` exactly as
contrast does):

```
observed mean t on real images = 0.700  →  amplification 1/0.700 = 1.43×
ceiling allowed by t_ε = 0.1            →  amplification 10×
required for NH-HAZE dense regions      →  1/0.023 ≈ 43×
```

---

## 8. Engineering log

Every issue actually hit, its root cause, and the fix.

### Environment

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| 1 | `Authentication required to call the Kaggle API` | Kaggle CLI ≥2.x ignores `kaggle.json` and reads `~/.kaggle/access_token` via `read_text().strip()`, which does **not** strip a UTF-8 BOM. The BOM corrupted the token. | Rewrote both credential files BOM-free |
| 2 | All three dataset downloads returned HTTP 403 | Every dataset ref in `download_datasets.py` was dead | Corrected two refs; dropped `rshaze` (out of scope, ref does not exist) |
| 3 | `FileNotFoundError: kaggle` | `kaggle.exe` lives in the user `Scripts` dir, not on `PATH` | `subprocess([sys.executable, "-m", "kaggle", …])` |
| 4 | `ModuleNotFoundError: torchvision` | Only `torch` was installed from the cu128 index | Installed `torchvision==0.26.0+cu128` (dry-run first, to confirm torch untouched) |
| 5 | `ModuleNotFoundError: cv2` | `opencv-python` missing | Installed |

### Training crashes

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| 6 | `load_state_dict` crash on startup | `train.py` auto-resumes from `mamba_last.pth`; the file on disk was a different architecture — 61 missing / 35 unexpected keys | Archive stale checkpoints before a fresh run |
| 7 | `UnicodeEncodeError: '★'` killed the run at epoch 1 | `print("★ New best …")` with stdout **redirected**: Windows defaults redirected stdout to cp1252, which cannot encode `★`. Worked in the console, died in `> train.log`. | `sys.stdout.reconfigure(encoding='utf-8')` at the entry point — fixes every non-ASCII print, not just this one |
| 8 | `UnpicklingError: Weights only load failed` on resume | torch ≥2.6 flipped `torch.load`'s `weights_only` default to `True`. `history['lr']` contains a `np.float64` because `WarmupCosineScheduler` uses `np.cos` — so the checkpoint became unloadable at exactly the **first post-warmup epoch** | `weights_only=False` at **both** call sites (`trainer.py`, `inference_engine.py`) — the second would have broken the web backend identically |
| 9 | `RuntimeError: Couldn't open shared file mapping … error code: 1455` at epoch 29 | Windows `ERROR_COMMITMENT_LIMIT`. `persistent_workers=True` on **both** DataLoaders kept 6 train + 6 val workers resident with `pin_memory=True`; commit free had fallen to 9.2 GB of 47.1 GB | Val loader → 2 transient unpinned workers (it runs 118 batches once per epoch). Resident workers 12 → 6 |
| 10 | Log flooded with `RuntimeError: main thread is not in main loop` | matplotlib defaults to **TkAgg** on Windows; `plot_history()` creates Tk widgets each epoch whose `__del__` fires off-thread. Harmless (`Exception ignored in`) but buries the output | `matplotlib.use("Agg")` — the correct backend when only writing PNGs |
| 11 | `FutureWarning` × 18 per epoch | `torch.cuda.amp.autocast/GradScaler` deprecated; repeated because Windows respawns DataLoader workers each epoch, re-importing the modules | Migrated 7 sites to `torch.amp.*`. **The `enabled=False` decorators were kept** — they are the fp32 guard, not incidental |

### Application layer

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| 12 | Backend could not load the trained model | `inference_engine.py` hardcoded `MambaDehaze(embed_dim=64, n_layers=4)` — the *AOD baseline* at *stale dims*. Shapes mismatch, and the return types differ (dict vs tensor) | Rebuild the architecture from the checkpoint's own stored `config`, so any capacity or model type loads without edits |
| 13 | `/status` reported `"NVIDIA RTX 3050"` | Hardcoded string | Report the real device |
| 14 | Output blurry, visible 8-px blocks, 4K images returned soft | The ASM was evaluated at 256×256 and the **result** upscaled. Measured Laplacian variance **2.4** versus **44.8** for the input — the pipeline was destroying ~95% of the detail | Evaluate the ASM at **native resolution**: the net still sees 256² to estimate `t` and `A`, but `t` is bilinearly upsampled, guided-filter refined, and `J = (I−A)/t + A` applied to every original pixel. Sharpness **2.4 → 126.8** |

Fix 14 works *because* of the physics: `t` is depth-like and genuinely smooth, so upsampling
it costs nothing, while dividing by a smooth `t` preserves `I`'s high frequencies. The block
artefacts came from `SpatialReconstruction` upsampling a 32×32 token grid 8× with two
`ConvTranspose2d` layers where **kernel == stride**, so every output block descends from
exactly one input cell — hard edges by construction.

### Ideas measured and **rejected**

| Idea | Result | Verdict |
|---|---|---|
| Run the network at 512² via interpolated `pos_embed` | +0.12 dB on NH-HAZE for 4× compute | Rejected — inside noise |
| "Bigger model will be much slower" (predicted 2.5–4×) | Measured **1.0×** — 1.4M params ran *faster* than 367K | Estimate was wrong; the GPU is launch-bound. Capacity was raised 9.6× instead of 3.9× |
| Neutralise `A` to grey to fix the warm cast | Cast got **worse** (0.02266 vs 0.01365) | Rejected — the cast is partly correct physics |

### Why the output is warm — and why that is mostly correct

Mean colour shift from hazy image to its clear **target** (what the model is taught to produce):

| source | dR | dG | dB | warm bias (dR−dB) |
|---|---|---|---|---|
| archive | −0.144 | −0.188 | −0.180 | +0.036 |
| NH-HAZE | −0.078 | −0.114 | −0.194 | **+0.117** |
| BeDDE | −0.013 | −0.002 | +0.012 | −0.025 |

Haze scatters blue most, so removing it *should* warm the image — the ground truth itself is
warmer than its input. The excess warmth came from `A_B` being over-estimated, which is what
`w_A` supervision addresses. Blending `A` 50/50 with a dark-channel estimate cut measured cast
55% (0.01365 → 0.00608), but that replaces learned physics with a heuristic and was not
adopted.

---

## 9. Stability — do not regress these

The original AOD model diverged at **epoch 3–4 every run** with a colour explosion.

1. **Bound `K`** in the AOD baseline: `K = k_max·tanh(K)`. Unbounded per-channel `K` lets
   channels saturate independently at the clamp — that *is* the colour explosion.
2. **GroupNorm, never BatchNorm**, in the decoder. BatchNorm running statistics drift with
   small/accumulated batches and corrupt eval-mode inference.
3. **`lr = 1e-4` with 2-epoch warmup.** The old 2e-4 peak hit the danger zone at epoch 3–4.
   Removing warmup makes it crash *sooner*, not later. Note that gradient clipping is weak
   under Adam — the step size is ≈ lr regardless of the clip.
4. **SSM as a single FFT convolution**, verified identical to the recurrence (1.49e-08).
5. **fp32 loss** plus a non-finite-step skip guard in the training loop.

---

## 10. Calculations

**Oversampling weight.** For a target real fraction `f` with `n_real` real and `n_synth`
synthetic samples, each real sample gets weight `w` (synthetic get 1):

```
f = n_real·w / (n_real·w + n_synth)
⇒ w = (f / (1 − f)) · (n_synth / n_real)
    = (0.20 / 0.80) · (12,826 / 311)
    = 0.25 × 41.24
    = 10.3×
```

Verified empirically: one drawn epoch contained **20.26%** real images (target 20%).

**Dehazing amplification.** From `J = (I − A)/max(t, t_ε) + A`, the gain applied to `(I − A)`
is `1/t`:

```
ceiling from t_ε = 0.1        →  10.0×
observed mean t = 0.700 (real) →   1.43×
needed for NH-HAZE dense (t=0.023) → 43×
```

`t_ε` is *not* currently the binding constraint — the model's own floor is.

**Transmission derived from a pair.** Rearranging the ASM per pixel:

```
t = (I − A) / (J − A)          masked where |J − A| > 0.10 (unstable otherwise)
```

**Joint `(A, t)` recovery.** With `A` shared across the image, fit 3 parameters by
Nelder-Mead minimising `‖J·t(A) + A(1−t(A)) − I‖₁`, with `t(A)` in closed form above.
Reconstruction error: archive 0.0016, Dense_Haze 0.0142, O-HAZE 0.0136, NH-HAZE 0.0354.

**Metrics.**

```
PSNR   = 10 · log₁₀(1 / MSE)                       on [0,1] images
SSIM   = windowed, 11×11 Gaussian σ=1.5, C₁=0.01², C₂=0.03²
cast   = mean_c |d_c − mean(d)| ,  d = mean(J − I) per channel
DCP A  = mean of I over the pixels whose dark channel is in the top 0.1%
         dark channel = erode(min_c I, 15×15)
```

---

## 11. Setup

### ⚠️ Blackwell (sm_120) requires cu128

The default PyPI torch wheel (cu124) fails at runtime with *"no kernel image is available for
execution on the device"*.

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # must print True
```

### Data

```bash
python download_datasets.py          # needs ~/.kaggle/access_token (BOM-free!)
python process_data.py --dry-run     # parsers + leakage check, no writes
python process_data.py               # ~10 min, writes data/processed/
```

### Train / evaluate / serve

```bash
python -m training.train                                   # 50 epochs, ~2.5 h
PYTHONPATH=. python scripts/eval_report.py --split val     # per-dataset + A diagnostics
PYTHONPATH=. python scripts/probe_epoch_time.py            # size a run before committing hours

python -m uvicorn backend.main:app --host 0.0.0.0 --port 5000
cd frontend && npm run dev                                 # http://localhost:3000
```

### Self-checks

```bash
python -m models.physics_mamba        # shapes, ranges, FiLM identity, FiLM-off ablation arm
python -m models.mamba_arch           # FFT-SSM == recurrence
python -m inference.inference_engine  # native-res inference, resolution round-trip
```

Modules must be run with `-m` (or `PYTHONPATH=.`) from the repo root.

---

## 12. Limitations and roadmap

**Known limitations**

* Real-world performance trails synthetic by **6.79 dB**; real images are 2.4% of training.
* The transmission floor (§7.4) prevents removal of severe/distant haze.
* **1.8% of output pixels clip to pure black** — `(I−A)/t + A` going negative, then clamped.
* The ASM is a per-pixel rescale: it **cannot restore high-frequency detail** that scattering
  physically destroyed.
* `t` is estimated from a squashed 256² view while the network is now trained on
  aspect-correct crops — a mild train/test mismatch, not yet measured.

**Roadmap**

1. **ASM-fitted `(A, t)` supervision.** Precompute per-pixel `t` and per-channel `A` targets
   for all 16,418 pairs by the least-squares fit of §10. Gives `t` targets reaching 0.019 with
   spatial std 0.324 (versus the model's current 0.12) and supplies `A` for the 311 real
   training images that currently have none. Targets the transmission floor directly.
2. **`t_ε` 0.1 → 0.05** and **edge-aware TV**, weighting `TV(t)` by `exp(−|∇I|)` so `t` may
   jump at depth discontinuities. The current uniform TV actively suppresses depth structure.
3. **Clipped-pixel penalty** for the 1.8% crushed to black.
4. **`patch_size` 8 → 4** — a 64×64 transmission grid for sharper depth edges (4× SSM
   sequence length; cost to be probed, not assumed).
5. **Learned detail branch** for high-frequency reconstruction. This synthesises detail not
   present in the input, so it trades against the low-hallucination behaviour the current
   model has. Mitigations: bound the residual magnitude and keep the contrastive term, which
   penalises departure from the ground truth.

**Also pending:** the §9 ablation — `physics_mamba` vs FiLM-off vs the AOD baseline at matched
capacity. The FiLM-off arm is implemented (`PhysicsMambaDehaze(use_film=False)`) and verified
to share a `state_dict` layout, be identical at init, and diverge once FiLM is active.

---

## 13. Repository layout

```
models/
  physics_mamba.py     PhysicsMambaDehaze — the thesis model (FiLM + explicit t, A, β)
  mamba_arch.py        S4D block, bidirectional Vim block, patch embed, decoder, AOD baseline
  physics.py           ASM reconstruction
training/
  train.py             config, dataset, WeightedRandomSampler, entry point
  trainer.py           loop, AMP, warmup+cosine schedule, checkpointing
  losses.py            SSIM, ConvNeXt contrastive, PhysicsDehazeLoss
  augmentations.py     domain randomisation, joint random crop
scripts/
  eval_report.py       per-dataset table, A-collapse check, A-binned PSNR
  probe_epoch_time.py  measure epoch time and VRAM before committing to a run
inference/             native-resolution ASM inference engine
backend/ frontend/     FastAPI + React demo app
process_data.py        8-dataset parser, leakage-free scene-grouped split, meta.csv
```

`data/` and `outputs/` are git-ignored and regenerated per machine — except
`outputs/checkpoints/mamba_best.pth`, which ships via Git LFS.
