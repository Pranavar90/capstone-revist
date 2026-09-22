"""
Physics-Explicit, Condition-Adaptive Vision Mamba Dehazer
=========================================================
The dynamic/adaptive model (vs the static AOD baseline in mamba_arch.py):

  Hazy I ──► AtmosphericStateEncoder ──► s (FiLM code), A (light), beta (density)
        │                                    │
        └─► PatchEmbed ─► [VimBlock ─► FiLM(s)] xN ─► t-head ─► t(x)
                                                             │
        Physics (ASM):  J = (I - A) / max(t, eps) + A   (disentangled t, A)
        Consistency:    Î = J*t + A*(1 - t)  ≈  I

"Dynamic" = the estimated atmospheric state s modulates every block (FiLM), so
the network computes a *different function per environment*, continuously.

Reuses the Mamba backbone / physics from the existing modules — no new backbone.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.mamba_arch import PatchEmbedding, BiDirectionalVimBlock, SpatialReconstruction
from models.physics import PhysicsReconstruction


class AtmosphericStateEncoder(nn.Module):
    """Small CNN over the hazy image → global atmospheric state.

    Returns:
        s:    (B, state_dim)  FiLM conditioning code
        A:    (B, 3)          atmospheric light in [0, 1]
        beta: (B,)            haze density (> 0)
    """
    def __init__(self, state_dim=64):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.GroupNorm(8, 32), nn.GELU(),      # /2
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.GELU(),     # /4
            nn.Conv2d(64, state_dim, 3, stride=2, padding=1),
            nn.GroupNorm(min(8, state_dim), state_dim), nn.GELU(),                          # /8
            nn.AdaptiveAvgPool2d(1),
        )
        self.to_A = nn.Sequential(nn.Linear(state_dim, 3), nn.Sigmoid())     # light ∈ [0,1]
        self.to_beta = nn.Sequential(nn.Linear(state_dim, 1), nn.Softplus())  # density > 0

    def forward(self, x):
        s = self.backbone(x).flatten(1)          # (B, state_dim)
        A = self.to_A(s)                         # (B, 3)
        beta = self.to_beta(s).squeeze(-1)       # (B,)
        return s, A, beta


class FiLM(nn.Module):
    """Feature-wise linear modulation of (B, L, D) tokens from a state code.

    Zero-initialised → starts as identity, so training begins from the plain
    (working) backbone and learns to modulate from there.
    """
    def __init__(self, state_dim, d_model):
        super().__init__()
        self.gamma = nn.Linear(state_dim, d_model)
        self.beta = nn.Linear(state_dim, d_model)
        nn.init.zeros_(self.gamma.weight); nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight); nn.init.zeros_(self.beta.bias)

    def forward(self, x, s):
        g = self.gamma(s).unsqueeze(1)   # (B, 1, D)
        b = self.beta(s).unsqueeze(1)
        return x * (1.0 + g) + b


class PhysicsMambaDehaze(nn.Module):
    def __init__(self, img_size=256, patch_size=8, embed_dim=64, d_state=16,
                 n_layers=4, dropout=0.1, state_dim=64, t_eps=0.1, use_film=True):
        super().__init__()
        # use_film=False is the §9 ablation arm: the AtmosphericStateEncoder still runs and
        # A still drives the ASM, so what is removed is ONLY the per-block modulation —
        # which is precisely the "dynamic" claim. The FiLM modules are still constructed so
        # the state_dict stays identical across arms (they simply remain at their zero init).
        self.use_film = use_film
        self.state_enc = AtmosphericStateEncoder(state_dim)
        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, embed_dim)
        self.vim_blocks = nn.ModuleList(
            [BiDirectionalVimBlock(embed_dim, d_state, dropout) for _ in range(n_layers)]
        )
        self.films = nn.ModuleList([FiLM(state_dim, embed_dim) for _ in range(n_layers)])
        # Transmission map head (1 channel), reuse the CNN refinement decoder.
        self.t_head = SpatialReconstruction(embed_dim, img_size, patch_size, out_channels=1)
        self.physics = PhysicsReconstruction(epsilon=t_eps)

    def forward(self, hazy):
        s, A, beta = self.state_enc(hazy)

        x = self.patch_embed(hazy)                       # (B, L, D)
        for blk, film in zip(self.vim_blocks, self.films):
            x = blk(x)
            if self.use_film:
                x = film(x, s)                           # FiLM-modulated per block

        t = torch.sigmoid(self.t_head(x))                # (B, 1, H, W) ∈ (0,1)
        if t.shape[2:] != hazy.shape[2:]:
            t = F.interpolate(t, size=hazy.shape[2:], mode='bilinear', align_corners=False)

        J = self.physics(hazy, t, A)                     # (I-A)/max(t,eps)+A, clamped [0,1]
        return {'J': J, 't': t, 'A': A, 'beta': beta}


# =============================================================================
# Self-check
# =============================================================================
if __name__ == "__main__":
    torch.manual_seed(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = PhysicsMambaDehaze(img_size=256, embed_dim=64, n_layers=4).to(dev).eval()

    params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"PhysicsMambaDehaze params: {params:,}")

    x = torch.rand(2, 3, 256, 256, device=dev)
    with torch.no_grad():
        out = m(x)

    assert out['J'].shape == x.shape, out['J'].shape
    assert out['t'].shape == (2, 1, 256, 256), out['t'].shape
    assert out['A'].shape == (2, 3), out['A'].shape
    assert out['beta'].shape == (2,), out['beta'].shape
    assert out['J'].min() >= 0 and out['J'].max() <= 1, "J out of [0,1]"
    assert out['t'].min() > 0 and out['t'].max() < 1, "t out of (0,1)"
    # FiLM is zero-init → identity: gamma/beta params are all zero at start.
    assert all(float(f.gamma.weight.abs().sum()) == 0 for f in m.films), "FiLM not identity at init"
    print(f"[selfcheck] shapes/ranges OK | J[{out['J'].min():.3f},{out['J'].max():.3f}] "
          f"t[{out['t'].min():.3f},{out['t'].max():.3f}] A~{out['A'].mean():.3f} beta~{out['beta'].mean():.3f}")

    # Ablation arm: identical state_dict, but FiLM must have no effect on the output.
    off = PhysicsMambaDehaze(img_size=256, embed_dim=64, n_layers=4, use_film=False).to(dev).eval()
    off.load_state_dict(m.state_dict())
    assert set(off.state_dict()) == set(m.state_dict()), "arms must share a state_dict layout"
    with torch.no_grad():
        out_off = off(x)
    # At init FiLM is identity, so both arms must agree exactly.
    assert torch.allclose(out['J'], out_off['J'], atol=1e-6), "FiLM is not identity at init"
    # Give FiLM a non-zero gamma: now the arms MUST diverge, or use_film is not wired up.
    with torch.no_grad():
        for f in m.films:
            f.gamma.bias.fill_(0.5)
    with torch.no_grad():
        out_on = m(x)
    assert not torch.allclose(out_on['J'], out_off['J'], atol=1e-6), "use_film=False changed nothing"
    print("[selfcheck] FiLM-off arm OK: same state_dict, identity at init, diverges once FiLM is active")
