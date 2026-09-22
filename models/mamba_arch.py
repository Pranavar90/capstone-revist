"""
Phase B + C: End-to-End Vision Mamba (Vim) Dehazer
==================================================
World-first: Physics-Guided State Space Model for Single Image Dehazing.

Architecture:
  Input Image  -->  PatchEmbedding  -->  Bi-directional SSM Blocks
       -->  Spatial Unflatten  -->  CNN Refinement  -->  K(x) prediction
       -->  AOD Physics:  J(x) = K(x) * I(x) - K(x) + 1

Key design choices:
  - Pure PyTorch SSM (no mamba-ssm CUDA kernels) for Windows compatibility
  - Bi-directional scanning to preserve 2D spatial awareness
  - End-to-End AOD formulation: NO division, NO separate A/t(x) prediction
  - Target VRAM: <3.5 GB on RTX 3050
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# =============================================================================
# Pure-PyTorch State Space Model Block (S4-inspired, no CUDA kernels)
# =============================================================================

class S4Block(nn.Module):
    """
    Simplified State Space Sequence Model (S4-inspired).
    Implements a discretised continuous-time SSM:
        x_{k+1} = A_bar * x_k + B_bar * u_k
        y_k     = C * x_k + D * u_k

    Uses diagonal state matrix for efficiency (like S4D / DSS).
    Purely PyTorch — no custom CUDA kernels required.
    """
    def __init__(self, d_model, d_state=16, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # Diagonal SSM parameters, per-channel SISO (S4D-style).
        # Log-space A for numerical stability; B/C are plain diagonal params.
        self.A_log = nn.Parameter(torch.empty(d_model, d_state))
        self.B = nn.Parameter(torch.randn(d_model, d_state) * 0.1)
        self.C = nn.Parameter(torch.randn(d_model, d_state) * 0.1)
        self.D = nn.Parameter(torch.ones(d_model))  # skip connection

        # Learnable ZOH step, init log-uniform in [1e-3, 0.1]
        self.log_dt = nn.Parameter(
            torch.rand(d_model) * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3)
        )

        # Input projection + gating (SiLU like Mamba)
        self.in_proj = nn.Linear(d_model, d_model * 2)
        self.out_proj = nn.Linear(d_model, d_model)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self._init_A()

    def _init_A(self):
        """HiPPO-lite: spread eigenvalues over 1..N (stable, diverse decay)."""
        with torch.no_grad():
            A_mag = torch.arange(1, self.d_state + 1, dtype=torch.float32)
            self.A_log.copy_(A_mag.log().unsqueeze(0).expand(self.d_model, -1).clone())

    def _kernel(self, L, device):
        """
        SSM convolution kernel K[d, l] = Σ_n C[d,n] Bbar[d,n] Abar[d,n]^l.
        The recurrence is LTI (Abar input-independent) so it equals a causal
        conv with this kernel — no scan needed.
        """
        dt = torch.exp(self.log_dt).clamp(min=1e-3, max=0.1)      # (D,)
        A = -torch.exp(self.A_log)                                # (D, N), <0 → stable
        log_Abar = A * dt.unsqueeze(-1)                           # (D, N), <=0
        Bbar = self.B * dt.unsqueeze(-1)                          # (D, N), ZOH ≈ dt*B
        l = torch.arange(L, device=device, dtype=log_Abar.dtype)  # (L,)
        powers = torch.exp(log_Abar.unsqueeze(-1) * l)           # (D, N, L), decays → 0
        K = torch.einsum('dn,dnl->dl', self.C * Bbar, powers)    # (D, L)
        return K

    def _ssm(self, u):
        """
        Apply the SSM as a single causal FFT conv (compute-bound, no Python loop).
        Args:
            u: (B, L, D)
        Returns:
            y: (B, L, D)
        """
        B_batch, L, D = u.shape
        K = self._kernel(L, u.device)               # (D, L)
        u_t = u.transpose(1, 2)                      # (B, D, L)

        # Zero-pad to 2L so the circular FFT product yields a *linear* (causal) conv.
        n = 2 * L
        Uf = torch.fft.rfft(u_t, n=n, dim=-1)
        Kf = torch.fft.rfft(K, n=n, dim=-1).unsqueeze(0)   # (1, D, F)
        y = torch.fft.irfft(Uf * Kf, n=n, dim=-1)[..., :L]  # (B, D, L)

        y = y + u_t * self.D.view(1, D, 1)           # skip connection
        return y.transpose(1, 2)                     # (B, L, D)

    @torch.amp.autocast('cuda', enabled=False)
    def forward(self, x):
        """
        Args:
            x: (B, L, D)
        Returns:
            (B, L, D)
        """
        # Force EVERYTHING inside this block to FP32 to prevent Autocast NaNs
        x = x.to(torch.float32)
        residual = x
        x = self.norm(x)

        # Input projection + gating
        xz = self.in_proj(x)
        x_proj, z = xz.chunk(2, dim=-1)
        z = F.silu(z)  # gate

        # SSM (single FFT conv, no Python scan)
        y = self._ssm(x_proj)
        y = y * z  # gated output

        y = self.out_proj(y)
        y = self.dropout(y)

        return y + residual


# =============================================================================
# Bi-Directional Vision Mamba Block
# =============================================================================

class BiDirectionalVimBlock(nn.Module):
    """
    Processes the 1D patch sequence in both Forward and Backward directions,
    then fuses the results. This preserves spatial context that would be
    lost with a single-direction sweep.
    """
    def __init__(self, d_model, d_state=16, dropout=0.1):
        super().__init__()
        self.forward_ssm = S4Block(d_model, d_state, dropout)
        self.backward_ssm = S4Block(d_model, d_state, dropout)
        self.fusion = nn.Linear(d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        Args:
            x: (B, L, D)
        Returns:
            (B, L, D)
        """
        # Forward sweep
        y_fwd = self.forward_ssm(x)

        # Backward sweep  (reverse → scan → reverse back)
        x_rev = torch.flip(x, dims=[1])
        y_bwd = self.backward_ssm(x_rev)
        y_bwd = torch.flip(y_bwd, dims=[1])

        # Fuse forward + backward
        y = torch.cat([y_fwd, y_bwd], dim=-1)
        y = self.fusion(y)
        y = self.norm(y)

        return y


# =============================================================================
# Patch Embedding (Image → 1D Sequence)
# =============================================================================

class PatchEmbedding(nn.Module):
    """
    Chops the input image into non-overlapping patches and projects each
    patch into the embedding dimension D.
    """
    def __init__(self, img_size=256, patch_size=8, in_channels=3, embed_dim=64):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim

        # Conv2d acts as the patch projection
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )
        # Learnable positional embedding
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches, embed_dim) * 0.02
        )

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W)
        Returns:
            (B, num_patches, embed_dim)
        """
        x = self.proj(x)                          # (B, D, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)           # (B, num_patches, D)
        x = x + self.pos_embed
        return x


# =============================================================================
# Spatial Reconstruction + CNN Refinement
# =============================================================================

class SpatialReconstruction(nn.Module):
    """
    Unflattens the 1D Mamba sequence back to 2D spatial grid and applies
    lightweight depthwise-separable CNN refinement for local edge recovery.
    Outputs K(x) tensor for end-to-end AOD physics.
    """
    def __init__(self, embed_dim=64, img_size=256, patch_size=8, out_channels=3):
        super().__init__()
        self.grid_size = img_size // patch_size   # e.g., 32
        self.embed_dim = embed_dim

        # Project embedding back to spatial channels
        self.proj = nn.Linear(embed_dim, embed_dim)

        # Lightweight CNN refinement (depthwise separable for VRAM efficiency)
        self.refine = nn.Sequential(
            # Upsample from grid_size to img_size.
            # GroupNorm (not BatchNorm): batch stats drift with small/accumulated
            # batches and corrupt eval-mode inference → colour explosion.
            nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=4, stride=4, padding=0),
            nn.GroupNorm(min(8, embed_dim // 2), embed_dim // 2),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim // 2, embed_dim // 4, kernel_size=2, stride=2, padding=0),
            nn.GroupNorm(min(8, embed_dim // 4), embed_dim // 4),
            nn.GELU(),
            # Depthwise separable conv for local edge refinement
            nn.Conv2d(embed_dim // 4, embed_dim // 4, kernel_size=3, padding=1,
                      groups=embed_dim // 4, bias=False),
            nn.Conv2d(embed_dim // 4, out_channels, kernel_size=1, bias=True),
        )

    def forward(self, x):
        """
        Args:
            x: (B, num_patches, embed_dim)
        Returns:
            K: (B, 3, H, W) — the unified physics parameter
        """
        B = x.shape[0]
        x = self.proj(x)

        # Unflatten: (B, L, D) → (B, D, grid, grid)
        x = x.transpose(1, 2).reshape(B, self.embed_dim, self.grid_size, self.grid_size)

        # CNN refinement → K(x)
        K = self.refine(x)
        return K


# =============================================================================
# Main Model: MambaDehaze (End-to-End AOD Physics)
# =============================================================================

class MambaDehaze(nn.Module):
    """
    End-to-End Vision Mamba Dehazer.

    Architecture:
        PatchEmbedding → N × BiDirectionalVimBlock → SpatialReconstruction → K(x)
        J(x) = K(x) * I(x) - K(x) + 1

    Args:
        img_size: input resolution (default 256)
        patch_size: patch size for embedding (default 8)
        embed_dim: SSM embedding dimension (default 64)
        d_state: SSM hidden state dimension (default 16)
        n_layers: number of stacked Vim blocks (default 4)
        dropout: dropout rate
    """
    def __init__(self, img_size=256, patch_size=8, embed_dim=64,
                 d_state=16, n_layers=4, dropout=0.1, k_max=5.0):
        super().__init__()

        self.img_size = img_size
        # Calibration knob: bounds |K| so the AOD equation can't run away.
        # Higher = stronger dehazing headroom, lower = tighter stability.
        self.k_max = k_max

        # Stage 1: Patch Embedding
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=3,
            embed_dim=embed_dim
        )

        # Stage 2: Stacked Bi-Directional Vim Blocks
        self.vim_blocks = nn.ModuleList([
            BiDirectionalVimBlock(embed_dim, d_state, dropout)
            for _ in range(n_layers)
        ])

        # Stage 3: Spatial Reconstruction → K(x)
        self.spatial_head = SpatialReconstruction(
            embed_dim=embed_dim,
            img_size=img_size,
            patch_size=patch_size,
            out_channels=3
        )

    def forward(self, hazy_input):
        """
        Args:
            hazy_input: (B, 3, H, W) — the hazy image tensor

        Returns:
            j_pred: (B, 3, H, W) — the dehazed output (End-to-End AOD physics)
        """
        # --- Mamba Backbone ---
        x = self.patch_embed(hazy_input)               # (B, L, D)

        for block in self.vim_blocks:
            x = block(x)                                # (B, L, D)

        # --- Predict K(x) ---
        K = self.spatial_head(x)                        # (B, 3, H, W)

        # Ensure K(x) output matches input spatial size
        if K.shape[2:] != hazy_input.shape[2:]:
            K = F.interpolate(K, size=hazy_input.shape[2:], mode='bilinear', align_corners=False)

        # Bound K: unbounded per-channel K is what drives the epoch-3-4 blow-up and
        # the colour explosion (channels saturate independently at the clamp).
        K = self.k_max * torch.tanh(K)

        # --- End-to-End AOD Physics ---
        # J(x) = K(x) * I(x) - K(x) + 1
        j_pred = K * hazy_input - K + 1.0

        # Clamp output to valid image range
        j_pred = torch.clamp(j_pred, 0.0, 1.0)

        return j_pred


# =============================================================================
# Quick test & parameter count
# =============================================================================

def _selfcheck_ssm():
    """FFT-conv SSM must equal the naive recurrence it replaces."""
    torch.manual_seed(0)
    blk = S4Block(d_model=8, d_state=4, dropout=0.0).eval()
    u = torch.randn(2, 16, 8)

    fft_y = blk._ssm(u)

    # Reference: explicit recurrence x_k = Abar⊙x_{k-1} + Bbar*u_k ; y_k = <C,x_k> + D*u_k
    dt = torch.exp(blk.log_dt).clamp(1e-3, 0.1)
    Abar = torch.exp(-torch.exp(blk.A_log) * dt.unsqueeze(-1))   # (D, N)
    Bbar = blk.B * dt.unsqueeze(-1)                              # (D, N)
    Bn, L, D = u.shape
    x = torch.zeros(Bn, D, blk.d_state)
    ref = []
    for k in range(L):
        x = Abar * x + Bbar * u[:, k, :].unsqueeze(-1)
        ref.append((blk.C * x).sum(-1) + blk.D * u[:, k, :])
    ref_y = torch.stack(ref, dim=1)

    err = (fft_y - ref_y).abs().max().item()
    assert torch.allclose(fft_y, ref_y, atol=1e-4), f"SSM mismatch, max err={err}"
    print(f"[selfcheck] FFT SSM == recurrence OK (max err={err:.2e})")


if __name__ == "__main__":
    _selfcheck_ssm()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MambaDehaze(img_size=256, embed_dim=64, n_layers=4).to(device)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"MambaDehaze Trainable Parameters: {params:,}")

    dummy = torch.randn(1, 3, 256, 256).to(device)
    with torch.no_grad():
        out = model(dummy)
    print(f"Input:  {dummy.shape}")
    print(f"Output: {out.shape}")

    if torch.cuda.is_available():
        mem = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"Peak VRAM: {mem:.3f} GB")
