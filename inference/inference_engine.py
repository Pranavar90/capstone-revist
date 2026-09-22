"""
Inference Engine for the Vision Mamba dehazers.

The architecture is rebuilt from the checkpoint's OWN stored config, so a checkpoint
trained at any capacity — or either model type — loads without editing this file.
Hardcoding the dims here is what silently broke it after every retrain.
"""
import os

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from models.mamba_arch import MambaDehaze
from models.physics_mamba import PhysicsMambaDehaze


class DehazeInference:
    def __init__(self, checkpoint_path, device=None):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"No checkpoint at {checkpoint_path}")

        # weights_only=False: our own checkpoints carry non-tensor state (config,
        # history). torch>=2.6 defaults to True and refuses them.
        ck = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        cfg = ck.get("config", {})
        self.model_type = cfg.get("model", "aod")

        dims = dict(
            img_size=cfg.get("image_size", 256),
            embed_dim=cfg.get("embed_dim", 64),
            d_state=cfg.get("d_state", 16),
            n_layers=cfg.get("n_layers", 4),
        )
        if self.model_type == "physics_mamba":
            self.model = PhysicsMambaDehaze(t_eps=cfg.get("t_eps", 0.1), **dims)
        else:
            self.model = MambaDehaze(k_max=cfg.get("k_max", 5.0), **dims)

        self.t_eps = cfg.get("t_eps", 0.1)
        self.model.load_state_dict(ck["model_state_dict"])
        self.model.to(self.device).eval()

        self.image_size = dims["img_size"]
        self.transform = T.Compose([
            T.Resize((self.image_size, self.image_size)),
            T.ToTensor(),
        ])
        print(f"[Inference] {type(self.model).__name__} "
              f"(embed={dims['embed_dim']}, layers={dims['n_layers']}, "
              f"d_state={dims['d_state']}) on {self.device} "
              f"— PSNR {ck.get('best_psnr', float('nan')):.2f} dB")

    @staticmethod
    def _guided(guide, src, radius, eps=1e-3):
        """Edge-aware refinement of src using guide (He et al.). Box filters -> O(1) in radius."""
        box = lambda m: cv2.boxFilter(m, -1, (radius, radius))
        mg, ms = box(guide), box(src)
        a = (box(guide * src) - mg * ms) / (box(guide * guide) - mg * mg + eps)
        return box(a) * guide + box(ms - a * mg)

    def predict(self, image_path, full_res=True):
        """
        Dehaze a single image.

        For the physics model the ASM is evaluated at the image's NATIVE resolution:
        t is depth-like and genuinely smooth, so upsampling it costs nothing, while
        J = (I - A)/t + A keeps every original pixel of I. Computing J at 256 and
        upscaling the result instead threw away ~98% of the detail (measured
        Laplacian variance 2.4 vs 126.0 on a 2966x3202 image).

        Returns:
            dehazed_img: PIL Image at the original resolution
            metadata: dict; includes the estimated physics params for physics_mamba
        """
        image = Image.open(image_path).convert("RGB")
        original_size = image.size                      # (width, height)
        x = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self.model(x)

        # PhysicsMambaDehaze returns {'J','t','A','beta'}; MambaDehaze returns a tensor.
        if isinstance(out, dict) and full_res:
            W, H = original_size
            I = np.asarray(image, dtype=np.float32) / 255.0          # (H,W,3) RGB
            A = out["A"][0].cpu().numpy()
            t = cv2.resize(out["t"][0, 0].cpu().numpy(), (W, H), interpolation=cv2.INTER_LINEAR)
            # snap t to real edges so it does not halo across depth discontinuities
            r = max(16, min(H, W) // 32) | 1
            t = np.clip(self._guided(cv2.cvtColor(I, cv2.COLOR_RGB2GRAY), t, r), 0.01, 1.0)
            J = np.clip((I - A) / np.maximum(t[..., None], self.t_eps) + A, 0.0, 1.0)
            dehazed_img = Image.fromarray((J * 255).astype(np.uint8))
        else:
            j = out["J"] if isinstance(out, dict) else out
            dehazed_img = T.ToPILImage()(j.squeeze(0).cpu().clamp(0, 1))
            dehazed_img = dehazed_img.resize(original_size, Image.LANCZOS)

        metadata = {
            "model_type": self.model_type,
            "input_size": original_size,
            "processing_size": (self.image_size, self.image_size),
        }
        if isinstance(out, dict):
            metadata["A"] = [round(v, 4) for v in out["A"][0].cpu().tolist()]
            metadata["beta"] = round(float(out["beta"][0]), 4)
            metadata["t_mean"] = round(float(out["t"].mean()), 4)

        return dehazed_img, metadata


# =============================================================================
# Self-check: load the real checkpoint and dehaze a real image.
# Run:  python -m inference.inference_engine
# =============================================================================
if __name__ == "__main__":
    import glob

    ck = "outputs/checkpoints/mamba_best.pth"
    engine = DehazeInference(ck)

    samples = sorted(glob.glob("uitestimages/*.jpg") + glob.glob("uitestimages/*.avif")) \
        or sorted(glob.glob("data/processed/test/hazy/*.png"))[:1]
    assert samples, "no sample image found"

    sharp = lambda im: cv2.Laplacian(
        cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()

    src = Image.open(samples[0]).convert("RGB")
    hi, _ = engine.predict(samples[0])                    # native-resolution physics
    lo, meta = engine.predict(samples[0], full_res=False)  # old 256-then-upscale path

    assert hi.size == src.size, f"resolution not preserved: {hi.size} != {src.size}"
    assert sharp(hi) > sharp(lo), "full-res path is not sharper than the 256 path"
    print(f"[selfcheck] {samples[0]} {src.size} -> {hi.size} | {meta}")
    print(f"[selfcheck] sharpness  full-res {sharp(hi):.1f}  vs  256-upscale {sharp(lo):.1f} "
          f"(input {sharp(src):.1f})")
