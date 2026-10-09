"""fakinflu — noeuds ComfyUI de l'animation SCAIL-2 (plan motion, Phase 2).

SCAIL-2 attend, a cote de la video de pose, un masque colore de la personne : bleu pur (0, 0, 1),
seule couleur d'une identite unique dans la palette d'entrainement, sur fond noir (mode animation)
ou blanc (mode remplacement). WanSCAILToVideo le convertit en 28 canaux avec un seuil a 225/255.

Le blueprint officiel le fabrique avec SAM3_VideoTrack + SCAIL2ColoredMask. Ici on NE refait PAS le
suivi (decision utilisateur 2026-10-09) : masks.npz vient de body3d (FakinfluExportMHR, meme suivi
SAM 3.1, memes images que la video de controle). Pas de codec video entre les deux : couleurs exactes.

  FakinfluSCAILPrep     images de controle + masks.npz -> (pose, masque colore) a la cadence et a la taille cibles
  FakinfluSCAILRefMask  image noir et blanc (blanc = personne) -> masque colore de la reference
"""
import numpy as np
import torch

import comfy.model_management
import comfy.utils

BLUE = (0.0, 0.0, 1.0)  # DEFAULT_PALETTE[0] de comfy_extras/nodes_scail.py


def frame_indices(n_src, src_fps, fps):
    """Copie identique de workflow.frame_indices (verifiee par scail/test_scail.py)."""
    if not fps or fps >= src_fps:
        return list(range(n_src))
    step = src_fps / fps
    return [i for i in (int(round(k * step)) for k in range(int(n_src / step) + 1)) if i < n_src]


def load_masks(path):
    """masks.npz de body3d : bits = packbits(masques, axis=-1), shape = (T, H, W)."""
    d = np.load(path)
    t, h, w = (int(v) for v in d["shape"])
    return np.unpackbits(d["bits"], axis=-1, count=w).reshape(t, h, w).astype(bool)


def colorize(mask_bool, background):
    """(T, H, W) bool -> (T, H, W, 3) float : personne en bleu, fond noir ou blanc."""
    bg = 1.0 if background == "white" else 0.0
    out = np.full(mask_bool.shape + (3,), bg, dtype=np.float32)
    out[mask_bool] = BLUE
    return out


def _fit(images, width, height, method):
    """Meme recadrage que WanSCAILToVideo / ResizeImageMaskNode (centre), pour que pose et masque coincident."""
    return comfy.utils.common_upscale(images.movedim(-1, 1), width, height, method, "center").movedim(1, -1)


class FakinfluSCAILPrep:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "masks_path": ("STRING", {"default": ""}),
            "src_fps": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 240.0}),
            "fps": ("FLOAT", {"default": 16.0, "min": 1.0, "max": 240.0}),
            "width": ("INT", {"default": 512, "min": 32, "max": 4096, "step": 32}),
            "height": ("INT", {"default": 1024, "min": 32, "max": 4096, "step": 32}),
            "replacement_mode": ("BOOLEAN", {"default": False}),
        }}

    RETURN_TYPES = ("IMAGE", "IMAGE", "INT")
    RETURN_NAMES = ("pose_video", "pose_video_mask", "frames")
    FUNCTION = "run"
    CATEGORY = "fakinflu"

    def run(self, images, masks_path, src_fps, fps, width, height, replacement_mode):
        masks = load_masks(masks_path)
        n_src = min(images.shape[0], masks.shape[0])
        if abs(images.shape[0] - masks.shape[0]) > 1:
            raise ValueError(f"controle ({images.shape[0]} images) et masks.npz ({masks.shape[0]}) ne viennent pas du meme run")
        idx = frame_indices(n_src, src_fps, fps)
        dev, dt = comfy.model_management.intermediate_device(), comfy.model_management.intermediate_dtype()
        pose = _fit(images[idx].to(device=dev, dtype=dt), width, height, "area")
        # Animation : masque sur fond noir. Remplacement : fond blanc (cf. SCAIL2ColoredMask).
        colored = torch.from_numpy(colorize(masks[idx], "white" if replacement_mode else "black"))
        mask = _fit(colored.to(device=dev, dtype=dt), width, height, "nearest-exact")
        return (pose, mask, len(idx))


class FakinfluSCAILRefMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), "replacement_mode": ("BOOLEAN", {"default": False})}}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("reference_image_mask",)
    FUNCTION = "run"
    CATEGORY = "fakinflu"

    def run(self, image, replacement_mode):
        person = (image[..., :3].mean(dim=-1) > 0.5).cpu().numpy()
        # Reference : fond blanc en animation, noir en remplacement (inverse de la video de pose).
        return (torch.from_numpy(colorize(person, "black" if replacement_mode else "white")),)


NODE_CLASS_MAPPINGS = {"FakinfluSCAILPrep": FakinfluSCAILPrep, "FakinfluSCAILRefMask": FakinfluSCAILRefMask}
NODE_DISPLAY_NAME_MAPPINGS = {"FakinfluSCAILPrep": "fakinflu SCAIL-2 : pose + masque (masks.npz)",
                              "FakinfluSCAILRefMask": "fakinflu SCAIL-2 : masque de la reference"}
