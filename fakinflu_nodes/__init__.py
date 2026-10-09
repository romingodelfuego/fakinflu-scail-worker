"""fakinflu — noeuds ComfyUI de l'animation SCAIL-2 (plan motion, Phase 2).

SCAIL-2 attend, a cote de la video de pose, un masque colore de la personne : bleu pur (0, 0, 1),
seule couleur d'une identite unique dans la palette d'entrainement, sur fond noir (mode animation)
ou blanc (mode remplacement). WanSCAILToVideo le convertit en 28 canaux avec un seuil a 225/255.

Le blueprint officiel le fabrique avec SAM3_VideoTrack + SCAIL2ColoredMask. Ici on NE refait PAS le
suivi (decision utilisateur 2026-10-09) : masks.npz vient de body3d (FakinfluExportMHR, meme suivi
SAM 3.1, memes images que la video de controle). Pas de codec video entre les deux : couleurs exactes.

Memoire (OOM mesure sur body3d le 2026-10-09 : ComfyUI garde chaque IMAGE en float32) :
  - FakinfluSCAILPrep decode lui-meme la video de controle et ne garde QUE les images retenues, deja
    a la taille cible (LoadVideo decoderait tout a pleine resolution : ~5,4 Go pour 554 images 632x1280) ;
  - FakinfluSaveSegment encode chaque morceau en H.264 des qu'il est decode, FakinfluConcatVideo les
    assemble sans reencodage : plus de concatenation de tenseurs (ImageBatch) ni de CreateVideo.

  FakinfluSCAILPrep     control.mp4 + masks.npz -> (pose, masque colore) a la cadence et a la taille cibles
  FakinfluSCAILRefMask  image noir et blanc (blanc = personne) -> masque colore de la reference
  FakinfluSaveSegment   morceau decode -> segment_<k>.mp4 (recouvrement retire), liste chainee des segments
  FakinfluConcatVideo   segments -> animation.mp4 (ffmpeg concat, copie de flux)
"""
import os
import subprocess

import numpy as np
import torch

import comfy.model_management
import comfy.utils
import folder_paths

BLUE = (0.0, 0.0, 1.0)  # DEFAULT_PALETTE[0] de comfy_extras/nodes_scail.py
CHUNK = 16  # images redimensionnees a la fois (pic memoire borne)


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


def decode_frames(path, keep):
    """Generateur (index, image RGB uint8 HxWx3) des seules images `keep` de la video, dans l'ordre."""
    import av  # PyAV, present dans worker-comfyui (LoadVideo s'en sert)
    wanted, last = set(keep), max(keep)
    with av.open(path) as c:
        for i, frame in enumerate(c.decode(video=0)):
            if i in wanted:
                yield i, frame.to_ndarray(format="rgb24")
            if i >= last:
                break


class FakinfluSCAILPrep:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "video_path": ("STRING", {"default": ""}),
            "masks_path": ("STRING", {"default": ""}),
            "src_frames": ("INT", {"default": 0, "min": 1, "max": 100000}),
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

    def run(self, video_path, masks_path, src_frames, src_fps, fps, width, height, replacement_mode):
        masks = load_masks(masks_path)
        if abs(src_frames - masks.shape[0]) > 1:
            raise ValueError(f"controle ({src_frames} images) et masks.npz ({masks.shape[0]}) ne viennent pas du meme run")
        idx = frame_indices(min(src_frames, masks.shape[0]), src_fps, fps)
        dev, dt = comfy.model_management.intermediate_device(), comfy.model_management.intermediate_dtype()
        pose = torch.empty((len(idx), height, width, 3), dtype=dt, device=dev)
        mask = torch.empty((len(idx), height, width, 3), dtype=dt, device=dev)
        # Animation : masque sur fond noir. Remplacement : fond blanc (cf. SCAIL2ColoredMask).
        bg = "white" if replacement_mode else "black"
        pos = {src: k for k, src in enumerate(idx)}
        buf = []

        def flush():
            ks = [pos[i] for i, _ in buf]
            frames = torch.from_numpy(np.stack([f for _, f in buf])).to(device=dev, dtype=dt) / 255.0
            pose[ks[0]:ks[-1] + 1] = _fit(frames, width, height, "area")
            colored = torch.from_numpy(colorize(masks[[i for i, _ in buf]], bg)).to(device=dev, dtype=dt)
            mask[ks[0]:ks[-1] + 1] = _fit(colored, width, height, "nearest-exact")
            buf.clear()

        got = 0
        for i, frame in decode_frames(video_path, idx):
            buf.append((i, frame))
            got += 1
            if len(buf) == CHUNK:
                flush()
        if buf:
            flush()
        if got != len(idx):
            raise ValueError(f"{video_path} : {got} images decodees sur {len(idx)} attendues")
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


def _out_dir(prefix):
    d = os.path.join(folder_paths.get_output_directory(), prefix)
    os.makedirs(d, exist_ok=True)
    return d


def encode(images, path, fps):
    """IMAGE (N, H, W, 3) float -> H.264 (yuv420p, crf 16), envoye par paquets a ffmpeg."""
    n, h, w, _ = images.shape
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                          "-r", f"{fps}", "-i", "-", "-c:v", "libx264", "-crf", "16", "-preset", "medium",
                          "-pix_fmt", "yuv420p", path], stdin=subprocess.PIPE)
    try:
        for s in range(0, n, CHUNK):
            p.stdin.write((images[s:s + CHUNK].clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy().tobytes())
    finally:
        p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError(f"ffmpeg a echoue sur {path}")


class FakinfluSaveSegment:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "filename_prefix": ("STRING", {"default": "fakinflu/animation"}),
            "index": ("INT", {"default": 0, "min": 0, "max": 1000}),
            "skip": ("INT", {"default": 0, "min": 0, "max": 1000, "tooltip": "images de recouvrement a retirer en tete"}),
            "fps": ("FLOAT", {"default": 16.0, "min": 1.0, "max": 240.0}),
        }, "optional": {"previous": ("STRING", {"forceInput": True})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("segments",)
    FUNCTION = "run"
    CATEGORY = "fakinflu"

    def run(self, images, filename_prefix, index, skip, fps, previous=None):
        path = os.path.join(_out_dir(filename_prefix), f"segment_{index:03d}.mp4")
        encode(images[skip:], path, fps)
        return ("\n".join([p for p in (previous or "").split("\n") if p] + [path]),)


class FakinfluConcatVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"segments": ("STRING", {"forceInput": True}),
                             "filename_prefix": ("STRING", {"default": "fakinflu/animation"})}}

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = "fakinflu"

    def run(self, segments, filename_prefix):
        paths = [p for p in segments.split("\n") if p]
        d = _out_dir(filename_prefix)
        lst = os.path.join(d, "segments.txt")
        with open(lst, "w") as f:
            f.writelines(f"file '{p}'\n" for p in paths)
        out = os.path.join(d, "animation_00001_.mp4")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy",
                        "-movflags", "+faststart", out], check=True, timeout=600)
        for p in paths + [lst]:
            os.remove(p)
        return {"ui": {"text": [out]}}


NODE_CLASS_MAPPINGS = {"FakinfluSCAILPrep": FakinfluSCAILPrep, "FakinfluSCAILRefMask": FakinfluSCAILRefMask,
                       "FakinfluSaveSegment": FakinfluSaveSegment, "FakinfluConcatVideo": FakinfluConcatVideo}
NODE_DISPLAY_NAME_MAPPINGS = {"FakinfluSCAILPrep": "fakinflu SCAIL-2 : pose + masque (masks.npz)",
                              "FakinfluSCAILRefMask": "fakinflu SCAIL-2 : masque de la reference",
                              "FakinfluSaveSegment": "fakinflu SCAIL-2 : encodage d'un morceau",
                              "FakinfluConcatVideo": "fakinflu SCAIL-2 : assemblage de la video"}
