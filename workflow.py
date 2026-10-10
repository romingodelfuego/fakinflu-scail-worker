"""fakinflu — graphe ComfyUI (format API) du character replacement SCAIL-2.

Derive des blueprints officiels « Character Replacement (SCAIL-2 Base / Extend) » (ComfyUI >= 0.35,
noeuds identiques dans 0.34.0). Comme eux, `pose_video` est la VIDEO SOURCE brute : SCAIL-2 est entraine
« sans carte de pose intermediaire » (un squelette rendu le degrade : cadrage, echelle et decor perdus).
Un seul ecart voulu :

  - PAS de suivi SAM 3.1 ici (decision utilisateur 2026-10-09) : le masque colore de la video de
    pose est rendu par FakinfluSCAILPrep depuis masks.npz, deja produit par body3d (meme suivi).

Mode par defaut : replacement (decor, lumiere et camera de la source gardes ; masque sur fond blanc).

  source.mp4 + masks.npz -> FakinfluSCAILPrep (decode SEULEMENT les images retenues, deja a la taille cible)
  ref.png -> reference + CLIP vision ; ref_mask.png (option) -> FakinfluSCAILRefMask
  Wan 2.1 14B SCAIL-2 -> LoRA DPO -> LoRA lightx2v (distillation 6 etapes, cfg 1) -> ModelSamplingSD3
  morceau k : images [k*(L-P), +L) -> WanSCAILToVideo (previous_frames = morceau k-1) -> SamplerCustom
              -> VAEDecode -> FakinfluSaveSegment (H.264 des le decodage, recouvrement retire)
  -> FakinfluConcatVideo (ffmpeg concat, sans reencodage) : aucune concatenation de tenseurs en RAM
  prep + ref -> FakinfluInputsSheet : planche « vu par le modele » (sortie `sheet`)
"""
import json, os, sys

OUTPUT_FILES = {"video": "animation_*.mp4", "sheet": "inputs_sheet_*.png"}
OUTPUT_NAMES = {"video": "animation.mp4", "sheet": "inputs_sheet.png"}

UNET_FILES = {"fp16": "wan2.1_14B_SCAIL_2_fp16.safetensors", "int8": "wan2.1_14B_SCAIL_2_int8_convrot.safetensors"}
DPO_LORA = "wan2.1_SCAIL_2_DPO_lora_bf16.safetensors"
DISTILL_LORA = "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors"
TEXT_ENCODER = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
CLIP_VISION = "clip_vision_h.safetensors"
VAE = "Wan2_1_VAE_bf16.safetensors"

SEGMENT = 81   # images par morceau (entrainement SCAIL-2)
OVERLAP = 5    # images de recouvrement entre deux morceaux (« SCAIL-2 trained at 5 »)


def frame_indices(n_src, src_fps, fps):
    """Indices des images source gardees pour passer de src_fps a fps (sans interpolation).
    Copie identique dans fakinflu_nodes (le noeud ne peut pas importer ce module en local)."""
    if not fps or fps >= src_fps:
        return list(range(n_src))
    step = src_fps / fps
    return [i for i in (int(round(k * step)) for k in range(int(n_src / step) + 1)) if i < n_src]


def _len4(n):
    """Longueur valide pour Wan : 4m + 1 (sinon WanSCAILToVideo tronque en silence)."""
    return ((n - 1) // 4) * 4 + 1 if n >= 1 else 0


def segments(n, seg=SEGMENT, overlap=OVERLAP):
    """[(debut, longueur)] couvrant n images ; chaque morceau apres le premier apporte longueur - overlap images."""
    out, start = [], 0
    while True:
        length = _len4(min(seg, n - start))
        if out and length - overlap < 4:
            break
        if not out and length < 5:
            raise ValueError(f"video de controle trop courte : {n} images (5 minimum)")
        out.append((start, length))
        if start + length >= n:
            break
        start += seg - overlap
    return out


def output_frames(segs, overlap=OVERLAP):
    return sum(length if i == 0 else length - overlap for i, (_, length) in enumerate(segs))


def size_for(src_w, src_h, short):
    """Largeur et hauteur multiples de 32 (exigence WanSCAILToVideo), cote court = short."""
    scale = short / min(src_w, src_h)
    r32 = lambda v: max(32, int(round(v * scale / 32)) * 32)
    return r32(src_w), r32(src_h)


def build_workflow(params, prompt, negative, files, prefix, meta):
    """params : preset d'animation resolu (app/src/schemas AnimatePreset) ; prompt / negative : DA.
    files : chemins absolus {source, masks} (lus par FakinfluSCAILPrep) et noms dans input/ {ref, ref_mask?}.
    meta : {width, height, fps, frames} de la video source ; params.masks_fps = cadence de masks.npz (0 = celle de la video)."""
    p = params or {}
    mode = p.get("mode", "replacement")
    if mode not in ("animation", "replacement"):
        raise ValueError(f"mode inconnu : {mode}")
    replacement = mode == "replacement"
    sampling = p.get("sampling", {})
    distill = bool(sampling.get("distill", True))
    steps = int(sampling.get("steps") or (6 if distill else 40))
    cfg = float(sampling.get("cfg") or (1.0 if distill else 5.0))
    seed = int(p.get("seed") if p.get("seed") is not None else 0)
    pose = p.get("pose", {})
    fps = float(p.get("fps") or meta["fps"])
    width, height = size_for(meta["width"], meta["height"], int(p.get("resolution", 512)))

    n = len(frame_indices(meta["frames"], meta["fps"], fps))
    segs = segments(n)

    wf = {
        "prep": {"class_type": "FakinfluSCAILPrep", "_meta": {"title": "pose + masque (masks.npz)"},
                 "inputs": {"video_path": files["source"], "masks_path": files["masks"],
                            "src_frames": int(meta["frames"]), "src_fps": float(meta["fps"]),
                            "masks_fps": float(p.get("masks_fps") or 0.0),
                            "fps": fps, "width": width, "height": height, "replacement_mode": replacement}},
        "ref": {"class_type": "LoadImage", "_meta": {"title": "image d'identite"}, "inputs": {"image": files["ref"]}},
        # --- modeles
        "unet": {"class_type": "UNETLoader", "_meta": {"title": "chargement Wan 2.1 SCAIL-2"},
                 "inputs": {"unet_name": UNET_FILES[p.get("precision", "fp16")], "weight_dtype": "default"}},
        "dpo": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["unet", 0], "lora_name": DPO_LORA,
                                                                 "strength_model": float(p.get("dpo_strength", 1.0))}},
        "clip": {"class_type": "CLIPLoader", "_meta": {"title": "chargement umT5"},
                 "inputs": {"clip_name": TEXT_ENCODER, "type": "wan"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "clip_vision": {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": CLIP_VISION}},
        "clip_vision_enc": {"class_type": "CLIPVisionEncode",
                            "inputs": {"clip_vision": ["clip_vision", 0], "image": ["ref", 0], "crop": "none"}},
        "pos": {"class_type": "CLIPTextEncode", "_meta": {"title": "prompt"},
                "inputs": {"clip": ["clip", 0], "text": prompt or ""}},
        "neg": {"class_type": "CLIPTextEncode", "_meta": {"title": "prompt negatif"},
                "inputs": {"clip": ["clip", 0], "text": negative or ""}},
        "sampler": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
    }
    model = ["dpo", 0]
    if distill:
        wf["distill"] = {"class_type": "LoraLoaderModelOnly",
                         "inputs": {"model": model, "lora_name": DISTILL_LORA,
                                    "strength_model": float(sampling.get("distill_strength", 0.8))}}
        model = ["distill", 0]
    wf["shift"] = {"class_type": "ModelSamplingSD3", "inputs": {"model": model, "shift": float(sampling.get("shift", 5.0))}}
    wf["sigmas"] = {"class_type": "BasicScheduler",
                    "inputs": {"model": model, "scheduler": "simple", "steps": steps, "denoise": 1.0}}

    ref_mask = None
    if files.get("ref_mask"):
        wf["ref_mask_img"] = {"class_type": "LoadImage", "inputs": {"image": files["ref_mask"]}}
        wf["ref_mask"] = {"class_type": "FakinfluSCAILRefMask", "_meta": {"title": "masque de la reference"},
                          "inputs": {"image": ["ref_mask_img", 0], "replacement_mode": replacement}}
        ref_mask = ["ref_mask", 0]
    wf["sheet"] = {"class_type": "FakinfluInputsSheet", "_meta": {"title": "planche des entrees"},
                   "inputs": {"pose_video": ["prep", 0], "pose_video_mask": ["prep", 1], "reference_image": ["ref", 0],
                              "width": width, "height": height, "filename_prefix": prefix,
                              "title": f"{mode} · {width}x{height} · {fps:g} i/s · {len(segs)} morceau(x) · "
                                       f"pose {pose.get('strength', 1.0)} · seed {seed}"}}
    if ref_mask:
        wf["sheet"]["inputs"]["reference_image_mask"] = ref_mask

    saved, prev = None, None
    for k, (start, length) in enumerate(segs):
        s = f"s{k}"
        wf[f"{s}_pose"] = {"class_type": "ImageFromBatch", "inputs": {"image": ["prep", 0], "batch_index": start, "length": length}}
        wf[f"{s}_mask"] = {"class_type": "ImageFromBatch", "inputs": {"image": ["prep", 1], "batch_index": start, "length": length}}
        cond = {"positive": ["pos", 0], "negative": ["neg", 0], "vae": ["vae", 0],
                "width": width, "height": height, "length": length, "batch_size": 1,
                "pose_video": [f"{s}_pose", 0], "pose_video_mask": [f"{s}_mask", 0], "replacement_mode": replacement,
                "pose_strength": float(pose.get("strength", 1.0)), "pose_start": float(pose.get("start", 0.0)),
                "pose_end": float(pose.get("end", 1.0)), "reference_image": ["ref", 0],
                "clip_vision_output": ["clip_vision_enc", 0], "video_frame_offset": 0, "previous_frame_count": OVERLAP}
        if ref_mask:
            cond["reference_image_mask"] = ref_mask
        if prev:
            cond["previous_frames"] = prev
        wf[f"{s}_cond"] = {"class_type": "WanSCAILToVideo", "_meta": {"title": f"morceau {k + 1}/{len(segs)}"}, "inputs": cond}
        wf[f"{s}_sample"] = {"class_type": "SamplerCustom", "_meta": {"title": f"generation {k + 1}/{len(segs)}"},
                             "inputs": {"model": ["shift", 0], "add_noise": True, "noise_seed": seed + k, "cfg": cfg,
                                        "positive": [f"{s}_cond", 0], "negative": [f"{s}_cond", 1],
                                        "sampler": ["sampler", 0], "sigmas": ["sigmas", 0], "latent_image": [f"{s}_cond", 2]}}
        wf[f"{s}_dec"] = {"class_type": "VAEDecode", "_meta": {"title": f"decodage {k + 1}/{len(segs)}"},
                          "inputs": {"samples": [f"{s}_sample", 1], "vae": ["vae", 0]}}
        prev = [f"{s}_dec", 0]
        save = {"images": prev, "filename_prefix": prefix, "index": k, "skip": OVERLAP if k else 0, "fps": fps}
        if saved:
            save["previous"] = saved
        wf[f"{s}_save"] = {"class_type": "FakinfluSaveSegment", "_meta": {"title": f"encodage {k + 1}/{len(segs)}"}, "inputs": save}
        saved = [f"{s}_save", 0]

    wf["save"] = {"class_type": "FakinfluConcatVideo", "_meta": {"title": "assemblage"},
                  "inputs": {"segments": saved, "filename_prefix": prefix}}
    return wf


def plan(params, meta):
    """Resume sans GPU (dry-run, estimation) : taille, cadence, morceaux, images produites."""
    p = params or {}
    fps = float(p.get("fps") or meta["fps"])
    n = len(frame_indices(meta["frames"], meta["fps"], fps))
    segs = segments(n)
    w, h = size_for(meta["width"], meta["height"], int(p.get("resolution", 512)))
    return {"width": w, "height": h, "fps": fps, "frames_in": n, "segments": len(segs), "frames_out": output_frames(segs)}


def ram_bytes(plan_):
    """Pic de RAM des IMAGE gardees par ComfyUI (float32 RGB) : pose + masque (frames_in chacun) et les
    morceaux decodes, en cache jusqu'a la fin du graphe (~frames_out + recouvrements). Hors poids du modele."""
    per_frame = plan_["width"] * plan_["height"] * 3 * 4
    return per_frame * (2 * plan_["frames_in"] + plan_["frames_out"] + OVERLAP * plan_["segments"])


def _validate(url, preset_path, masks_path):
    """Soumet le graphe a un ComfyUI (local, 0 credit) : il le valide avant toute execution.
    input.mp4 (source) a cote de masks.npz ; input/ doit contenir ref.png ; modeles en fichiers vides."""
    import urllib.error, urllib.request
    preset = json.load(open(preset_path))
    preset.setdefault("masks_fps", 30.0)
    meta = {"width": 632, "height": 1280, "fps": 59.94, "frames": 1103}  # shadow-boxing ; masks.npz a 30 i/s
    source = os.path.join(os.path.dirname(os.path.abspath(masks_path)), "input.mp4")
    wf = build_workflow(preset, "validation", "", {"source": source, "ref": "ref.png", "masks": masks_path},
                        "fakinflu/validate", meta)
    post = lambda path, body: urllib.request.urlopen(urllib.request.Request(
        url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=30)
    try:
        r = json.loads(post("/prompt", {"prompt": wf}).read())
    except urllib.error.HTTPError as e:
        print(e.read().decode()[:6000])
        sys.exit(1)
    post("/queue", {"clear": True}).close()  # accepte -> on vide la file (modeles factices)
    post("/interrupt", {}).close()
    print(f"OK : graphe accepte par ComfyUI ({len(wf)} noeuds), prompt_id={r.get('prompt_id')}")


if __name__ == "__main__":
    if len(sys.argv) >= 5 and sys.argv[1] == "--validate":
        _validate(sys.argv[2], sys.argv[3], sys.argv[4])
        sys.exit(0)
    params = json.load(open(sys.argv[1])) if len(sys.argv) > 1 else {}
    meta = {"width": 632, "height": 1280, "fps": 30.0, "frames": 554}
    print(json.dumps(plan(params, meta)))
    print(json.dumps(build_workflow(params, "prompt", "", {"source": "/tmp/source.mp4", "ref": "ref.png", "masks": "/tmp/masks.npz"},
                                    "fakinflu/run", meta), indent=1))
