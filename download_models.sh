#!/usr/bin/env bash
# fakinflu — poids de l'animation SCAIL-2 (plan motion, Phase 2), BAKES dans l'image.
# Memes fichiers que les blueprints officiels « Character Replacement (SCAIL-2 ...) » de ComfyUI,
# SANS SAM 3.1 (le masque vient de masks.npz de body3d). Depots EPINGLES sur un commit, chaque fichier
# verifie par son sha256 (= oid LFS lu via l'API HF le 2026-10-09).
#   Wan 2.1 14B SCAIL-2 (+ LoRA DPO)  : MIT (Comfy-Org/SCAIL-2)        -> diffusion_models/, loras/
#   umT5 XXL fp8, CLIP vision H       : Apache-2.0 (Wan 2.1 repackaged) -> text_encoders/, clip_vision/
#   LoRA lightx2v (distillation 6 et.) et VAE Wan 2.1 : Kijai/WanVideo_comfy -> loras/, vae/
#
#   MODELS_DIR=/comfyui/models ./download_models.sh core   # ~10,2 Go (LoRA, encodeurs, VAE)
#   MODELS_DIR=/comfyui/models ./download_models.sh fp16   # 32,8 Go (preset precision=fp16)
#   MODELS_DIR=/comfyui/models ./download_models.sh int8   # 16,7 Go (preset precision=int8)
set -euo pipefail

MODELS_DIR="${MODELS_DIR:-/comfyui/models}"

# DL <depot> <commit> <chemin dans le depot> <sous-dossier local> <sha256>
DL() {
  local repo="$1" rev="$2" src="$3" sub="$4" sha="$5" name
  name="$(basename "$src")"
  local dest="$MODELS_DIR/$sub/$name" url="https://huggingface.co/${repo}/resolve/${rev}/${src}"
  mkdir -p "$MODELS_DIR/$sub"
  echo ">> $sub/$name"
  local t0=$SECONDS
  if command -v aria2c >/dev/null; then
    aria2c -q -c -x 16 -s 16 -k 64M --max-tries=10 --retry-wait=5 --timeout=30 \
           --lowest-speed-limit=1M --file-allocation=none --console-log-level=warn \
           -d "$MODELS_DIR/$sub" -o "$name" "$url"
  else
    wget -q -c --tries=10 --timeout=30 --waitretry=5 -O "$dest" "$url"
  fi
  echo "   telecharge en $((SECONDS - t0)) s"
  echo "$sha  $dest" | sha256sum -c --quiet - || { echo "!! sha256 invalide : $dest"; rm -f "$dest"; exit 1; }
}

SCAIL=fe3c728bc793ba21ca674688f822afb709ad44fb
WAN=123acf1cc74bccbb9bfff8ac1ee72edc08c2341d
KIJAI=8260d429d19fd7a72304cad059160b95d843913f

case "${1:-}" in
  core)
    DL Comfy-Org/SCAIL-2 "$SCAIL" loras/wan2.1_SCAIL_2_DPO_lora_bf16.safetensors loras \
       b106522036f64e50f5f8ae3b808973515ff442cc2fac27b65d875eafb95b89e2
    DL Kijai/WanVideo_comfy "$KIJAI" Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors loras \
       85c4a61c30e0497aa44b91d93a893b624708461a56fe5485183b28fa07e2dfb3
    DL Comfy-Org/Wan_2.1_ComfyUI_repackaged "$WAN" split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors text_encoders \
       c3355d30191f1f066b26d93fba017ae9809dce6c627dda5f6a66eaa651204f68
    DL Comfy-Org/Wan_2.1_ComfyUI_repackaged "$WAN" split_files/clip_vision/clip_vision_h.safetensors clip_vision \
       64a7ef761bfccbadbaa3da77366aac4185a6c58fa5de5f589b42a65bcc21f161
    DL Kijai/WanVideo_comfy "$KIJAI" Wan2_1_VAE_bf16.safetensors vae \
       1ab9a32cc2c740f6e39d80d367ce5dcc28db8c71b79b28670546b8973e9d75f9 ;;
  fp16)
    DL Comfy-Org/SCAIL-2 "$SCAIL" diffusion_models/wan2.1_14B_SCAIL_2_fp16.safetensors diffusion_models \
       e3f9cd52184949ee8ffdc01ba14de4c619e0ddfa09ae105cb7efce5c8dc9fe43 ;;
  int8)
    DL Comfy-Org/SCAIL-2 "$SCAIL" diffusion_models/wan2.1_14B_SCAIL_2_int8_convrot.safetensors diffusion_models \
       a2114c18a811243185f103765cd5004b61652141aa55a9016d3bdceebceb5ac0 ;;
  *) echo "usage: $0 core|fp16|int8" >&2; exit 2 ;;
esac
