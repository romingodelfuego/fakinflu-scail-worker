# fakinflu — worker RunPod : scail (animation SCAIL-2, Wan 2.1 14B via ComfyUI). Plan motion, Phase 2.
# Build par RunPod (integration GitHub) a chaque GitHub Release : make worker-release W=scail
#
# Base : runpod/worker-comfyui 5.10.0-base = ComfyUI 0.34.0, comme body3d. WanSCAILToVideo y est NATIF
# (comfy_extras/nodes_scail.py, identique en 0.37.0) ; seuls les blueprints arrivent en 0.35.0, et ce
# worker construit son graphe lui-meme (workflow.py). Seul ajout : fakinflu_nodes (masque depuis masks.npz).
ARG WORKER_TAG=5.10.0-base
FROM runpod/worker-comfyui:${WORKER_TAG}

# Garde-fou de parite de version : echoue ICI si ComfyUI n'a pas les noeuds attendus.
RUN grep -q '"WanSCAILToVideo"' /comfyui/comfy_extras/nodes_scail.py \
 && grep -q 'pose_video_mask' /comfyui/comfy_extras/nodes_scail.py \
 && grep -q '"SamplerCustom"' /comfyui/comfy_extras/nodes_custom_sampler.py

# ComfyUI dimensionne sa memoire epinglee sur la RAM de l'HOTE (psutil : 503 Go sur un A40 RunPod -> 464 Go
# epinglables), pas sur la limite du conteneur : en chargeant les 31 Go du modele, le noyau tue ComfyUI sans
# trace (job du 2026-10-09). start.sh de worker-comfyui n'a aucun point d'entree pour des options : on y
# ajoute ${COMFY_EXTRA_ARGS} (defaut ci-dessous, modifiable dans l'env de l'endpoint sans rebuild).
RUN sed -i 's#--log-stdout &#--log-stdout ${COMFY_EXTRA_ARGS} \&#' /start.sh \
 && test "$(grep -c 'COMFY_EXTRA_ARGS} &' /start.sh)" -ge 1
ENV COMFY_EXTRA_ARGS="--disable-pinned-memory"

RUN apt-get update && apt-get install -y --no-install-recommends aria2 \
 && rm -rf /var/lib/apt/lists/*

# --- Poids BAKES : core ~10,2 Go + fp16 32,8 Go (WITH_FP16) + int8 16,7 Go (WITH_INT8) = ~60 Go --------
# Les deux par defaut : le preset choisit sans rebuild (int8 = modele resident en VRAM sur 48 Go).
ARG WITH_FP16=1
ARG WITH_INT8=1
COPY download_models.sh /tmp/download_models.sh
RUN MODELS_DIR=/comfyui/models bash /tmp/download_models.sh core
RUN if [ "${WITH_FP16}" = "1" ]; then MODELS_DIR=/comfyui/models bash /tmp/download_models.sh fp16; fi
RUN if [ "${WITH_INT8}" = "1" ]; then MODELS_DIR=/comfyui/models bash /tmp/download_models.sh int8; fi \
 && rm -f /tmp/download_models.sh

COPY fakinflu_nodes /comfyui/custom_nodes/fakinflu_nodes

# --- Handler fakinflu (remplace /handler.py de worker-comfyui) ---------------
# VERSION (ecrit par make worker-release dans le depot miroir) : absent en build local -> "dev"
COPY transfer.py workflow.py VERSION* /fakinflu/
COPY handler.py /handler.py
COPY test_input.json /test_input.json
ENV PYTHONPATH=/fakinflu \
    COMFY_LOG_LEVEL=INFO

CMD ["/start.sh"]
