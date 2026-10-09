# fakinflu-scail-worker

Worker RunPod Serverless de l'**animation SCAIL-2** (plan « Motion », Phase 2) : le personnage de
l'image d'identité rejoue le mouvement extrait par `body3d`, dans la direction artistique (DA) décrite.

- Base `runpod/worker-comfyui:5.10.0-base` = ComfyUI **0.34.0** (comme body3d). `WanSCAILToVideo` y est
  natif, identique en 0.37.0. Le Dockerfile échoue au build s'il manque.
- Graphe (`workflow.py`) dérivé des blueprints officiels « Character Replacement (SCAIL-2 Base / Extend) » :
  Wan 2.1 14B SCAIL-2 → LoRA DPO → LoRA lightx2v (6 étapes, cfg 1) → morceaux de 81 images enchaînés
  par `previous_frames` (5 images de recouvrement), puis concaténés.
- **Pas de suivi SAM 3.1 ici** (décision du 2026-10-09) : le masque coloré que SCAIL-2 attend (bleu pur sur
  fond noir en animation, sur fond blanc en remplacement) est rendu par `fakinflu_nodes` depuis
  `masks.npz` de body3d, sans codec vidéo. Les couleurs restent donc exactes.
- La vidéo de contrôle (`control_pose.mp4`) arrive du PC (décision 7 du plan) : ni rendu ni suivi ici.
- Poids bakés, commits Hugging Face épinglés et sha256 vérifiés (`download_models.sh`) :
  - `core` : LoRA, umT5 fp8, CLIP vision, VAE, ~10,2 Go ;
  - `fp16` : 32,8 Go, par défaut (`WITH_FP16=1`) ;
  - `int8` : 16,7 Go, en option (`WITH_INT8=1`).

| Entrée | Fichier client |
|---|---|
| `control` | `data/work/<id>/motion/control_pose.mp4` |
| `masks` | `data/work/<id>/motion/masks.npz` (même run que `control`) |
| `ref` | `data/work/<id>/identity/ref.png` |
| `ref_mask` (option) | `data/work/<id>/identity/ref_mask.png` : blanc = personne |

| Sortie | Fichier client |
|---|---|
| `video` | `data/work/<id>/animate/<run_id>/animation.mp4` |

Paramètres : preset `presets/animate/*.json` (`AnimatePreset` dans `app/src/schemas`), plus le prompt de
`data/work/<id>/da/prompt.txt` et le négatif de `negative.txt`.

Côté client :

```bash
make animate ID=<id> [PRESET=default|preview|int8] [SEED=n] DRY=1
make animate ID=<id>
```

Vérifier le graphe contre un ComfyUI local, avec des modèles en fichiers vides (0 crédit) :

```bash
python3 workflow.py --validate http://127.0.0.1:8188 ../../presets/animate/default.json /chemin/masks.npz
```

## GPU (point ouvert 4)

- fp16 : 32,8 Go de poids, donc pool 48 Go (`AMPERE_48` à 1,22 $/h, `ADA_48_PRO` à 1,75 $/h).
- int8 : 16,7 Go. À essayer sur 24 Go (`AMPERE_24`, 0,69 $/h) avec le preset `int8` et une image
  construite avec `WITH_INT8=1`.

## Licences

- Poids SCAIL-2 (Comfy-Org/SCAIL-2) : MIT.
- Encodeurs Wan 2.1 : Apache-2.0.
- LoRA lightx2v (Kijai/WanVideo_comfy, issu de lightx2v) : Apache-2.0 en amont, à confirmer avant un
  usage commercial.
