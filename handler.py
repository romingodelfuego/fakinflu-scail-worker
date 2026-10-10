#!/usr/bin/env python3
"""fakinflu — worker scail : character replacement SCAIL-2 (Wan 2.1 14B via ComfyUI).

Remplace le /handler.py de runpod/worker-comfyui ; /start.sh lance ComfyUI (127.0.0.1:8188).
Handler GENERATEUR : progression morceau par morceau et apercu via /stream (return_aggregate_stream = False).
Meme squelette que body3d/serverless/handler.py (aide ComfyUI recopiee : chaque worker est autonome).

Entree (contrat transfer.py) :
  {"run_id", "r2": bool,
   "inputs":  {"source":  video source (celle que body3d a suivie : motion/input.mp4),
               "masks":   masks.npz de body3d (meme source ; cadence = params.preset.masks_fps),
               "ref":     image d'identite (PNG/JPEG),
               "ref_mask": option, image noir et blanc (blanc = personne)},
   "outputs": {"video": {put_url, key, max_bytes} | {max_bytes},
               "sheet": idem, option : planche « vu par le modele » (inputs_sheet.png)},
   "outputs_json": {put_url, key} | null,
   "params":  {"preset": {...preset d'animation resolu (app/src/schemas AnimatePreset)...},
               "prompt": str, "negative": str}}

Comme le blueprint officiel, pose_video = la video source brute (SCAIL-2 n'attend pas de carte de pose).
Le worker ne refait pas le suivi : le masque vient de masks.npz.
"""
import glob, json, os, queue, re, shutil, subprocess, tempfile, threading, time, traceback, urllib.error, urllib.request, uuid

import runpod

from transfer import Outputs, fetch_inputs, preview_event, version_event
from workflow import build_workflow, plan, ram_bytes, OUTPUT_FILES, OUTPUT_NAMES

COMFY = os.environ.get("COMFY_URL", "http://127.0.0.1:8188")
COMFY_DIR = os.environ.get("COMFY_DIR", "/comfyui")
JOB_TIMEOUT = int(os.environ.get("SCAIL_JOB_TIMEOUT_S", "3500"))
# Images gardees en RAM par ComfyUI (pose, masque, morceaux decodes, float32) : elles ne doivent pas depasser
# cette part de la limite memoire du conteneur, le reste allant aux poids (offload) et a PyTorch.
IMAGES_RAM_SHARE = float(os.environ.get("SCAIL_IMAGES_RAM_SHARE", "0.30"))
IMAGES_RAM_FALLBACK_GB = float(os.environ.get("SCAIL_IMAGES_RAM_GB", "12"))  # si la limite est illisible


def _http(method, path, payload=None, timeout=30):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(COMFY + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode() or "{}")


def wait_comfy(max_s=600):
    t = time.time()
    while time.time() - t < max_s:
        try:
            _http("GET", "/system_stats", timeout=5)
            return
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
            time.sleep(1)
    raise RuntimeError(f"ComfyUI injoignable sur {COMFY} apres {max_s}s")


def _ws_progress(client_id, q, stop):
    """Relaye les messages de progression ComfyUI (websocket) vers la file q."""
    try:
        import websocket  # websocket-client, present dans worker-comfyui
    except ImportError:
        return
    url = COMFY.replace("http", "ws", 1) + f"/ws?clientId={client_id}"
    try:
        ws = websocket.create_connection(url, timeout=5)
    except Exception:
        return
    ws.settimeout(1)
    while not stop.is_set():
        try:
            msg = ws.recv()
        except Exception:
            continue
        if not isinstance(msg, str):
            continue
        try:
            m = json.loads(msg)
        except ValueError:
            continue
        if m.get("type") in ("progress", "executing"):
            q.put(m)
    ws.close()


def run_workflow(wf):
    """Generateur : soumet le graphe, emet la progression, renvoie (outputs, secondes) via StopIteration."""
    client_id = uuid.uuid4().hex
    q, stop = queue.Queue(), threading.Event()
    th = threading.Thread(target=_ws_progress, args=(client_id, q, stop), daemon=True)
    th.start()
    try:
        try:
            r = _http("POST", "/prompt", {"prompt": wf, "client_id": client_id})
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"ComfyUI a refuse le graphe (HTTP {e.code}) : {e.read().decode()[:1500]}")
        pid = r["prompt_id"]
        t0, last_poll = time.time(), 0.0
        names = {k: v.get("_meta", {}).get("title") or v["class_type"] for k, v in wf.items()}
        while True:
            if time.time() - t0 > JOB_TIMEOUT:
                _http("POST", "/interrupt", {})
                raise RuntimeError(f"timeout ComfyUI ({JOB_TIMEOUT}s)")
            try:
                m = q.get(timeout=1)
                d = m.get("data", {})
                if m["type"] == "executing" and d.get("node"):
                    yield {"type": "progress", "step": names.get(d["node"], d["node"])}
                elif m["type"] == "progress" and d.get("max"):
                    yield {"type": "progress", "step": names.get(d.get("node"), "calcul"),
                           "pct": round(100 * d["value"] / d["max"]), "frame": d["value"]}
            except queue.Empty:
                pass
            if time.time() - last_poll < 2:
                continue
            last_poll = time.time()
            try:
                h = _http("GET", f"/history/{pid}")
            except urllib.error.HTTPError:
                raise
            except urllib.error.URLError as e:
                # processus ComfyUI tue (le plus souvent : limite memoire du conteneur atteinte)
                raise RuntimeError(f"ComfyUI s'est arrete en cours de calcul ({e.reason}) : {oom_report()}") from e
            if pid not in h:
                continue
            st = h[pid].get("status") or {}
            if st.get("status_str") == "error":
                msgs = [m for m in st.get("messages", []) if m and m[0] == "execution_error"]
                detail = msgs[-1][1] if msgs else st
                if isinstance(detail, dict):
                    detail = {k: detail.get(k) for k in ("node_id", "node_type", "exception_message")}
                raise RuntimeError(f"erreur d'execution ComfyUI : {json.dumps(detail, ensure_ascii=False)[:800]}")
            if st.get("completed"):
                return time.time() - t0
    finally:
        stop.set()


def collect(prefix):
    """{role: chemin} des fichiers produits sous output/<prefix>/."""
    base = os.path.join(COMFY_DIR, "output", prefix)
    found = {}
    for role, pattern in OUTPUT_FILES.items():
        hits = sorted(glob.glob(os.path.join(base, pattern)))
        if hits:
            found[role] = hits[-1]
    return found


def ffprobe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
                          "-show_entries", "stream=width,height,r_frame_rate,nb_read_packets",
                          "-of", "json", path], capture_output=True, text=True, timeout=120)
    s = json.loads(out.stdout)["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    return {"width": int(s["width"]), "height": int(s["height"]),
            "fps": float(num) / float(den or 1), "frames": int(s.get("nb_read_packets") or 0)}


def thumb(video, t=1.0):
    """Image JPEG de l'animation pour l'apercu de l'UI (bien sous 1 Mio)."""
    dst = video + ".jpg"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(t), "-i", video, "-frames:v", "1",
                    "-vf", "scale=480:-2", "-q:v", "5", dst], timeout=60, check=True)
    with open(dst, "rb") as f:
        return f.read()


def memory_limit_bytes():
    """Limite memoire du conteneur (cgroup v2 puis v1) ; None si absente ou illisible. (= body3d)"""
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            with open(path) as f:
                v = f.read().strip()
        except OSError:
            continue
        if v.isdigit() and int(v) < 1 << 60:  # « max » ou valeur geante = pas de limite
            return int(v)
    return None


def _cgroup(name):
    for base in ("/sys/fs/cgroup", "/sys/fs/cgroup/memory"):
        try:
            with open(os.path.join(base, name)) as f:
                return f.read()
        except OSError:
            continue
    return None


def oom_report():
    """Pourquoi ComfyUI a disparu : le compteur oom_kill du cgroup tranche (sinon : plantage natif, CUDA...)."""
    events = _cgroup("memory.events") or ""
    kills = next((int(l.split()[1]) for l in events.splitlines() if l.startswith("oom_kill ")), None)
    peak = _cgroup("memory.peak") or _cgroup("memory.max_usage_in_bytes")
    limit = memory_limit_bytes()
    gb = lambda b: f"{int(b) / 1024 ** 3:.1f} Go"
    mem = f"pic {gb(peak.strip())}" if peak and peak.strip().isdigit() else "pic inconnu"
    mem += f", limite {gb(limit)}" if limit else ", limite inconnue"
    if kills:
        return (f"tue par le noyau faute de memoire (oom_kill={kills}, {mem}). Verifier COMFY_EXTRA_ARGS "
                f"(actuel : {os.environ.get('COMFY_EXTRA_ARGS', '')!r}), sinon preset int8 ou resolution plus basse")
    return f"aucun oom_kill dans le cgroup ({mem}) : plantage natif probable (journaux RunPod du worker)"


def _gb(b):
    return round(int(b) / 1024 ** 3, 2) if b not in (None, "") and str(b).strip().isdigit() else None


def probe_report():
    """Sonde (params.probe) : ce que voit le conteneur et ce que voit ComfyUI, SANS charger de modele.
    Sert a choisir COMFY_EXTRA_ARGS / la precision avant de payer un vrai job (README, arbre de decision)."""
    events = _cgroup("memory.events") or ""
    stats = {}
    try:
        stats = _http("GET", "/system_stats", timeout=10)
    except Exception as e:  # ComfyUI pas encore pret : la sonde repond quand meme
        stats = {"error": f"{type(e).__name__}: {e}"}
    disk = {}
    for path in ("/", COMFY_DIR):
        try:
            st = os.statvfs(path)
            disk[path] = {"free_gb": round(st.f_bavail * st.f_frsize / 1024 ** 3, 2), "total_gb": round(st.f_blocks * st.f_frsize / 1024 ** 3, 2)}
        except OSError:
            pass
    system = stats.get("system") or {}
    ram_seen = system.get("ram_total")
    return {
        "cgroup": {"limit_gb": _gb(memory_limit_bytes()), "current_gb": _gb((_cgroup("memory.current") or "").strip()),
                   "peak_gb": _gb((_cgroup("memory.peak") or _cgroup("memory.max_usage_in_bytes") or "").strip()),
                   "events": dict(l.split() for l in events.splitlines() if len(l.split()) == 2)},
        "comfyui": {"version": system.get("comfyui_version"), "pytorch": system.get("pytorch_version"),
                    "argv": system.get("argv"), "ram_total_gb": _gb(ram_seen),
                    # meme formule que comfy/model_management.py (0.34.0), avec la RAM que ComfyUI croit avoir
                    "max_pinned_gb_estime": round(max(0.4 * ram_seen, min(0.9 * ram_seen, ram_seen - 4 * 1024 ** 3)) / 1024 ** 3, 1)
                    if ram_seen and "--disable-pinned-memory" not in (system.get("argv") or []) else 0,
                    "devices": [{k: d.get(k) for k in ("name", "type", "vram_total", "vram_free")} for d in stats.get("devices") or []],
                    "error": stats.get("error")},
        "disk": disk,
        "env": {"COMFY_EXTRA_ARGS": os.environ.get("COMFY_EXTRA_ARGS", "")},
    }


# fp16 : 31 Go de poids + ~8 Go d'encodeurs + activations -> ne tient pas resident sous 48 Go
FP16_MIN_VRAM_GB = float(os.environ.get("SCAIL_FP16_MIN_VRAM_GB", "44"))


def check_vram(preset):
    """Refuse fp16 sur une carte trop petite AVANT de charger 31 Go (le plantage arriverait apres)."""
    try:
        devices = _http("GET", "/system_stats", timeout=10).get("devices") or []
    except Exception:
        return None
    vram = max((d.get("vram_total") or 0 for d in devices), default=0) / 1024 ** 3
    if preset.get("precision", "fp16") == "fp16" and 0 < vram < FP16_MIN_VRAM_GB:
        raise ValueError(f"precision fp16 sur {vram:.0f} Go de VRAM : il en faut {FP16_MIN_VRAM_GB:g} ; utiliser le preset int8")
    return f"VRAM {vram:.0f} Go, precision {preset.get('precision', 'fp16')}"


def check_memory(summary):
    """Refuse AVANT tout calcul GPU un job dont les images feraient sauter le conteneur."""
    need = ram_bytes(summary)
    limit = memory_limit_bytes()
    budget = IMAGES_RAM_SHARE * limit if limit else IMAGES_RAM_FALLBACK_GB * 1024 ** 3
    gb = lambda b: f"{b / 1024 ** 3:.1f} Go"
    info = f"images en RAM {gb(need)}, limite conteneur {gb(limit) if limit else 'inconnue'}, budget {gb(budget)}"
    if need > budget:
        raise ValueError(f"animation trop lourde pour le worker ({info}) : baisser resolution ou fps du preset")
    return info


GEN = re.compile(r"^generation (\d+)/(\d+)$")


def global_progress(events):
    """Le pourcentage de SamplerCustom repart de 0 a chaque morceau : on le ramene a l'ensemble du job
    (5 % -> 95 %, le reste pour la preparation et l'assemblage)."""
    for ev in events:
        m = GEN.match(ev.get("step", "")) if ev.get("type") == "progress" else None
        if m and "pct" in ev:
            k, n = int(m.group(1)), int(m.group(2))
            ev = {**ev, "pct": round(5 + 90 * ((k - 1) + ev["pct"] / 100) / n)}
        yield ev


def global_progress_wrap(gen):
    """yield from conserve la valeur de retour (secondes) de run_workflow."""
    while True:
        try:
            ev = next(gen)
        except StopIteration as stop:
            return stop.value
        yield from global_progress([ev])


def _to_input(path, tag):
    """Copie une entree dans input/ de ComfyUI (LoadVideo / LoadImage n'y lisent que la)."""
    name = f"{tag}_{os.path.basename(path)}"
    shutil.copy(path, os.path.join(COMFY_DIR, "input", name))
    return name


def handler(job):
    inp = job.get("input") or {}
    params = inp.get("params") or {}
    preset = params.get("preset") or {}
    prompt, negative = (params.get("prompt") or "").strip(), (params.get("negative") or "").strip()
    work = tempfile.mkdtemp(prefix="scail_")
    copied = []
    tag = uuid.uuid4().hex[:8]
    prefix = f"fakinflu/{inp.get('run_id') or job.get('id', 'run')}"
    t0 = time.time()
    yield version_event("scail")  # toujours en premier : le client verifie la version de l'image
    if params.get("probe"):
        try:
            outputs = Outputs(inp)
            wait_comfy(120)
            report = probe_report()
            print(f"[scail] sonde : {json.dumps(report)}", flush=True)
            path = os.path.join(work, "probe.json")
            with open(path, "w") as f:
                json.dump(report, f, indent=2)
            yield from outputs.emit("probe", path, "probe.json")
            yield {"type": "done", "seed": None, "timings": {"total_s": round(time.time() - t0, 1)}}
        except Exception as e:
            traceback.print_exc()
            yield {"type": "error", "message": f"{type(e).__name__}: {e}"}
        finally:
            shutil.rmtree(work, ignore_errors=True)
        return
    outputs, sheet_sent = None, False
    try:
        outputs = Outputs(inp)
        paths, events = fetch_inputs(inp, os.path.join(work, "in"), outputs)
        yield from events
        missing = [r for r in ("source", "masks", "ref") if r not in paths]
        if missing:
            raise ValueError(f"entrees manquantes : {missing}")
        if not prompt:
            raise ValueError("prompt vide : SCAIL-2 a besoin de la description de la DA")
        meta = ffprobe(paths["source"])
        summary = plan(preset, meta)
        mem_info = check_memory(summary)
        print(f"[scail] {mem_info} ; COMFY_EXTRA_ARGS={os.environ.get('COMFY_EXTRA_ARGS', '')!r}", flush=True)
        yield {"type": "progress", "step": mem_info, "pct": 1}
        yield {"type": "progress", "pct": 2,
               "step": f"source {meta['width']}x{meta['height']} {meta['fps']:.2f} fps, {meta['frames']} images -> "
                       f"{summary['width']}x{summary['height']} a {summary['fps']:g} fps, "
                       f"{summary['segments']} morceau(x), {summary['frames_out']} images"}

        files = {"source": paths["source"], "masks": paths["masks"], "ref": _to_input(paths["ref"], tag)}
        if "ref_mask" in paths:
            files["ref_mask"] = _to_input(paths["ref_mask"], tag)
        copied = [os.path.join(COMFY_DIR, "input", files[k]) for k in ("ref", "ref_mask") if k in files]
        wait_comfy()
        vram_info = check_vram(preset)
        if vram_info:
            yield {"type": "progress", "step": vram_info, "pct": 3}
        wf = build_workflow(preset, prompt, negative, files, prefix, meta)
        seconds = yield from global_progress_wrap(run_workflow(wf))

        produced = collect(prefix)
        if "sheet" in produced and "sheet" in outputs.specs:
            yield from outputs.emit("sheet", produced["sheet"], OUTPUT_NAMES["sheet"])
            sheet_sent = True
        if "video" not in produced:
            raise RuntimeError(f"aucune video produite (fichiers : {sorted(produced)})")
        yield preview_event(thumb(produced["video"]), "image/jpeg")
        yield from outputs.emit("video", produced["video"], OUTPUT_NAMES["video"])
        outputs.write_summary()
        yield {"type": "done", "seed": preset.get("seed"),
               "timings": {"comfy_s": round(seconds, 1), "total_s": round(time.time() - t0, 1),
                           "segments": summary["segments"], "frames_out": summary["frames_out"]}}
    except Exception as e:
        traceback.print_exc()
        sheet = collect(prefix).get("sheet")  # la planche aide surtout quand le job echoue
        if sheet and outputs and "sheet" in outputs.specs and not sheet_sent:
            try:
                yield from outputs.emit("sheet", sheet, OUTPUT_NAMES["sheet"])
            except Exception:
                traceback.print_exc()
        yield {"type": "error", "message": f"{type(e).__name__}: {e}"}
    finally:
        shutil.rmtree(work, ignore_errors=True)
        shutil.rmtree(os.path.join(COMFY_DIR, "output", prefix), ignore_errors=True)
        for p in copied:
            if os.path.exists(p):
                os.remove(p)


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler, "return_aggregate_stream": False})
