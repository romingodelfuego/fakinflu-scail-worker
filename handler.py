#!/usr/bin/env python3
"""fakinflu — worker scail : animation SCAIL-2 (Wan 2.1 14B via ComfyUI). Plan motion, Phase 2.

Remplace le /handler.py de runpod/worker-comfyui ; /start.sh lance ComfyUI (127.0.0.1:8188).
Handler GENERATEUR : progression morceau par morceau et apercu via /stream (return_aggregate_stream = False).
Meme squelette que body3d/serverless/handler.py (aide ComfyUI recopiee : chaque worker est autonome).

Entree (contrat transfer.py) :
  {"run_id", "r2": bool,
   "inputs":  {"control": video de controle body3d (control_pose.mp4),
               "masks":   masks.npz de body3d (meme run que control),
               "ref":     image d'identite (PNG/JPEG),
               "ref_mask": option, image noir et blanc (blanc = personne)},
   "outputs": {"video": {put_url, key, max_bytes} | {max_bytes}},
   "outputs_json": {put_url, key} | null,
   "params":  {"preset": {...preset d'animation resolu (app/src/schemas AnimatePreset)...},
               "prompt": str, "negative": str}}

La video de controle arrive du PC (decision 7 du plan) : le worker ne refait ni le rendu ni le suivi.
"""
import glob, json, os, queue, shutil, subprocess, tempfile, threading, time, traceback, urllib.error, urllib.request, uuid

import runpod

from transfer import Outputs, fetch_inputs, preview_event, version_event
from workflow import build_workflow, plan, OUTPUT_FILES, OUTPUT_NAMES

COMFY = os.environ.get("COMFY_URL", "http://127.0.0.1:8188")
COMFY_DIR = os.environ.get("COMFY_DIR", "/comfyui")
JOB_TIMEOUT = int(os.environ.get("SCAIL_JOB_TIMEOUT_S", "3500"))


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
            h = _http("GET", f"/history/{pid}")
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
    try:
        outputs = Outputs(inp)
        paths, events = fetch_inputs(inp, os.path.join(work, "in"), outputs)
        yield from events
        missing = [r for r in ("control", "masks", "ref") if r not in paths]
        if missing:
            raise ValueError(f"entrees manquantes : {missing}")
        if not prompt:
            raise ValueError("prompt vide : SCAIL-2 a besoin de la description de la DA")
        meta = ffprobe(paths["control"])
        summary = plan(preset, meta)
        yield {"type": "progress", "pct": 2,
               "step": f"controle {meta['width']}x{meta['height']} {meta['fps']:.2f} fps, {meta['frames']} images -> "
                       f"{summary['width']}x{summary['height']} a {summary['fps']:g} fps, "
                       f"{summary['segments']} morceau(x), {summary['frames_out']} images"}

        files = {"control": _to_input(paths["control"], tag), "ref": _to_input(paths["ref"], tag), "masks": paths["masks"]}
        if "ref_mask" in paths:
            files["ref_mask"] = _to_input(paths["ref_mask"], tag)
        copied = [os.path.join(COMFY_DIR, "input", files[k]) for k in ("control", "ref", "ref_mask") if k in files]
        wait_comfy()
        wf = build_workflow(preset, prompt, negative, files, prefix, meta)
        seconds = yield from run_workflow(wf)

        produced = collect(prefix)
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
        yield {"type": "error", "message": f"{type(e).__name__}: {e}"}
    finally:
        shutil.rmtree(work, ignore_errors=True)
        shutil.rmtree(os.path.join(COMFY_DIR, "output", prefix), ignore_errors=True)
        for p in copied:
            if os.path.exists(p):
                os.remove(p)


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler, "return_aggregate_stream": False})
