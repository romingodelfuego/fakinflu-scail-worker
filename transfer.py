"""fakinflu — transferts de fichiers cote worker (plan motion 0.2). Stdlib uniquement.

Fichier CANONIQUE : r2smoke/serverless/transfer.py. Chaque worker en embarque une copie
identique (verifie par la CI : make transfer-lib-check). Meme regle de routage que le
client TypeScript (app/src/transfer/routing.ts).

Le worker n'a AUCUNE cle R2 : il ne recoit que des URL pre-signees (GET pour les entrees,
PUT pour les sorties). Ne jamais journaliser une URL : c'est un jeton au porteur.

Entree d'un fichier  : {"b64", "name", "bytes", "sha256"}  ou  {"url", "name", "bytes", "sha256"}
Sortie attendue      : {"put_url", "key", "max_bytes"}     ou  {"max_bytes"} (sans R2)
Evenements produits  : version (toujours le premier), file (inline | r2), chunk / chunk_end (repli, 1 Mio),
                       input_read, preview (<= 1 Mio).
"""
import base64, hashlib, json, os, urllib.request

INLINE_MAX_B64 = 7 * 1024 * 1024
# Mesure du 2026-10-08 : RunPod SUPPRIME SANS ERREUR un evenement /stream de ~2,7 Mo serialise
# (1,87 Mo passe). Morceaux et apercus plafonnes a 1 Mio brut (~1,4 Mo en base64).
CHUNK_BYTES = 1024 * 1024
MAX_EVENT_RAW = 1024 * 1024
_IO = 1024 * 1024


def b64_len(n):
    return 4 * ((n + 2) // 3)


def route_for(n, r2_available, direction="input"):
    """input : dans le corps de /run (< 7 Mo en base64). output : dans un evenement /stream (<= 1 Mio)."""
    small = b64_len(n) < INLINE_MAX_B64 if direction == "input" else n <= MAX_EVENT_RAW
    if small:
        return "inline"
    return "r2" if r2_available else "stream_fallback"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(_IO), b""):
            h.update(b)
    return h.hexdigest()


def _safe(name):
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in os.path.basename(name or "file"))


def version_event(worker):
    """Premier evenement de chaque job : version de l'image (fichier VERSION ecrit par
    `make worker-release` a cote de transfer.py). Le client annule le job si elle ne correspond
    pas a celle attendue (worker.json) : on ne paie pas un job sur une ancienne image."""
    try:
        with open(os.environ.get("WORKER_VERSION_FILE") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "VERSION")) as f:
            version = f.read().strip() or "dev"
    except OSError:
        version = "dev"  # build local, hors release
    return {"type": "version", "worker": worker, "version": version}


def preview_event(data, mime, **extra):
    """Apercu leger pour l'UI ; None (et rien d'emis) s'il depasse la taille sure d'un evenement."""
    if len(data) > MAX_EVENT_RAW:
        return {"type": "progress", "step": f"apercu ignore ({len(data)} octets > {MAX_EVENT_RAW})"}
    return {"type": "preview", "mime": mime, "data": base64.b64encode(data).decode(), **extra}


def fetch_input(role, spec, dest_dir):
    """Ecrit l'entree dans dest_dir et verifie taille + empreinte. Renvoie (chemin, evenements)."""
    os.makedirs(dest_dir, exist_ok=True)
    path = os.path.join(dest_dir, _safe(spec.get("name") or role))
    h = hashlib.sha256()
    events = []
    if "b64" in spec:
        data = base64.b64decode(spec["b64"])
        h.update(data)
        with open(path, "wb") as f:
            f.write(data)
    elif "url" in spec:
        with urllib.request.urlopen(spec["url"], timeout=600) as r, open(path, "wb") as f:
            for b in iter(lambda: r.read(_IO), b""):
                h.update(b)
                f.write(b)
        events.append({"type": "input_read", "role": role})  # le client peut supprimer l'objet
    else:
        raise ValueError(f"entree {role} : ni b64 ni url")
    size = os.path.getsize(path)
    if spec.get("bytes") is not None and size != spec["bytes"]:
        raise ValueError(f"entree {role} : {size} octets recus, {spec['bytes']} annonces")
    if spec.get("sha256") and h.hexdigest() != spec["sha256"]:
        raise ValueError(f"entree {role} : empreinte sha256 differente")
    return path, events


def _put(url, path, size):
    with open(path, "rb") as f:
        req = urllib.request.Request(url, data=f, method="PUT",
                                     headers={"Content-Length": str(size), "Content-Type": "application/octet-stream"})
        with urllib.request.urlopen(req, timeout=1800) as r:
            if r.status not in (200, 201):
                raise RuntimeError(f"PUT -> HTTP {r.status}")


def _chunks(role, path, size, digest, name):
    total = max(1, (size + CHUNK_BYTES - 1) // CHUNK_BYTES)
    with open(path, "rb") as f:
        for i in range(total):
            b = f.read(CHUNK_BYTES)
            yield {"type": "chunk", "role": role, "index": i, "total": total, "name": name,
                   "sha256": hashlib.sha256(b).hexdigest(), "data": base64.b64encode(b).decode()}
    yield {"type": "chunk_end", "role": role, "sha256": digest, "bytes": size, "done": True}


class Outputs:
    """Emet les sorties selon la regle de routage et tient le recapitulatif outputs.json."""

    def __init__(self, job_input):
        self.run_id = job_input.get("run_id", "")
        self.specs = job_input.get("outputs") or {}
        self.summary_spec = job_input.get("outputs_json")
        self.r2 = bool(job_input.get("r2"))
        self.files = []
        self.inputs_read = []

    def emit(self, role, path, name=None):
        """Generateur d'evenements pour une sortie definitive."""
        spec = self.specs.get(role, {})
        size = os.path.getsize(path)
        digest = sha256_file(path)
        name = name or os.path.basename(path)
        max_bytes = spec.get("max_bytes")
        if max_bytes is not None and size > max_bytes:
            # PUT pre-signe ne borne pas la taille cote R2 : c'est le worker qui refuse (plan 0.6)
            raise ValueError(f"sortie {role} : {size} octets > {max_bytes} annonces")
        route = route_for(size, self.r2 and "put_url" in spec, "output")
        if route == "inline":
            with open(path, "rb") as f:
                data = base64.b64encode(f.read()).decode()
            yield {"type": "file", "role": role, "route": "inline", "bytes": size, "sha256": digest, "name": name, "data": data}
            return
        if route == "r2":
            try:
                _put(spec["put_url"], path, size)
                self.files.append({"role": role, "key": spec["key"], "bytes": size, "sha256": digest})
                yield {"type": "file", "role": role, "route": "r2", "key": spec["key"], "bytes": size, "sha256": digest, "name": name}
                return
            except Exception as e:  # URL expiree, R2 injoignable... -> repli sans cout
                yield {"type": "progress", "step": f"R2 indisponible pour {role} ({type(e).__name__}), repli en morceaux"}
        yield from _chunks(role, path, size, digest, name)

    def write_summary(self):
        """Depose runs/<run_id>/outputs.json : permet la reprise si le client a manque le stream."""
        if not self.summary_spec or not self.files:
            return
        body = json.dumps({"run_id": self.run_id, "files": self.files, "inputs_read": self.inputs_read}).encode()
        req = urllib.request.Request(self.summary_spec["put_url"], data=body, method="PUT",
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=120).close()


def fetch_inputs(job_input, dest_dir, outputs=None):
    """Recupere toutes les entrees ; renvoie ({role: chemin}, evenements input_read)."""
    paths, events = {}, []
    for role, spec in (job_input.get("inputs") or {}).items():
        paths[role], ev = fetch_input(role, spec, dest_dir)
        events += ev
        if ev and outputs is not None:
            outputs.inputs_read.append(role)
    return paths, events
