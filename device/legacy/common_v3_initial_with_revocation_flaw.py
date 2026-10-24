"""Shared helpers for the v3 experiments (update security, signed poisoning, entailment verifier)."""
import json, hashlib, time, shutil, subprocess, threading, re
from pathlib import Path
import numpy as np
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization
from cryptography.exceptions import InvalidSignature

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results_v3"; OUT.mkdir(exist_ok=True)
KEYS = ROOT / "keys_v3"; KEYS.mkdir(exist_ok=True)

def sha256(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def load_or_make_key(name):
    f = KEYS / f"{name}.pem"
    if name == "publisher" and (ROOT / "signing_key.pem").exists():
        return serialization.load_pem_private_key((ROOT / "signing_key.pem").read_bytes(), password=None)
    if f.exists():
        return serialization.load_pem_private_key(f.read_bytes(), password=None)
    k = Ed25519PrivateKey.generate()
    f.write_bytes(k.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    return k

def key_id(sk):
    raw = sk.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return hashlib.sha256(raw).hexdigest()[:16]

def write_pack(dst, src_files, manifest_fields, sk, sign_files_from=None):
    """Create a pack directory: copy files, write manifest (with hashes) and detached Ed25519 signature."""
    dst = Path(dst); shutil.rmtree(dst, ignore_errors=True); dst.mkdir(parents=True)
    for name, src in src_files.items(): shutil.copy(src, dst / name)
    hash_dir = Path(sign_files_from) if sign_files_from else dst
    man = dict(manifest_fields)
    man["files"] = {name: sha256(hash_dir / name) for name in src_files}
    man["key_id"] = key_id(sk)
    mb = json.dumps(man, sort_keys=True).encode()
    (dst / "manifest.json").write_bytes(mb); (dst / "manifest.sig").write_bytes(sk.sign(mb))
    return dst

def trusted_keys():
    """Device trust store: publisher keys known to the device, with revocation status."""
    return json.load(open(KEYS / "trust_store.json"))

def _pub(pem): return serialization.load_pem_public_key(pem.encode())

def verify_signature_only(pack):
    """Baseline verifier: signature by any known key + file hashes (as in the v1/v2 experiments)."""
    pack = Path(pack); mb = (pack / "manifest.json").read_bytes(); sig = (pack / "manifest.sig").read_bytes()
    man = json.loads(mb); ts = trusted_keys()
    ok_sig = False
    for kid, entry in ts["keys"].items():
        try: _pub(entry["public_pem"]).verify(sig, mb); ok_sig = True; break
        except InvalidSignature: pass
    if not ok_sig: return False, "bad signature"
    for f, h in man["files"].items():
        if sha256(pack / f) != h: return False, f"hash mismatch: {f}"
    return True, "accepted"

def verify_layered(pack, state, device):
    """Layered verifier: signature + hashes + key revocation + monotonic sequence + expiry + compatibility."""
    ok, why = verify_signature_only(pack)
    if not ok: return False, why
    man = json.loads((Path(pack) / "manifest.json").read_bytes()); ts = trusted_keys()
    entry = ts["keys"].get(man.get("key_id"))
    if entry is None or entry.get("revoked"): return False, "signing key revoked or unknown"
    if man.get("pack_id") != state["pack_id"]: return False, "wrong pack_id"
    if int(man.get("sequence", -1)) <= int(state["installed_sequence"]): return False, "rollback/replay: sequence not newer"
    exp = man.get("expires_utc")
    if exp and time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) > exp: return False, "expired"
    if man.get("embedding_model") != device["embedding_model"] or int(man.get("dim", -1)) != device["dim"]:
        return False, "incompatible: embedding model/dimension"
    return True, "accepted"

class Power(threading.Thread):
    """Approximate board power from the Raspberry Pi 5 PMIC (vcgencmd)."""
    def __init__(s):
        super().__init__(daemon=True); s.samples = []; s.run_ = True
        s.ok = shutil.which("vcgencmd") is not None and s.read() is not None
    def read(s):
        try: out = subprocess.run(["vcgencmd", "pmic_read_adc"], capture_output=True, text=True, timeout=2).stdout
        except Exception: return None
        cur, volt = {}, {}
        for n, v, u in re.findall(r"(\S+)\s+\w+\(\d+\)=([\d.]+)([AV])", out):
            (cur if u == "A" else volt)[n[:-2]] = float(v)
        w = sum(cur[k] * volt[k] for k in cur if k in volt); return w if w > 0 else None
    def run(s):
        while s.run_:
            w = s.read()
            if w is not None: s.samples.append((time.time(), w))
            time.sleep(0.5)
    def mean(s, a, b):
        v = [w for t, w in s.samples if a <= t <= b]; return float(np.mean(v)) if v else None
