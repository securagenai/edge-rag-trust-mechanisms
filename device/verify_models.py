"""Verify SHA-256 digests of the GGUF files against pins.json. Exits non-zero on any mismatch or missing file."""
import hashlib, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent; P = json.load(open(ROOT / "pins.json"))
names = sys.argv[1:] or list(P["gguf_sha256"])
bad = 0
for n in names:
    f = ROOT / "models" / n
    if n not in P["gguf_sha256"]: print("UNPINNED", n); bad += 1; continue
    if not f.exists(): print("MISSING", n); bad += 1; continue
    h = hashlib.sha256()
    with open(f, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""): h.update(chunk)
    ok = h.hexdigest() == P["gguf_sha256"][n]; bad += (not ok)
    print(("OK      " if ok else "MISMATCH"), n)
sys.exit(1 if bad else 0)
