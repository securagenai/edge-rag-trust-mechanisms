"""Adds the mismatched-signer case: manifest declares the CURRENT publisher key id but is signed with the REVOKED key."""
import json, shutil, time
from pathlib import Path
from common_v3 import ROOT, KEYS, load_or_make_key, key_id, sha256
pub, old = load_or_make_key("publisher"), load_or_make_key("old_publisher")
src = ROOT / "packs_v3" / "valid_newer"; dst = ROOT / "packs_v3" / "revoked_key_claims_current_id"
shutil.rmtree(dst, ignore_errors=True); dst.mkdir(parents=True)
for f in ["corpus.json", "index.hnsw"]: shutil.copy(src / f, dst / f)
man = json.loads((src / "manifest.json").read_bytes()); man["key_id"] = key_id(pub)   # claims current key
mb = json.dumps(man, sort_keys=True).encode()
(dst / "manifest.json").write_bytes(mb); (dst / "manifest.sig").write_bytes(old.sign(mb))  # signed by revoked key
print("built", dst.name)
