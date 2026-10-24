"""Phase A (offline): signature-only vs layered verification of the variant packs."""
import json, time, numpy as np
from common_v3 import ROOT, OUT, KEYS, verify_signature_only, verify_layered
state = json.load(open(KEYS / "device_state.json")); dev = json.load(open(KEYS / "device_profile.json"))
EXPECT = {"valid_newer": True, "rollback_older": False, "replay_same_version": False, "expired": False,
          "incompatible_embedding": False, "unknown_signer": False, "revoked_key": False,
          "tampered_passage": False, "mixed_version_index": False, "poisoned_signed": False}
rows = []
for case, should_accept in EXPECT.items():
    p = ROOT / "packs_v3" / case
    a1, r1 = verify_signature_only(p); a2, r2 = verify_layered(p, state, dev)
    t = []
    for _ in range(20):
        t0 = time.perf_counter(); verify_layered(p, state, dev); t.append((time.perf_counter() - t0) * 1e3)
    rows.append({"case": case, "should_accept": should_accept, "signature_only": a1, "signature_only_reason": r1,
                 "layered": a2, "layered_reason": r2, "layered_ms_median": float(np.median(t))})
    print(f"{case:24s} sig-only={'ACCEPT' if a1 else 'reject':7s} layered={'ACCEPT' if a2 else 'reject':7s} ({r2})")
json.dump(rows, open(OUT / "update_security.json", "w"), indent=1)
