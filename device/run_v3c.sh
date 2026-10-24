#!/usr/bin/env bash
# v3c: corrected layered verifier + mismatched-signer test + model provenance (~2-5 min; no network or sudo needed).
#   bash run_v3c.sh > run_v3c.log 2>&1
set -euo pipefail
cd "$(dirname "$0")"; source venv/bin/activate
python build_v3c.py
python update_security.py
python provenance.py
echo "ALL DONE V3C $(date)"
