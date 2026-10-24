#!/usr/bin/env bash
# Regenerates every reported table value from the released per-query outputs in ../data.
# Usage (from the repository root):   bash analysis/reproduce_all.sh
set -euo pipefail
cd "$(dirname "$0")"; D="$(cd ../data && pwd)"
python analyze_v2.py   "$D"   # Table 4 inputs, retrieval AUROC, gate threshold  -> data/results_v2/summary_v2.json
python final_analysis.py "$D" # Table 4 policies, Figs. 3-4 (if matplotlib)    -> data/results_v2/final_summary.json
python analyze_v3b.py  "$D"   # Table 5 (verifier), poisoning (Sect. 5.5)       -> data/results_v3/summary_v3b.json
python paired_stats.py "$D"   # paired bootstrap intervals, output breakdowns  -> data/results_v3/paired_stats.json
python heldout_eval.py "$D"   # held-out evaluation of M+V+G and M+V+G+E       -> data/results_v3/heldout_eval.json
echo "Table 6 (update security) is read directly from data/results_v3/update_security_v3c.json"
echo "REPRODUCTION DONE"
