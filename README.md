# Code and data: Trust Mechanisms for Offline Generative AI at the Edge

Release accompanying the paper *"Trust Mechanisms for Offline Generative AI at the Edge: A Connectivity-Regime Framework and an Empirical Study on a Raspberry Pi 5"* (Feretzakis, Papaspyridis).

## 1. Regenerate all reported results from the released outputs (no hardware needed)
Requirements: Python 3.10+, `numpy`; `matplotlib` is optional (figures only).
```bash
bash analysis/reproduce_all.sh
```
This rewrites, with identical values, the files below from the per-query outputs in `data/`:

| Paper element | Script | Output |
|---|---|---|
| Sect. 5.1 (retrieval, gate), Table 4 inputs | `analysis/analyze_v2.py` | `data/results_v2/summary_v2.json` |
| Table 4 (policies), Figs. 3-4 | `analysis/final_analysis.py` | `data/results_v2/final_summary.json`, `Fig3.pdf`, `Fig4.pdf` |
| Table 5 (entailment verifier), Sect. 5.5 (poisoning) | `analysis/analyze_v3b.py` | `data/results_v3/summary_v3b.json` |
| Paired bootstrap intervals, output breakdowns | `analysis/paired_stats.py` | `data/results_v3/paired_stats.json` |
| Held-out evaluation of M+V+G and M+V+G+E | `analysis/heldout_eval.py` | `data/results_v3/heldout_eval.json` |
| Table 6 (update security) | measured on the device | `data/results_v3/update_security_v3c.json` |

Notes on validation scope: paired bootstrap intervals resample questions (not articles) and keep the fitted thresholds fixed (they exclude threshold-estimation uncertainty); Tables 4-5 apply mean thresholds to the full sample, and `heldout_eval.py` re-fits both thresholds in each of 2,000 splits. When the split-conformal rank floor(alpha(n+1)) is below 1, the threshold is set to a value that rejects nothing (0 for NLI probabilities, -1 for cosine scores); this affects the NLI calibration of the 1.5B configuration only.

## 2. Reproduce the hardware experiment (Raspberry Pi 5, 16 GB)
Section 1 (regenerating every table from the released logs) has been tested end to end. A new hardware run is a separate, longer procedure.

Copy the contents of `device/` into one directory on the device (e.g. `~/edge_rag_experiment`); all device scripts expect to run from that single directory. By default the scripts run in **strict reproduction mode** (`REPRO=1`): `setup_pi.sh` checks out the pinned llama.cpp commit and installs the exact Python package versions in `requirements_pi.txt` (PyTorch 2.14.1, CPU); BGE-M3 and the NLI checkpoint are loaded at the Hugging Face revisions in `pins.json`; and `verify_models.py` checks the SHA-256 of every GGUF file and aborts on a mismatch. No alternative model provider or NLI checkpoint is used: strict mode selects only the official 7B shards (an alternative GGUF already on disk is ignored), aborts if any pinned artefact is missing or fails its hash check, rejects any unpinned Hugging Face model, stops with a non-zero exit if the NLI checkpoint cannot be loaded, and rebuilds llama.cpp from the pinned commit unless the existing binary was built by `setup_pi.sh` from that commit on a clean tracked tree and still has the SHA-256 recorded at build time (the `llama.cpp/` folder is a disposable dependency checkout and is replaced on rebuild). `bash device/tests/test_strict_mode.sh` checks 25 assertions on these paths offline with synthetic fixtures and exits non-zero if any fails. Setting `REPRO=0` re-enables the exploratory fallbacks used during development (smaller embedding model on low-RAM devices, alternative 7B GGUF source, alternative NLI checkpoint). Pinning was added after the original run and has not itself been re-executed on hardware; the recorded hashes and revisions are those of the original run.
1. `bash setup_pi.sh` - system packages, pinned llama.cpp build, pinned Python environment, SQuAD 2.0 dev, BGE-M3 and Qwen2.5-3B (hash-checked).
2. `setsid nohup bash run_v2.sh > run_v2.log 2>&1 &` - main study: downloads the 1.5B and 7B models (hash-checked), builds the signed pack, runs the five configurations inside a network namespace (needs a NOPASSWD sudoers rule for `/usr/bin/unshare` only), then analysis (~12 h).
3. `setsid nohup bash run_v3.sh > run_v3.log 2>&1 &` - variant packs, signed poisoning, first (superseded) NLI verifier.
4. `bash run_v3b.sh` - final NLI verifier with question-to-declarative rewriting and span check.
5. `bash run_v3c.sh` - corrected layered verifier, all update-security tests (incl. the mismatched-signer case), model provenance.
Then copy `results_v2/`, `results_v3/`, `questions_v2.json`, `pack_v2_build_stats.json` into `data/` and run Section 1.

## 3. Pinned artefacts
`data/results_v3/model_provenance.json` records SHA-256 digests of all GGUF files and the Hugging Face revisions:
- Qwen2.5-Instruct GGUF (official `Qwen/Qwen2.5-*-Instruct-GGUF`, Q4_K_M): 1.5B `6a1a2eb6...e9407e`, 3B `626b4a66...15c62d`, 7B (2 shards) `dfce12e3...0580db`, `539cf93f...f3d72a`
- `BAAI/bge-m3` revision `5617a9f61b028005a4858fdac845db406aefb181`
- `cross-encoder/nli-deberta-v3-base` revision `6c749ce3425cd33b46d187e45b92bbf96ee12ec7`
Python package versions on the device: see `data/results_pilot_v1/pip_freeze.txt` (main study) and `data/results_v3/pip_freeze_v3.txt` (v3 experiments).

## 4. Legacy and corrections (for transparency)
- `device/nli_verify.py` and `data/results_v3/nli_gen_*.jsonl`, `summary_v3.json`: first NLI verifier with a flawed hypothesis format (question + "The answer is ..."); superseded by `nli_verify_v2.py` (`nli2_*`).
- `device/legacy/common_v3_initial_with_revocation_flaw.py`: initial layered verifier, which verified the signature with any trusted key but checked revocation against the key identifier declared in the manifest; it therefore accepted a manifest signed with a revoked key that declared the current key's identifier. `data/results_v3/update_security.json` holds its results; the corrected verifier (`device/common_v3.py`) and `update_security_v3c.json` are the ones reported in the paper.
- `data/results_pilot_v1/`: the earlier 210-query pilot (3B model only).

## Licences
- **Code:** the original code in this repository is released under the MIT licence (see `LICENSE`; copyright holder: securagen.ai P.C.). The MIT licence does not relicense any third-party material listed below.
- **Data:** SQuAD 2.0 is distributed under CC BY-SA 4.0; the derived passages, questions and per-query outputs in `data/` follow that licence.
- **Models (not redistributed):** the Qwen2.5-Instruct GGUF files, BGE-M3 and the NLI cross-encoder are downloaded from their Hugging Face pages and remain governed by their own licences (see each model card; for example, Qwen2.5-3B-Instruct is released under the Qwen Research licence, while the 1.5B and 7B models are released under Apache-2.0).
- **Software dependencies:** llama.cpp and the Python packages in `device/requirements_pi.txt` are used under their respective licences.

## Citation
If you use this code or data, please cite the paper (see `CITATION.cff`):

Feretzakis G, Papaspyridis A (2026) Trust mechanisms for offline generative AI at the edge: a connectivity-regime framework and an empirical study on a Raspberry Pi 5. Preprint.
