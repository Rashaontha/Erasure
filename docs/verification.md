# Verification record

This file records actual checks. It does not report research measurements.

## Executed on 2026-11-10 (macOS arm64, 8 GB RAM)

- Python 3.12 environment installed from `pyproject.toml`.
- `python -m compileall -q src scripts tests`: passed.
- Imports of every `palak_repro` module: passed.
- `python -m pytest -q`: 3 passed.
- Hugging Face's `hf-internal-testing/tiny-stable-diffusion-pipe` loaded on CPU;
  a 64x64, two-step image was generated.
- On that tiny model, EEBshallow injection, UCE erasure, EEBdeep one-step
  optimization, ESD one-step optimization, checkpoint saving and reloading,
  and condition image generation completed. These settings are explicitly
  reduced and the model is not Stable Diffusion v1.4.
- Tiny-model UCE edit also completed on MPS with CPU linear solve.
- `palak-experiments --config configs/cpu_smoke.yaml --stage generate
  --target bird --attack none --erasure clean`: completed.
- Final offline CPU smoke generated 11 new tiny-model images in
  `results/audit_smoke/`; no network download was started.
- Final `python -m pytest -q`: 3 passed; final syntax and package import
  checks passed.
- Report stage generated six rows with `NOT MEASURED` ASR values when there
  were no evaluated conditions.
- The classifier architecture was compared against Phan's source: all 122
  state-dict tensor names and shapes match.

## Unexecuted as of this record

- Stable Diffusion v1.4 baseline and full matrix: the fp16 download was
  intentionally stopped before completion at the user's request.
- All SD 1.4 research metrics: **NOT MEASURED**.
- Phan pretrained classifier inference: its 979 MB weights archive has not
  finished downloading in this environment. The original Box URL returns 404;
  the Google Drive mirror is documented in `scripts/download_classifier.py`.
- COCO FID and CLIPScore: the COCO 2014 data have not been acquired here.

The paper's reference numbers in the comparison CSV are not results of this
project. The tiny-model output is an integration test only.
