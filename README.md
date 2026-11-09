# Persistence of Backdoor Triggers After Concept Erasure

This repository implements the benign CIFAR-10 object subset of the Assignment 1
proposal. It asks whether a trigger mapped to an object remains effective after
ESD or UCE concept erasure of Stable Diffusion v1.4. It compares EEBshallow
(closed-form cross-attention edits) and EEBdeep (LoRA score self-distillation)
against clean controls. The three configured targets are bird, ship, and dog;
the latter two are configurable choices because the proposal does not fix them.

## Setup

Use Python 3.10-3.12, a CUDA GPU with at least 24 GB VRAM for practical full
runs, and enough disk space for SD 1.4, teacher/student models, checkpoints,
COCO images, and generated images. CPU is supported for logic and very slow
smoke runs. MPS is experimental. PyTorch is installed via `pyproject.toml`; for
a particular CUDA build, install the matching official PyTorch wheel first.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
huggingface-cli login  # only if model access requires authentication
python scripts/download_classifier.py
```

The base model ID is `CompVis/stable-diffusion-v1-4` in the YAML file. Its model
weights download on first use. Download Phan's pretrained CIFAR-10 ResNet-18
state dict before evaluation. See [data/README.md](data/README.md) for COCO 2014
download and directory layout. Model and dataset terms remain the user's
responsibility to review before running.

## Run

Edit `configs/experiment.yaml` to set targets, sample counts, device, and
hyperparameters. The default is the proposal's 100 target, 100 trigger, and ten
per retention class images per condition. COCO defaults to 100 image/caption
pairs; the original paper used a much larger set. Default EEBdeep and ESD each
run 1000 optimization steps. Complete runs require significant GPU time.

```bash
palak-experiments --config configs/experiment.yaml
# Equivalent entry point:
python scripts/run_experiments.py --config configs/experiment.yaml

# Individual stages; filters restrict to a condition:
palak-experiments --config configs/experiment.yaml --stage generate --target bird --attack none --erasure clean
palak-experiments --config configs/experiment.yaml --stage evaluate --target bird --attack none --erasure clean
palak-experiments --config configs/experiment.yaml --stage inject --target bird --attack shallow
palak-experiments --config configs/experiment.yaml --stage erase --target bird --attack shallow --erasure uce
palak-experiments --config configs/experiment.yaml --stage generate --target bird --attack shallow --erasure uce
palak-experiments --config configs/experiment.yaml --stage evaluate --target bird --attack shallow --erasure uce
palak-experiments --config configs/experiment.yaml --stage report
```

The pipeline skips complete checkpoint files and generated image files so it can
resume. JSON manifests retain prompts, seeds, references, and image paths.
`results/tables/` contains measurements and a paper comparison CSV;
`results/plots/` contains bar charts for metrics with measured values;
`results/images/` contains the images and comparison grids. No plot is produced
for a metric without measurements.

For a lightweight CPU configuration, use `configs/cpu_smoke.yaml`. It uses the
tiny test pipeline, one-step training, and two-step sampling solely to check
execution. It cannot support research conclusions. Install dependencies and run:

```bash
palak-experiments --config configs/cpu_smoke.yaml --stage generate --target bird --attack none --erasure clean
python -m pytest -q
```

## Architecture and evaluation

`src/palak_repro/closed_form.py` implements UCE and EEBshallow;
`distill.py` implements EEBdeep and ESD; `experiment.py` orchestrates stages;
`metrics.py` calculates ASR, target and retention accuracy, FID, and CLIPScore;
`report.py` creates CSV and plots. The six erased conditions per target are
clean/ESD, clean/UCE, shallow/ESD, shallow/UCE, deep/ESD, deep/UCE; the clean
unerased baseline is also measured. The full method and numerical deviations
are in [docs/methodology.md](docs/methodology.md).

The paper's published ASR percentages are stored as reference values only.
`measured_asr` is `NOT MEASURED` until an actual condition is evaluated. Report mean,
sample standard deviation, and target count where multiple targets complete.
The classifier's ten-way outputs are a coarse proxy for image content, and
smaller COCO samples make FID especially noisy. This code has no claimed
experimental results until model runs complete.

## Current verification status

The unit tests and tiny-model CPU smoke pass. Tiny-model attack, erasure,
checkpoint, and generation paths have also run. The SD 1.4 baseline download
was stopped before completion, and the Phan weights and COCO images are not
present. Consequently, SD 1.4 ASR, target accuracy, retention accuracy, FID,
and CLIPScore are all **NOT MEASURED**. See [docs/verification.md](docs/verification.md)
for the executed checks. The tiny-model images and checkpoints under
`results/tiny/` and `checkpoints/tiny/` are integration artifacts, not research
results.

## References

- Braun et al., [Erased but Not Forgotten](https://arxiv.org/abs/2504.21072).
- Gandikota et al., [Erasing Concepts from Diffusion Models](https://arxiv.org/abs/2303.07345).
- Gandikota et al., [Unified Concept Editing](https://arxiv.org/abs/2308.14761).
- Phan, [PyTorch CIFAR Models](https://github.com/huyvnphan/PyTorch_CIFAR10).
