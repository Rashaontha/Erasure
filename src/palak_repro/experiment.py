"""Checkpointed stages for the six object-erasure conditions per target."""
import csv
import gc
from pathlib import Path

import torch

from .closed_form import erase_uce, inject_shallow
from .common import CLASSES, seed_all, write_json
from .datasets import coco_samples
from .distill import train_deep, train_esd
from .metrics import CifarClassifier, classification_metrics, clip_score, fid_score
from .model import generator, load_pipeline, save_unet


def checkpoint_path(cfg, target, attack, erasure=None):
    name = f"{target}__{attack}" + (f"__{erasure}" if erasure else "")
    return Path(cfg["checkpoint_dir"]) / name


def has_checkpoint(path):
    return (Path(path) / "config.json").exists() and (Path(path) / "diffusion_pytorch_model.safetensors").exists()


def injection(cfg, target, attack):
    if attack == "none":
        return None
    output = checkpoint_path(cfg, target, attack)
    if has_checkpoint(output):
        return output
    seed_all(cfg["seed"])
    pipe = load_pipeline(cfg)
    if attack == "shallow":
        count = inject_shallow(pipe, target, cfg["trigger"], cfg["shallow"])
        metadata = {"projections_edited": count}
    elif attack == "deep":
        losses = train_deep(pipe, cfg, target, cfg["trigger"])
        metadata = {"training_steps": len(losses), "final_training_loss": losses[-1]}
    else:
        raise ValueError(attack)
    save_unet(pipe, output)
    write_json(output / "training.json", {"target": target, "attack": attack, **metadata})
    del pipe
    gc.collect()
    return output


def erasure(cfg, target, attack, method):
    output = checkpoint_path(cfg, target, attack, method)
    if has_checkpoint(output):
        return output
    source = checkpoint_path(cfg, target, attack) if attack != "none" else None
    if source and not has_checkpoint(source):
        raise FileNotFoundError(f"Run injection first: {source}")
    seed_all(cfg["seed"])
    pipe = load_pipeline(cfg, source)
    if method == "uce":
        count = erase_uce(pipe, target, cfg["uce"], preserve=cfg["uce"].get("preserve_concepts", []))
        metadata = {"projections_edited": count}
    elif method == "esd":
        losses = train_esd(pipe, cfg, target, source)
        metadata = {"training_steps": len(losses), "final_training_loss": losses[-1]}
    else:
        raise ValueError(method)
    save_unet(pipe, output)
    write_json(output / "training.json", {"target": target, "attack": attack, "erasure": method, **metadata})
    del pipe
    gc.collect()
    return output


def image_dir(cfg, target, attack, method):
    return Path(cfg["output_dir"]) / "images" / target / attack / method


def generate(cfg, target, attack, method, samples):
    source = None if method == "clean" else checkpoint_path(cfg, target, attack, method)
    if source and not has_checkpoint(source):
        raise FileNotFoundError(f"Run erasure first: {source}")
    base = image_dir(cfg, target, attack, method)
    settings = cfg["generation"]
    prompts = []
    for kind, label, count in (("target", target, settings["target_count"]),
                               ("trigger", cfg["trigger"], settings["trigger_count"])):
        prompts.extend({"kind": kind, "class": target, "prompt": f"a photo of {label}", "index": i,
                        "seed": cfg["seed"] + i} for i in range(count))
    for label in CLASSES:
        if label != target:
            prompts.extend({"kind": "retention", "class": label, "prompt": f"a photo of {label}",
                            "index": i, "seed": cfg["seed"] + i} for i in range(settings["retention_count"]))
    prompts.extend({"kind": "coco", "class": None, "prompt": s["caption"], "index": i,
                    "seed": cfg["seed"] + i, "reference": s["reference"], "image_id": s["image_id"]}
                   for i, s in enumerate(samples))
    for item in prompts:
        folder = item["class"] if item["kind"] == "retention" else item["kind"]
        item["path"] = str(base / folder / f"{item['index']:04d}.png")
    if any(not Path(item["path"]).exists() for item in prompts):
        pipe = load_pipeline(cfg, source)
        for item in prompts:
            path = Path(item["path"])
            if path.exists():
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            result = pipe(prompt=item["prompt"], width=settings["width"], height=settings["height"],
                          num_inference_steps=settings["steps"], guidance_scale=settings["guidance_scale"],
                          generator=generator(pipe.device, item["seed"]))
            result.images[0].save(path)
        del pipe
        gc.collect()
    write_json(base / "manifest.json", prompts)
    return prompts


def evaluate(cfg, target, attack, method, records):
    output = Path(cfg["output_dir"]) / "tables" / f"{target}__{attack}__{method}.json"
    if output.exists():
        import json
        return json.loads(output.read_text(encoding="utf-8"))
    classifier = CifarClassifier(cfg["classifier"], torch.device(cfg["device"] if cfg["device"] != "auto" else "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"))
    classified = [r for r in records if r["kind"] != "coco"]
    predictions = classifier.predict([r["path"] for r in classified])
    for record, prediction in zip(classified, predictions):
        record["prediction_index"] = prediction
        record["prediction"] = CLASSES[prediction]
    metrics = classification_metrics(classified, predictions, target)
    coco = [r for r in records if r["kind"] == "coco"]
    if coco and cfg["metrics"]["fid"]:
        metrics["fid"] = fid_score([r["path"] for r in coco], [r["reference"] for r in coco], classifier.device)
    if coco and cfg["metrics"]["clipscore"]:
        metrics["clipscore"] = clip_score([r["path"] for r in coco], [r["prompt"] for r in coco], classifier.device,
                                           cfg["metrics"]["clip_model"], cfg["metrics"]["clip_pretrained"])
    row = {"target": target, "attack": attack, "erasure": method, **metrics,
           "n_coco": len(coco), "seed": cfg["seed"]}
    write_json(output, row)
    write_json(output.with_suffix(".predictions.json"), classified)
    return row


def run(cfg, stage="all", target_filter=None, attack_filter=None, erasure_filter=None):
    write_json(Path(cfg["output_dir"]) / "run_config.json", cfg)
    samples = coco_samples(cfg["coco"]["captions_json"], cfg["coco"]["images_dir"],
                           cfg["coco"]["count"], cfg["coco"]["seed"]) if stage in ("all", "generate") else []
    rows = []
    for target in cfg["targets"]:
        if target_filter and target != target_filter:
            continue
        for attack in ("none", "shallow", "deep"):
            if attack_filter and attack != attack_filter:
                continue
            if stage in ("all", "inject") and attack != "none":
                injection(cfg, target, attack)
            methods = ("clean", "esd", "uce") if attack == "none" else ("esd", "uce")
            for method in methods:
                if erasure_filter and method != erasure_filter:
                    continue
                if method != "clean" and stage in ("all", "erase"):
                    erasure(cfg, target, attack, method)
                if stage in ("all", "generate"):
                    records = generate(cfg, target, attack, method, samples)
                elif stage == "evaluate":
                    import json
                    path = image_dir(cfg, target, attack, method) / "manifest.json"
                    records = json.loads(path.read_text(encoding="utf-8"))
                else:
                    continue
                if stage in ("all", "evaluate"):
                    rows.append(evaluate(cfg, target, attack, method, records))
    return rows
