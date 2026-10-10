"""Train only the head from verified offline features and recorded action chunks."""

from dataclasses import asdict
import json
from pathlib import Path
import time
import torch
from torch.utils.data import DataLoader
from zeos_vla.cache import CachedDataset, collate
from zeos_vla.head import FlowHead, HeadConfig
from zeos_vla.backbone import MODEL_ID, MODEL_REVISION, PROMPT_VERSION


def fit_stats(records):
    actions = torch.cat([r["actions"][r["action_mask"]] for r in records])
    states = torch.stack([r["proprio"] for r in records])
    return {
        "action_mean": actions.mean(0),
        "action_std": actions.std(0, unbiased=False).clamp_min(1e-3),
        "proprio_mean": states.mean(0),
        "proprio_std": states.std(0, unbiased=False).clamp_min(1e-3),
    }


def normalize(batch, stats):
    return {
        **batch,
        "actions": (batch["actions"] - stats["action_mean"]) / stats["action_std"],
        "proprio": (batch["proprio"] - stats["proprio_mean"]) / stats["proprio_std"],
    }


def save_checkpoint(path, head, stats, provenance):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "config": asdict(head.config),
            "state_dict": {k: v.detach().cpu() for k, v in head.state_dict().items()},
            "normalization": {k: v.cpu() for k, v in stats.items()},
            "provenance": provenance,
        },
        path,
    )


class Policy:
    def __init__(self, checkpoint, device="cpu"):
        self.device = torch.device(device)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if payload["format_version"] != 1:
            raise ValueError("Unsupported checkpoint")
        self.provenance = payload["provenance"]
        if (
            self.provenance.get("model_revision") != MODEL_REVISION
            or self.provenance.get("prompt_version") != PROMPT_VERSION
        ):
            raise ValueError("Checkpoint backbone/prompt mismatch")
        self.head = FlowHead(HeadConfig(**payload["config"])).to(self.device).eval()
        self.head.load_state_dict(payload["state_dict"], strict=True)
        self.stats = {k: v.to(self.device) for k, v in payload["normalization"].items()}

    @torch.no_grad()
    def predict(self, features, mask, proprio, seed=0, steps=32):
        features = features.to(self.device, dtype=torch.float32).clone()
        mask = mask.to(self.device).clone()
        p = proprio.to(self.device, dtype=torch.float32).clone()
        normalized = (p - self.stats["proprio_mean"]) / self.stats["proprio_std"]
        actions = self.head.sample(
            features,
            normalized,
            mask,
            steps,
            generator=torch.Generator(device=self.device).manual_seed(seed),
        )
        result = actions * self.stats["action_std"] + self.stats["action_mean"]
        if not torch.isfinite(result).all():
            raise RuntimeError("Invalid policy chunk")
        return result


def train(
    cache_root,
    train_manifest,
    val_manifest,
    checkpoint,
    steps=1000,
    batch_size=8,
    device="cuda",
):
    if steps < 1 or batch_size < 1:
        raise ValueError("Positive training steps and batch size required")
    torch.manual_seed(42)
    train_ds = CachedDataset(cache_root, train_manifest)
    val_ds = CachedDataset(cache_root, val_manifest)
    train_eps = set(train_ds.manifest["episodes"])
    val_eps = set(val_ds.manifest["episodes"])
    if train_eps & val_eps:
        raise ValueError("Episode leakage between train and validation")
    for ds in (train_ds, val_ds):
        for i in range(len(ds)):
            meta = ds[i]["meta"]
            if (
                meta["model_revision"] != MODEL_REVISION
                or meta["prompt_version"] != PROMPT_VERSION
            ):
                raise ValueError("Training cache backbone/prompt mismatch")
            if meta["episode_index"] not in ds.manifest["episodes"]:
                raise ValueError("Cache record outside selected episodes")
            if (
                meta["dataset_id"] != ds.manifest["dataset_id"]
                or meta["dataset_revision"] != ds.manifest["dataset_revision"]
            ):
                raise ValueError("Cache dataset provenance mismatch")
    if (
        train_ds.manifest["dataset_id"] != val_ds.manifest["dataset_id"]
        or train_ds.manifest["dataset_revision"] != val_ds.manifest["dataset_revision"]
    ):
        raise ValueError("Different train/validation datasets")
    records = [train_ds[i] for i in range(len(train_ds))]
    stats = {k: v.to(device) for k, v in fit_stats(records).items()}
    first = records[0]
    cfg = HeadConfig(
        feature_dim=first["features"].shape[-1],
        proprio_dim=first["proprio"].numel(),
        action_dim=first["actions"].shape[-1],
    )
    head = FlowHead(cfg).to(device)
    loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=0.01)
    iterator = iter(loader)
    history = []
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for step in range(steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        b = normalize({k: v.to(device) for k, v in batch.items()}, stats)
        optimizer.zero_grad(set_to_none=True)
        loss = head.loss(
            b["actions"],
            b["features"],
            b["proprio"],
            b["feature_mask"],
            b["action_mask"],
        )
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
        if step % 100 == 0 or step == steps - 1:
            history.append({"step": step, "loss": loss.item()})
            print(history[-1], flush=True)
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    head.eval()
    val_losses = []
    mse = []
    with torch.no_grad():
        for batch in DataLoader(val_ds, batch_size=batch_size, collate_fn=collate):
            b = normalize({k: v.to(device) for k, v in batch.items()}, stats)
            val_losses.append(
                head.loss(
                    b["actions"],
                    b["features"],
                    b["proprio"],
                    b["feature_mask"],
                    b["action_mask"],
                ).item()
            )
            pred = head.sample(b["features"], b["proprio"], b["feature_mask"], steps=16)
            error = (pred - b["actions"]).square() * b["action_mask"][..., None]
            mse.append((error.sum() / (b["action_mask"].sum() * cfg.action_dim)).item())
    provenance = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "prompt_version": PROMPT_VERSION,
        "train_manifest": train_ds.manifest,
        "validation_manifest": val_ds.manifest,
        "seed": 42,
    }
    save_checkpoint(checkpoint, head, stats, provenance)
    report = {
        "steps": steps,
        "train_samples": len(train_ds),
        "validation_samples": len(val_ds),
        "history": history,
        "training_seconds": elapsed,
        "validation_flow_loss": sum(val_losses) / len(val_losses),
        "validation_normalized_action_mse": sum(mse) / len(mse),
        "head_parameters": sum(p.numel() for p in head.parameters()),
        "train_episodes": sorted(train_eps),
        "validation_episodes": sorted(val_eps),
    }
    if str(device).startswith("cuda"):
        report.update(
            gpu=torch.cuda.get_device_name(),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        )
    Path(str(checkpoint) + ".json").write_text(json.dumps(report, indent=2))
    return report
