"""Disk cache of detached original Gemma representations; strict provenance."""

import hashlib
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset
from zeos_vla.backbone import MODEL_ID, MODEL_REVISION, PROMPT_VERSION, SYSTEM


def tensor_digest(tensor):
    t = tensor.detach().cpu().contiguous()
    return hashlib.sha256(
        str(t.dtype).encode()
        + str(tuple(t.shape)).encode()
        + t.view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def provenance(sample, dataset_id, dataset_revision):
    return {
        "format_version": 1,
        "dataset_id": dataset_id,
        "dataset_revision": dataset_revision,
        "episode_index": sample["episode_index"],
        "frame_index": sample["frame_index"],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "prompt_version": PROMPT_VERSION,
        "system_sha256": hashlib.sha256(SYSTEM.encode()).hexdigest(),
        "instruction": sample["instruction"],
        "image_sha256": hashlib.sha256(sample["image"].tobytes()).hexdigest(),
        "image_size": list(sample["image"].size),
        "feature_dtype": "torch.bfloat16",
        "transformers": "4.53.3",
        "processor_policy": "Gemma3 default apply_chat_template RGB",
    }


def cache_key(meta):
    return hashlib.sha256(
        json.dumps(meta, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class FeatureCache:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def write(self, meta, features, mask, sample):
        key = cache_key(meta)
        f = features.detach().cpu().to(torch.bfloat16).clone()
        m = mask.detach().cpu().bool().clone()
        if (
            f.ndim != 2
            or m.shape != (len(f),)
            or not m.any()
            or not torch.isfinite(f).all()
        ):
            raise ValueError("Invalid features")
        record = {
            "meta": meta,
            "features": f,
            "feature_mask": m,
            "actions": sample["actions"],
            "proprio": sample["proprio"],
            "action_mask": sample["action_mask"],
        }
        record["digests"] = {
            k: tensor_digest(v)
            for k, v in record.items()
            if isinstance(v, torch.Tensor)
        }
        tmp = self.root / (key + ".tmp")
        torch.save(record, tmp)
        tmp.replace(self.root / (key + ".pt"))
        return key

    def read(self, key, expected_meta=None):
        record = torch.load(
            self.root / (key + ".pt"), map_location="cpu", weights_only=True
        )
        if cache_key(record["meta"]) != key or (
            expected_meta is not None and record["meta"] != expected_meta
        ):
            raise ValueError("Cache provenance mismatch")
        for k, digest in record["digests"].items():
            if tensor_digest(record[k]) != digest:
                raise ValueError("Cache checksum mismatch")
        return record

    def contains(self, meta):
        return (self.root / (cache_key(meta) + ".pt")).is_file()


class CachedDataset(Dataset):
    def __init__(self, root, manifest):
        self.cache = FeatureCache(root)
        self.manifest = json.loads(Path(manifest).read_text())
        self.keys = self.manifest["keys"]
        if not self.keys:
            raise ValueError("Empty feature manifest")

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        return self.cache.read(self.keys[i])


def collate(records):
    longest = max(len(r["features"]) for r in records)
    dim = records[0]["features"].shape[-1]
    features = torch.zeros(len(records), longest, dim, dtype=torch.float32)
    mask = torch.zeros(len(records), longest, dtype=torch.bool)
    for i, r in enumerate(records):
        n = len(r["features"])
        features[i, :n] = r["features"].float()
        mask[i, :n] = r["feature_mask"]
    return {
        "features": features,
        "feature_mask": mask,
        **{
            k: torch.stack([r[k] for r in records])
            for k in ("actions", "proprio", "action_mask")
        },
    }
