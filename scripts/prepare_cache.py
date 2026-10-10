"""Cache original Gemma final-layer features from LeRobot episodes."""

import argparse
import json
from pathlib import Path
from zeos_vla.backbone import FrozenGemma
from zeos_vla.cache import FeatureCache, provenance, cache_key
from zeos_vla.data import LeRobotAdapter

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/package.json")
    p.add_argument("--cache", default="cache/features")
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    train = set(cfg["train_episodes"])
    val = set(cfg["validation_episodes"])
    if train & val:
        raise ValueError("Episode splits overlap")
    ds = LeRobotAdapter(
        cfg["dataset_id"],
        cfg["dataset_revision"],
        sorted(train | val),
        camera_key=cfg["camera_key"],
        instruction=cfg["instruction"],
    ).select_frames(cfg["frames_per_episode"])
    model = FrozenGemma(a.device)
    cache = FeatureCache(a.cache)
    splits = {"train": [], "validation": []}
    for sample in ds:
        meta = provenance(sample, cfg["dataset_id"], cfg["dataset_revision"])
        if cache.contains(meta):
            cache.read(cache_key(meta), meta)
            key = cache_key(meta)
        else:
            f, m = model.extract(sample["image"], sample["instruction"])
            key = cache.write(meta, f[0], m[0], sample)
        split = "train" if sample["episode_index"] in train else "validation"
        splits[split].append(key)
    for split, keys in splits.items():
        Path(a.cache, f"{split}.json").write_text(
            json.dumps(
                {
                    "keys": keys,
                    "episodes": sorted(train if split == "train" else val),
                    "dataset_id": cfg["dataset_id"],
                    "dataset_revision": cfg["dataset_revision"],
                },
                indent=2,
            )
        )
