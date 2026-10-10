"""Run ordered research stages on the user's Modal GPU account."""

from pathlib import Path
import json
import modal

ROOT = Path(__file__).resolve().parents[1]
base_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install_from_requirements(ROOT / "requirements-core.txt")
    .env({"PYTHONPATH": "/prototype/src", "HF_HOME": "/cache/hf"})
)
image = base_image.add_local_dir(ROOT / "src", "/prototype/src")
data_image = (
    base_image.apt_install("ffmpeg")
    .pip_install_from_requirements(ROOT / "requirements-data.txt")
    .add_local_dir(ROOT / "src", "/prototype/src")
    .add_local_dir(ROOT / "configs", "/prototype/configs")
)
volume = modal.Volume.from_name("zeos-vla-research", create_if_missing=True)
app = modal.App("zeos-vla-research", image=image)


@app.function(
    gpu="L40S",
    memory=16384,
    timeout=1800,
    volumes={"/cache": volume},
    secrets=[modal.Secret.from_name("huggingface")],
)
def verify_gemma():
    import os
    import torch
    from PIL import Image, ImageDraw
    from zeos_vla.backbone import FrozenGemma

    # Accept conventional secret field names without printing credentials.
    if "HF_TOKEN" not in os.environ and "HUGGINGFACE_TOKEN" in os.environ:
        os.environ["HF_TOKEN"] = os.environ["HUGGINGFACE_TOKEN"]
    img = Image.new("RGB", (320, 240), "white")
    d = ImageDraw.Draw(img)
    d.rectangle((80, 60, 240, 180), fill="red")
    d.text((90, 100), "Maltesers package", fill="white")
    model = FrozenGemma()
    result = model.encode(img, "Manipulate the package of mini Maltesers slowly.")
    from zeos_vla.semantic import SemanticCommand

    SemanticCommand.parse(result.command_text)
    assert result.features.shape[-1] == model.hidden_dim
    assert torch.isfinite(result.features).all() and not result.features.requires_grad
    assert all(not p.requires_grad for p in model.model.parameters())
    report = {
        "stage": "gemma_multimodal_verification",
        "command": result.command_text,
        **result.metrics,
        "model_revision": __import__(
            "zeos_vla.backbone", fromlist=["MODEL_REVISION"]
        ).MODEL_REVISION,
    }
    Path("/cache/gemma-verification.json").write_text(json.dumps(report, indent=2))
    volume.commit()
    return report


@app.function(
    image=data_image,
    timeout=900,
    volumes={"/cache": volume},
    secrets=[modal.Secret.from_name("huggingface")],
)
def verify_dataset():
    from zeos_vla.data import LeRobotAdapter

    cfg = json.loads(Path("/prototype/configs/package.json").read_text())
    ds = LeRobotAdapter(
        cfg["dataset_id"],
        cfg["dataset_revision"],
        cfg["train_episodes"] + cfg["validation_episodes"],
        instruction=cfg["instruction"],
    ).select_frames(cfg["frames_per_episode"])
    first = ds[0]
    last = ds[len(ds) - 1]
    first["image"].save("/cache/sample.png")
    report = {
        "samples": len(ds),
        "dataset_id": cfg["dataset_id"],
        "revision": cfg["dataset_revision"],
        "first_state": first["proprio"].tolist(),
        "first_actions_shape": list(first["actions"].shape),
        "last_action_mask": last["action_mask"].tolist(),
        "image_size": list(first["image"].size),
        "episodes": ds.episode_ids.unique().tolist(),
        "instruction": first["instruction"],
    }
    Path("/cache/dataset-verification.json").write_text(json.dumps(report, indent=2))
    volume.commit()
    return report


@app.function(
    image=data_image,
    gpu="L40S",
    memory=16384,
    timeout=1800,
    volumes={"/cache": volume},
    secrets=[modal.Secret.from_name("huggingface")],
)
def cache_features():
    import time
    import torch
    from zeos_vla.backbone import FrozenGemma
    from zeos_vla.data import LeRobotAdapter
    from zeos_vla.cache import FeatureCache, provenance, cache_key

    cfg = json.loads(Path("/prototype/configs/package.json").read_text())
    ds = LeRobotAdapter(
        cfg["dataset_id"],
        cfg["dataset_revision"],
        cfg["train_episodes"] + cfg["validation_episodes"],
        instruction=cfg["instruction"],
    ).select_frames(cfg["frames_per_episode"])
    model = FrozenGemma()
    cache = FeatureCache("/cache/features")
    splits = {"train": [], "validation": []}
    elapsed = []
    verified = False
    torch.cuda.reset_peak_memory_stats()
    for i in range(len(ds)):
        sample = ds[i]
        meta = provenance(sample, cfg["dataset_id"], cfg["dataset_revision"])
        if not cache.contains(meta):
            torch.cuda.synchronize()
            start = time.perf_counter()
            f, m = model.extract(sample["image"], sample["instruction"])
            torch.cuda.synchronize()
            elapsed.append(time.perf_counter() - start)
            key = cache.write(meta, f[0], m[0], sample)
        else:
            key = cache_key(meta)
        if not verified:
            live, mask = model.extract(sample["image"], sample["instruction"])
            stored = cache.read(key, meta)
            torch.testing.assert_close(
                stored["features"].float(), live[0].cpu().float(), rtol=0, atol=0
            )
            assert torch.equal(stored["feature_mask"], mask[0].cpu())
            verified = True
        split = (
            "train"
            if sample["episode_index"] in cfg["train_episodes"]
            else "validation"
        )
        splits[split].append(key)
        if i % 20 == 0:
            print(f"Cached {i + 1}/{len(ds)}", flush=True)
    for split, keys in splits.items():
        manifest = {
            "keys": keys,
            "episodes": cfg[
                "train_episodes" if split == "train" else "validation_episodes"
            ],
            "dataset_id": cfg["dataset_id"],
            "dataset_revision": cfg["dataset_revision"],
        }
        Path(f"/cache/features/{split}.json").write_text(json.dumps(manifest, indent=2))
    report = {
        "samples": len(ds),
        "live_cache_exact_consistency": verified,
        "new_prefill_count": len(elapsed),
        "mean_prefill_seconds": sum(elapsed) / len(elapsed) if elapsed else None,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    Path("/cache/cache-report.json").write_text(json.dumps(report, indent=2))
    volume.commit()
    return report


@app.function(gpu="L40S", memory=16384, timeout=1800, volumes={"/cache": volume})
def train_head(steps: int = 1000):
    from zeos_vla.training import train

    report = train(
        "/cache/features",
        "/cache/features/train.json",
        "/cache/features/validation.json",
        "/cache/checkpoints/package.pt",
        steps=steps,
    )
    volume.commit()
    return report


@app.function(gpu="L40S", memory=4096, timeout=300, volumes={"/cache": volume})
def benchmark_head():
    import time
    import torch
    from zeos_vla.head import FlowHead, HeadConfig
    from zeos_vla.gate import ZeosActionGate
    from zeos_vla.semantic import SemanticCommand

    torch.manual_seed(42)
    head = FlowHead(HeadConfig()).cuda()
    f = torch.randn(4, 1024, 2560, device="cuda")
    p = torch.randn(4, 7, device="cuda")
    a = torch.randn(4, 16, 7, device="cuda")
    mask = torch.ones(4, 1024, device="cuda", dtype=torch.bool)
    opt = torch.optim.AdamW(head.parameters(), lr=3e-4)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    losses = []
    for _ in range(30):
        opt.zero_grad()
        loss = head.loss(a, f, p, mask)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    torch.cuda.synchronize()
    training = time.perf_counter() - start
    assert all(torch.isfinite(v.grad).all() for v in head.parameters())
    head.eval()
    head.sample(f[:1], p[:1], mask[:1], steps=32)  # warmup
    latencies = []
    for _ in range(20):
        torch.cuda.synchronize()
        start = time.perf_counter()
        chunk = head.sample(f[:1], p[:1], mask[:1], steps=32)
        torch.cuda.synchronize()
        latencies.append(time.perf_counter() - start)
    import statistics

    inference = statistics.median(latencies)
    command = SemanticCommand("manipulate", "package", "slow")
    report = {
        "scope": "Independent action head on synthetic conditioning/actions; not pretrained Gemma or recorded-action training",
        "gpu": torch.cuda.get_device_name(),
        "feature_shape": list(f.shape),
        "chunk_shape": list(chunk.shape),
        "training_iterations": 30,
        "training_seconds": training,
        "head_inference_seconds": inference,
        "head_inference_p95_seconds": sorted(latencies)[18],
        "head_inference_trials": len(latencies),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "all_gradients_finite": True,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "real_zeos_allow": ZeosActionGate(True).authorize(command).allowed,
        "real_zeos_veto": not ZeosActionGate(False).authorize(command).allowed,
    }
    Path("/cache/head-benchmark.json").write_text(json.dumps(report, indent=2))
    volume.commit()
    return report


@app.function(
    gpu="L40S",
    memory=16384,
    timeout=600,
    volumes={"/cache": volume},
    secrets=[modal.Secret.from_name("huggingface")],
)
def complete_demo():
    import time
    import torch
    from PIL import Image
    from zeos_vla.backbone import FrozenGemma
    from zeos_vla.cache import FeatureCache
    from zeos_vla.training import Policy
    from zeos_vla.gate import ZeosActionGate
    from zeos_vla.runtime import JointStateSimulator, run_encoded

    manifest = json.loads(Path("/cache/features/train.json").read_text())
    record = FeatureCache("/cache/features").read(manifest["keys"][0])
    image = Image.open("/cache/sample.png").convert("RGB")
    instruction = record["meta"]["instruction"]
    state = record["proprio"]
    backbone = FrozenGemma()
    policy = Policy("/cache/checkpoints/package.pt", "cuda")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    encode_start = time.perf_counter()
    encoding = backbone.encode(image, instruction)
    encoding_seconds = time.perf_counter() - encode_start
    torch.testing.assert_close(
        encoding.features[0].cpu().float(), record["features"].float(), rtol=0, atol=0
    )
    cases = {}
    for name, allow in [("veto", False), ("allow", True)]:
        robot = JointStateSimulator(state)
        start = time.perf_counter()
        cases[name] = run_encoded(
            backbone,
            policy,
            ZeosActionGate(allow),
            robot,
            encoding,
            image,
            instruction,
            state,
        )
        torch.cuda.synchronize()
        cases[name]["post_encoding_seconds"] = time.perf_counter() - start
        cases[name]["total_seconds_including_encoding"] = (
            encoding_seconds + cases[name]["post_encoding_seconds"]
        )
        cases[name]["simulator_chunks"] = len(robot.chunks)
    assert cases["veto"]["dispatched_steps"] == 0 and not cases["veto"]["head_called"]
    assert cases["veto"]["feedback"] is not None
    assert cases["allow"]["allowed"] and cases["allow"]["dispatched_steps"] == 16
    assert cases["allow"]["joint_delta_norm"] > 0
    report = {
        "scope": "Frozen pretrained Gemma + recorded-action-trained head + real ZEOS + joint-state simulator",
        "isaac_sim_tested": False,
        "hardware_tested": False,
        "gpu": torch.cuda.get_device_name(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "initial_proprio": state.tolist(),
        "cases": cases,
    }
    Path("/cache/complete-demo.json").write_text(json.dumps(report, indent=2))
    volume.commit()
    return report


@app.local_entrypoint()
def main(stage: str = "gemma", steps: int = 1000):
    calls = {
        "gemma": verify_gemma,
        "dataset": verify_dataset,
        "cache": cache_features,
        "train": train_head,
        "head": benchmark_head,
        "demo": complete_demo,
    }
    if stage not in calls:
        raise ValueError(f"Unknown stage: {stage}")
    result = calls[stage].remote(steps) if stage == "train" else calls[stage].remote()
    (ROOT / "reports" / f"{stage}-verification.json").write_text(
        json.dumps(result, indent=2)
    )
    if stage == "demo":
        summary = {k: v for k, v in result.items() if k != "cases"}
        summary["cases"] = {
            name: {
                k: (
                    {gk: gv for gk, gv in v.items() if gk != "journal"}
                    if k == "gate"
                    else v
                )
                for k, v in case.items()
                if k != "actions"
            }
            for name, case in result["cases"].items()
        }
        print(json.dumps(summary, indent=2))
    else:
        print(json.dumps(result, indent=2))
