"""Dedicated Isaac Sim environment; preserve the verified Gemma dependency lock."""

from pathlib import Path
import json
import modal

ROOT = Path(__file__).resolve().parents[1]
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libvulkan1", "libgl1", "libegl1", "libglib2.0-0", "libx11-6",
                 "libxext6", "libxrender1", "libxi6", "libxrandr2", "libsm6", "libgomp1")
    .pip_install("isaacsim[all,extscache]==6.1.0.0", extra_index_url="https://pypi.nvidia.com")
    .env({"OMNI_KIT_ACCEPT_EULA": "YES", "NVIDIA_DRIVER_CAPABILITIES": "all"})
    .apt_install("libxt6", "git")
    .pip_install("zeos @ git+https://github.com/metacognitionai/zeos.git@36aaeafbe185530f0c33a9e049745bbb1b3ba3bd")
    .env({"PYTHONPATH": "/src"})
    .add_local_dir(ROOT / "src", "/src")
    .add_local_dir(ROOT / "scripts", "/scripts")
)
if (ROOT / "assets/b601-RS").exists():
    image = image.add_local_dir(ROOT / "assets/b601-RS", "/assets/b601-RS")
app = modal.App("zeos-vla-isaac", image=image)
volume = modal.Volume.from_name("zeos-vla-research")


@app.function(gpu="L40S", cpu=4, memory=32768, timeout=600, scaledown_window=2,
              volumes={"/cache": volume})
def smoke(mode="smoke"):
    import time
    import subprocess
    import sys

    start = time.perf_counter()
    output = Path(f"/cache/isaac-{mode}")
    output.mkdir(exist_ok=True)
    report_path = output / "report.json"
    report_path.unlink(missing_ok=True)
    memory_samples = []
    with (output / "worker.log").open("w") as log:
        process = subprocess.Popen([
                sys.executable, "/scripts/isaac_worker.py", "--mode", mode,
                "--output", str(output),
            ], stdout=log, stderr=subprocess.STDOUT)
        deadline = time.perf_counter() + 480
        while process.poll() is None:
            if time.perf_counter() > deadline:
                process.kill()
                process.wait()
                returncode = "timeout"
                break
            gpu_memory = subprocess.run([
                "nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
            ], capture_output=True, text=True, timeout=5)
            if gpu_memory.returncode == 0:
                memory_samples.append(int(gpu_memory.stdout.strip().splitlines()[0]))
            time.sleep(0.5)
        else:
            returncode = process.returncode
    report = json.loads(report_path.read_text()) if report_path.exists() else {"completed": False}
    report.update(process_returncode=returncode, function_elapsed_seconds=time.perf_counter()-start)
    report.update(sampled_peak_gpu_memory_mib=max(memory_samples, default=None),
                  gpu_memory_sample_count=len(memory_samples), gpu_memory_sampling_interval_seconds=0.5,
                  memory_scope="nvidia-smi total device used, sampled while isolated worker runs")
    report_path.write_text(json.dumps(report, indent=2))
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True).stdout
    (output / "requirements.txt").write_text(freeze)
    volume.commit()
    return {"report": report, "files": {p.name: p.read_bytes() for p in output.iterdir() if p.is_file()}}


@app.local_entrypoint()
def main(mode: str = "smoke"):
    if mode not in {"smoke", "replay"}:
        raise ValueError("Mode must be smoke or replay")
    result = smoke.remote(mode)
    output = ROOT / f"reports/isaac-{mode}"
    output.mkdir(exist_ok=True)
    for name, data in result["files"].items():
        (output / name).write_bytes(data)
    print(json.dumps({"process_returncode": result["report"]["process_returncode"],
                      "saved_to": str(output)}, indent=2))
    if result["report"]["process_returncode"] != 0:
        raise RuntimeError(f"Isaac failed; inspect {output / 'worker.log'}")
