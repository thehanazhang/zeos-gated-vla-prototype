"""Bounded graphics/driver probe before installing Isaac Sim on Modal."""

from pathlib import Path
import json
import modal

ROOT = Path(__file__).resolve().parents[1]
image = modal.Image.debian_slim(python_version="3.11").apt_install(
    "vulkan-tools", "libvulkan1", "libgl1", "libegl1"
)
app = modal.App("zeos-vla-isaac-probe", image=image)


@app.function(gpu="L40S", memory=2048, cpu=1, timeout=120, scaledown_window=2)
def probe():
    import os
    import glob
    import subprocess

    def command(args):
        try:
            p = subprocess.run(args, capture_output=True, text=True, timeout=30)
            return {
                "exit_code": p.returncode,
                "stdout": p.stdout[-16000:],
                "stderr": p.stderr[-16000:],
            }
        except Exception as exc:
            return {"error": str(exc)}

    return {
        "gpu": command(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        ),
        "vulkan": command(["vulkaninfo", "--summary"]),
        "icd_files": glob.glob("/usr/share/vulkan/icd.d/*")
        + glob.glob("/etc/vulkan/icd.d/*"),
        "driver_capabilities": os.environ.get("NVIDIA_DRIVER_CAPABILITIES"),
        "glibc": command(["ldd", "--version"]),
        "isaac_installed": False,
    }


@app.local_entrypoint()
def main():
    result = probe.remote()
    (ROOT / "reports/isaac-platform-probe.json").write_text(
        json.dumps(result, indent=2)
    )
    print(json.dumps(result, indent=2))
