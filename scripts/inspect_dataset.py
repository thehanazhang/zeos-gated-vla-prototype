"""Authenticated dataset metadata discovery on Modal (no credentials returned)."""

import json
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("zeos-vla-dataset-inspection")
image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "huggingface-hub==1.31.0"
)


@app.function(image=image, secrets=[modal.Secret.from_name("huggingface")], timeout=120)
def inspect():
    import os
    from huggingface_hub import HfApi, hf_hub_download

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    repo = "thehanazhang/b601_rs_manipulate_package_v3_150"
    api = HfApi(token=token)
    info = api.dataset_info(repo)
    p = hf_hub_download(
        repo, "meta/info.json", repo_type="dataset", revision=info.sha, token=token
    )
    metadata = json.loads(Path(p).read_text())
    return {
        "repo_id": repo,
        "revision": info.sha,
        "metadata": metadata,
        "files": [s.rfilename for s in info.siblings],
    }


@app.local_entrypoint()
def main():
    result = inspect.remote()
    (ROOT / "reports" / "dataset-inspection.json").write_text(
        json.dumps(result, indent=2)
    )
    print(json.dumps({k: v for k, v in result.items() if k != "files"}, indent=2))
