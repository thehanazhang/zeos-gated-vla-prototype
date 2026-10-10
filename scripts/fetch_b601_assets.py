"""Fetch pinned manufacturer assets without committing meshes into this repo."""

from pathlib import Path
import argparse
import json
import shutil
import subprocess
import tempfile

REVISION = "79f01c7056d5f667beba7e838593ea6c299388c5"
URL = "https://github.com/Seeed-Projects/reBot-DevArm.git"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, default=Path("assets/b601-RS"))
    args = parser.parse_args()
    if args.destination.exists():
        raise SystemExit("Destination exists; inspect it before replacing assets")
    with tempfile.TemporaryDirectory() as temp:
        clone = Path(temp) / "source"
        subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout", URL, str(clone)], check=True)
        subprocess.run(["git", "-C", str(clone), "checkout", REVISION, "--", "Rebot_Arm_description/RS", "LICENSE"], check=True)
        shutil.copytree(clone / "Rebot_Arm_description/RS", args.destination)
        shutil.copy2(clone / "LICENSE", args.destination / "LICENSE")
    (args.destination / "SOURCE.json").write_text(json.dumps({
        "repository": URL, "revision": REVISION, "license": "CERN-OHL-W-2.0",
        "source_path": "Rebot_Arm_description/RS", "modifications": "none",
    }, indent=2))
    print(args.destination.resolve())


if __name__ == "__main__":
    main()
