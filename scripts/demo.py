"""Real Gemma + checkpoint + real ZEOS + joint-state simulator, no robot hardware."""

import argparse
import json
from pathlib import Path
from PIL import Image
import torch
from zeos_vla.backbone import FrozenGemma
from zeos_vla.training import Policy
from zeos_vla.gate import ZeosActionGate
from zeos_vla.runtime import JointStateSimulator, observe_and_act

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--image", required=True)
    p.add_argument(
        "--proprio", required=True, help="JSON list of recorded-unit joint values"
    )
    p.add_argument(
        "--instruction", default="Manipulate the package of mini Maltesers slowly."
    )
    p.add_argument("--gate-config", default="configs/gate.json")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", default="reports/demo.json")
    a = p.parse_args()
    cfg = json.loads(Path(a.gate_config).read_text())
    proprio = torch.tensor(json.loads(a.proprio), dtype=torch.float32)
    result = observe_and_act(
        FrozenGemma(a.device),
        Policy(a.checkpoint, a.device),
        ZeosActionGate(cfg["allow_package_manipulation"]),
        JointStateSimulator(proprio),
        Image.open(a.image),
        a.instruction,
        proprio,
    )
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("gate", "actions")}, indent=2
        )
    )
