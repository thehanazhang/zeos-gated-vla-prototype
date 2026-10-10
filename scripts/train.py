import argparse
import json
from zeos_vla.training import train

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="cache/features")
    p.add_argument("--checkpoint", default="checkpoints/package.pt")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    print(
        json.dumps(
            train(
                a.cache,
                f"{a.cache}/train.json",
                f"{a.cache}/validation.json",
                a.checkpoint,
                a.steps,
                a.batch_size,
                a.device,
            ),
            indent=2,
        )
    )
