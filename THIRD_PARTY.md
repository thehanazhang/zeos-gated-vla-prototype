# Upstream components and assets

- Metacognition ZEOS: AGPL-3.0-only, pinned commit `36aaeafbe185530f0c33a9e049745bbb1b3ba3bd`, [source](https://github.com/metacognitionai/zeos). Installed unmodified. This prototype uses AGPL-3.0-only.
- Google Gemma 4 E4B IT: Apache-2.0, pinned model revision in `configs/package.json`, [model card](https://huggingface.co/google/gemma-4-E4B-it). Weights are downloaded separately and excluded from this repository.
- Seeed Studio reBot B601-RS URDF and mesh assets: CERN-OHL-W-2.0, pinned commit `79f01c7056d5f667beba7e838593ea6c299388c5`, [manufacturer repository](https://github.com/Seeed-Projects/reBot-DevArm). `scripts/fetch_b601_assets.py` retains the upstream license and source record alongside the unmodified assets. Meshes and generated USD are excluded from Git. The simulator's display scaling is application code, not a calibration or modification of the original CAD files.
- NVIDIA Isaac Sim: proprietary licensed runtime, [license documentation](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/common/licenses-isaac-sim.html). Installed separately on the user-authorized Modal compute; no runtime binaries redistributed here.

The private recorded dataset and decoded sample image remain subject to their owner's access controls; no recorded RGB image is committed. Demonstration scene props are generic cubes and do not reproduce Maltesers packaging artwork.
