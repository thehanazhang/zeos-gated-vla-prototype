# ZEOS-gated Gemma action prototype

Frozen `google/gemma-4-E4B-it` produces a structured semantic proposal and the final hidden states of its original image/instruction prompt. The semantic proposal alone enters the real Metacognition ZEOS kernel. A separate deterministic guard job checks the proposal. Only an authorized device-pipe write permits the flow-matching head to sample a 16-step action chunk using the original Gemma representations and robot proprioception. A veto returns feedback to the same Gemma and dispatches zero actions.

No Gemma KV-cache modifications, embedding interpretation by ZEOS, second reasoning VLM, or trained semantic decoder. Native Transformers generation uses its normal cache. Feature extraction uses a separate cache-free forward through Gemma's base model, which exposes `last_hidden_state` directly and avoids retaining every layer.

The E4B model uses 4.5B effective parameters and has approximately 8B total parameters. The `it` checkpoint is instruction-tuned, making it appropriate for structured command generation. We disable optional thinking through the official chat-template argument and keep all model parameters frozen. The model revision is pinned to `ee0ef6023621cff504d758262d4e04895a5af4a2`.

This prototype uses a private Hugging Face dataset that has been filtered for data quality and encoder wobble. Metadata: LeRobot v3.0, 150 episodes, 39,559 frames, 30 Hz, wrist RGB 640×480, seven state/action components (six arm joints plus gripper).

The pilot uses 12 training episodes and 3 disjoint validation episodes, sparsely sampled. It is a recorded-action feasibility experiment, not evidence of successful package manipulation. A 16-frame chunk at 30 Hz spans about 0.53 seconds.

## Gate policy

`configs/gate.json` defaults to rejecting package manipulation. Set `allow_package_manipulation` to `true` to allow the same proposal. The policy is owned by the application guard; enforcement (holding writes, consulting jobs, verdict parsing, failure and timeout handling) belongs to the unmodified ZEOS kernel. This demonstrates permission gating, not collision avoidance or a proof that the sampled trajectory implements the semantic command.

## Upstream API references

- [Transformers 5.19.0 Gemma 4 implementation](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/gemma4/modeling_gemma4.py): `AutoModelForMultimodalLM` loads `Gemma4ForConditionalGeneration`; its `.model` returns the multimodal final hidden states; `AutoProcessor.apply_chat_template` handles image and text.
- [LeRobot 0.6.1 dataset implementation](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/datasets/lerobot_dataset.py): `delta_timestamps`, episode-boundary padding, `video_backend="pyav"`.
- [Pinned ZEOS action-gate mechanism](https://github.com/metacognitionai/zeos/blob/36aaeafbe185530f0c33a9e049745bbb1b3ba3bd/src/zeos/core/gates.py), [kernel](https://github.com/metacognitionai/zeos/blob/36aaeafbe185530f0c33a9e049745bbb1b3ba3bd/src/zeos/core/kernel.py), and [CommandSeat API](https://github.com/metacognitionai/zeos/blob/36aaeafbe185530f0c33a9e049745bbb1b3ba3bd/src/zeos/machine/seat.py).
- [Modal GPUs](https://modal.com/docs/guide/gpu) and [secrets](https://modal.com/docs/guide/secrets).

ZEOS source was inspected before implementing the adapter, including its remote-actuation gate and gate-timeout integration tests. ZEOS uses AGPL-3.0-only; this repository uses the same license.

## Install and test

Python 3.12 is required by the pinned ZEOS source. `uv.lock` fixes transitive versions as well as the explicitly pinned packages. No modifications to ZEOS or Gemma are required.

```bash
uv sync --frozen --extra dev --extra data --extra cloud
uv run pytest -q
uv run python scripts/gate_demo.py
```

The gate-only demo uses an authored semantic command and the real kernel. It is independent of Gemma downloads and is explicitly not an end-to-end learned-policy demonstration. Offline tests include a genuine tiny random Gemma4 multimodal model to check the Transformers contract, not a substitute for verifying the pretrained E4B weights.

## GPU stages on Modal

The Gemma 4 E4B model is public and ungated. Ensure the token in Modal's `huggingface` secret can read your private dataset. The app uses an L40S and an explicit 30-minute maximum per expensive stage; nothing is deployed as a persistent service. Weights, features and checkpoints persist in the `zeos-vla-research` Modal volume.

```bash
uv run modal run scripts/inspect_dataset.py
uv run modal run scripts/modal_run.py --stage gemma
uv run modal run scripts/modal_run.py --stage head
uv run modal run scripts/modal_run.py --stage dataset
uv run modal run scripts/modal_run.py --stage cache
uv run modal run scripts/modal_run.py --stage train --steps 1000
uv run modal run scripts/modal_run.py --stage demo
```

Run these in order. `gemma` exercises image/text generation, final-layer extraction, finite representations, complete freezing and measured CUDA peak memory/prefill/generation latency. `head` measures a synthetic head benchmark separately. `dataset` decodes the actual recorded wrist frame, verifies shapes and boundary masks. `cache` serializes original unprojected BF16 features and checks a live recomputation against the stored tensor exactly. `train` optimizes only the conditional head on recorded actions; `demo` reloads that checkpoint and demonstrates the same original encoding under veto and allow, with feedback to Gemma under veto.

Download artifacts if desired:

```bash
uv run modal volume get zeos-vla-research checkpoints/package.pt checkpoints/package.pt
uv run modal volume get zeos-vla-research sample.png reports/dataset-sample.png
uv run modal volume get zeos-vla-research complete-demo.json reports/complete-demo.json
```

## Local training and inference

Set `HF_TOKEN` in your environment (do not commit it), then use an NVIDIA GPU with sufficient measured capacity:

```bash
uv run python scripts/prepare_cache.py --config configs/package.json --cache cache/features
uv run python scripts/train.py --cache cache/features --checkpoint checkpoints/package.pt --steps 1000
uv run python scripts/demo.py --checkpoint checkpoints/package.pt --image reports/dataset-sample.png \
  --proprio '[0.02197,26.1043,64.0742,-70.9738,10.3495,7.4050,107.7352]'
```

The state values above belong to the first recorded frame used by the pilot. Use the contemporaneous state for any other image. Change `configs/gate.json` to allow package manipulation and rerun to dispatch a chunk. Do not confuse generated joint changes with task success.

## Learning details

Gemma stays frozen for both training and inference. The workflow is: verify its image/text features and structured command generation, cache the original features for recorded frames, train only the action head against recorded action chunks, then evaluate on held-out episodes and through the gate. Our head starts from random weights; Gemma is pretrained but has no learned B601 motor interface. Freezing makes offline caching valid and keeps the semantic generator unchanged. Backbone adaptation is a possible future experiment if task evaluation reveals inadequate visual features, not a prerequisite for this movement/veto demonstration. A successful gate demonstration or low action loss does not establish closed-loop manipulation competence.

The head is custom assembly around standard PyTorch `nn.TransformerDecoderLayer` / `nn.TransformerDecoder`, not a custom attention kernel. It uses four attention heads per layer. Queries come from noisy action tokens; keys and values are learned projections of Gemma’s frozen final states, not its internal KV cache. Action self-attention is bidirectional across the chunk.

LeRobot 0.6.1 SmolVLA also implements cross-attention and flow matching, but its [packaged expert](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/policies/smolvla/smolvlm_with_expert.py) is coupled to SmolVLM layers and per-layer KV/cache representations. It is not a drop-in final-hidden-state action head for Approach A. This small standalone head keeps the Gemma boundary explicit and retains the verified checkpoint. LeRobot is used for recorded dataset loading; no SmolVLM is loaded.

A two-layer, width-128 transformer decoder has self-attention over 16 action tokens and cross-attention to Gemma's original final-layer token representations. A trained projection reduces the Gemma feature dimension only inside the head. Time Fourier features, learned step positions and projected proprioception condition the velocity field. Training interpolates Gaussian noise and recorded normalized actions with `x(t)=(1-t)noise+t*action`; the squared velocity error targets `action-noise`. Euler integration from noise to actions samples the chunk.

Normalization statistics are fitted only on valid training actions and training proprioception. Padding is excluded from the flow loss. Training and validation episodes are disjoint. Checkpoints contain architecture, weights, normalization, dataset/model revisions, sample manifests and random seed. Validation reports flow loss and normalized sampled-action MSE; neither metric establishes closed-loop success.

Cache keys bind the original RGB image, instruction, semantic prompt version, model revision, dataset revision and frame identity. Each stored tensor has a checksum. Variable-length feature masks prevent padded tokens from entering cross-attention.

## Project continuity and compute budget

The user approved commits, installing/starting/using Isaac Sim or Isaac Lab locally or on Modal, and up to US$25 for training. `HANDOVER.md` records the completed milestone, permissions, exact artifacts, current blockers and next steps; update it and this README as the project advances. `configs/budget.json` tracks conservative planning allocations separately from actual billing. Use bounded runs and avoid unattended GPU services.

## Simulation and Isaac Sim

The demonstrated simulator records and applies joint targets in the dataset's original units. It has no contact dynamics, collision model or package-success metric. No hardware is contacted.

Isaac Sim 6.1.0.0 replay passed on Modal L40S with NVIDIA Vulkan and driver 610.57.04. The Mac host is unsupported. The pinned manufacturer [B601-RS URDF/STL assets](https://github.com/Seeed-Projects/reBot-DevArm/tree/79f01c7056d5f667beba7e838593ea6c299388c5/Rebot_Arm_description/RS) were imported unmodified using the current `URDFImporter`/`URDFImporterConfig` APIs, explicitly selecting the generated `Physics=physx` variant. The controller locates the actual articulation root and checks all eight DOF names. One recorded gripper component drives two slider DOFs for display.

The user deferred physical calibration. `display_mapping.py` applies arbitrary bounded deltas around a neutral pose: 0.05 radians per recorded arm unit and 1% of each slider's travel per recorded gripper unit. Gravity is zero, the base is fixed, and the box/red packages are approximate static props. These scales and slowed playback demonstrate visible motion; they do not reconstruct the recorded physical trajectory or establish package grasping. The veto dispatched zero trajectory targets and had exactly zero measured state drift; allow dispatched 16 targets and produced measured articulation motion. View [allow animation](reports/isaac-replay/allow.gif), [veto animation](reports/isaac-replay/veto.gif) and [report](reports/isaac-replay/report.json).

Isaac runs separately from the verified Gemma environment because it requires a different PyTorch dependency set (2.11.0 versus 2.7.1). It replays the previously generated trained-head output from the complete demo, records that source file's SHA256, and performs fresh real ZEOS authorization before trajectory dispatch. It does not run live Gemma on the rendered camera images. The full original live Gemma → gate → head sequence is verified by `demo-verification.json`; Isaac articulation playback is a separate verified extension.

After completing the six stages above:

```bash
uv run python scripts/fetch_b601_assets.py
uv run modal run scripts/modal_isaac.py --mode smoke
uv run modal run scripts/modal_isaac.py --mode replay
```

The worker saves its report, render GIFs/PNGs, kernel journals, actual package versions and native log in `reports/isaac-replay/`. Initial normal-cleanup runs aborted or segfaulted; their evidence is retained. The isolated worker uses NVIDIA's documented `SimulationApp.close(skip_cleanup=True, exit_code=...)` immediate exit, and its supervisor checks the process status. This is not a claim that normal cleanup is fixed. See [current lifecycle API](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/py/source/extensions/isaacsim.simulation_app/docs/index.html), [importer API](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/py/source/extensions/isaacsim.asset.importer.urdf/docs/index.html), and `THIRD_PARTY.md` for asset attribution.

## Status and evidence

All six stages have been executed successfully with Gemma 4 and recorded actions. The real ZEOS veto skipped the head and dispatched zero steps; allow dispatched one 16-step joint-state chunk. On L40S the complete first allow cycle measured 2.286 seconds and peak allocated GPU memory was 14.89 GiB. This is a gating demonstration, not real-time or contact-based manipulation validation.

See `reports/STATUS.md` for executed checks, measured results and outstanding blockers. Reports distinguish synthetic tests, real dataset reads, real kernel enforcement, pretrained backbone runs, Isaac Sim and hardware. An absent GPU result is not an assumed acceptable result.

An offline head-and-gate smoke demonstration is also available:

```bash
uv run python scripts/head_gate_demo.py
```

It trains a real flow head on explicitly synthetic fixtures, reloads its checkpoint and uses an authored package command through the real ZEOS kernel. Under allow, the head samples and dispatches 16 simulated steps. Under veto, there are zero head calls and zero dispatched steps. This smoke demonstration does not use pretrained Gemma, recorded training or Gemma feedback; the complete `demo` stage above is required to validate those integrations.
