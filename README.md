# ZEOS-gated Gemma action prototype

Approach A: one completely frozen `google/gemma-3-4b-it` produces a structured semantic proposal and the final hidden states of its original image/instruction prompt. The semantic proposal alone enters the real Metacognition ZEOS kernel. A separate deterministic guard job checks the proposal. Only an authorized device-pipe write permits the flow-matching head to sample a 16-step action chunk using the original Gemma representations and robot proprioception. A veto returns feedback to the same Gemma and dispatches zero actions.

No Gemma KV-cache modifications, embedding interpretation by ZEOS, second reasoning VLM, or trained semantic decoder. Native Transformers generation uses its normal cache. Feature extraction uses a separate cache-free forward through Gemma's base model, which exposes `last_hidden_state` directly and avoids retaining every layer.

## Data decision

Use recorded actions for the action head, not arbitrary plausible numbers. Synthetic tensors test dimensions, optimization and gradient flow. They cannot establish learned manipulation. This prototype uses the user's private `thehanazhang/b601_rs_manipulate_package_v3_150`, pinned to `d18511446db55a41bf3ae5d1ded06d05ddbbd835`. Metadata: LeRobot v3.0, 150 episodes, 39,559 frames, 30 Hz, wrist RGB 640×480, seven state/action components (six arm joints plus gripper).

The pilot uses 12 training episodes and 3 disjoint validation episodes, sparsely sampled. It is a recorded-action feasibility experiment, not evidence of successful package manipulation. Do not interpret recorded `.pos` values as radians without confirming the collection configuration and calibration. A 16-frame chunk at 30 Hz spans about 0.53 seconds.

## Gate policy

`configs/gate.json` defaults to rejecting package manipulation. Set `allow_package_manipulation` to `true` to allow the same proposal. The policy is owned by the application guard; enforcement (holding writes, consulting jobs, verdict parsing, failure and timeout handling) belongs to the unmodified ZEOS kernel. This demonstrates permission gating, not collision avoidance or a proof that the sampled trajectory implements the semantic command.

## Upstream API references

- [Transformers 4.53.3 Gemma implementation](https://github.com/huggingface/transformers/blob/v4.53.3/src/transformers/models/gemma3/modeling_gemma3.py): `Gemma3ForConditionalGeneration.model` returns the multimodal final hidden states; `AutoProcessor.apply_chat_template` handles image and text.
- [LeRobot 0.4.4 dataset implementation](https://github.com/huggingface/lerobot/blob/v0.4.4/src/lerobot/datasets/lerobot_dataset.py): `delta_timestamps`, episode-boundary padding, `video_backend="pyav"`.
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

The gate-only demo uses an authored semantic command and the real kernel. It is independent of Gemma access and is explicitly not an end-to-end learned-policy demonstration. Offline tests include a genuine tiny random Gemma3 multimodal model to check the Transformers contract, not a substitute for verifying the pretrained 4B weights.

## GPU stages on Modal

Accept the Gemma model terms on Hugging Face and ensure the token in Modal's `huggingface` secret can read both the gated model and your private dataset. The app uses an L40S and an explicit 30-minute maximum per expensive stage; nothing is deployed as a persistent service. Weights, features and checkpoints persist in the `zeos-vla-research` Modal volume.

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

A two-layer, width-128 transformer decoder has self-attention over 16 action tokens and cross-attention to Gemma's original final-layer token representations. A trained projection reduces the Gemma feature dimension only inside the head. Time Fourier features, learned step positions and projected proprioception condition the velocity field. Training interpolates Gaussian noise and recorded normalized actions with `x(t)=(1-t)noise+t*action`; the squared velocity error targets `action-noise`. Euler integration from noise to actions samples the chunk.

Normalization statistics are fitted only on valid training actions and training proprioception. Padding is excluded from the flow loss. Training and validation episodes are disjoint. Checkpoints contain architecture, weights, normalization, dataset/model revisions, sample manifests and random seed. Validation reports flow loss and normalized sampled-action MSE; neither metric establishes closed-loop success.

Cache keys bind the original RGB image, instruction, semantic prompt version, model revision, dataset revision and frame identity. Each stored tensor has a checksum. Variable-length feature masks prevent padded tokens from entering cross-attention.

## Simulation and Isaac Sim

The demonstrated simulator records and applies joint targets in the dataset's original units. It has no contact dynamics, collision model or package-success metric. No hardware is contacted.

An Isaac Sim scene is a subsequent integration step: obtain/import a B601 RS URDF/USD with collision geometry and joint limits; confirm the six arm joint values' units, signs and offsets plus the gripper conversion; match the wrist camera and package scene; then connect the simulator dispatch interface to an articulation controller. Isaac Sim angular targets use radians and its gripper articulation may differ from the recorded motor-angle field. Blindly converting all seven columns as arm angles is incorrect. See the [Isaac Sim 5.1 articulation controller documentation](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/robot_simulation/articulation_controller.html).

## Status and evidence

See `reports/STATUS.md` for executed checks, measured results and outstanding blockers. Reports distinguish synthetic tests, real dataset reads, real kernel enforcement, pretrained backbone runs, Isaac Sim and hardware. An absent GPU result is not an assumed acceptable result.

An offline head-and-gate smoke demonstration is also available:

```bash
uv run python scripts/head_gate_demo.py
```

It trains a real flow head on explicitly synthetic fixtures, reloads its checkpoint and uses an authored package command through the real ZEOS kernel. Under allow, the head samples and dispatches 16 simulated steps. Under veto, there are zero head calls and zero dispatched steps. This smoke demonstration does not use pretrained Gemma, recorded training or Gemma feedback; the complete `demo` stage above is required to validate those integrations.
