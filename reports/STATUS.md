# Verification status — 11 October 2026

Implemented repository; pretrained end-to-end verification and recorded-action training are pending gated Gemma access.

| Component | Executed evidence | Status |
|---|---|---|
| Pretrained `google/gemma-3-4b-it` download | Modal returned Hugging Face HTTP 403: account not authorized | Blocked |
| Transformers multimodal generation/final states | Genuine tiny random Gemma3, image tokens + pixels, final features, native generation, no backbone gradients | Offline contract passed; pretrained 4B unverified |
| Flow head | Shape, finite loss, all gradients, cross-attention gradient, masked features, synthetic optimization | Passed |
| Actual LeRobot adapter | User's pinned private dataset decoded on Modal; 120 selected samples from episodes 0–14; wrist RGB; 16×7 actions; episode-end mask | Passed |
| Feature cache | Tensor round-trip, checksum tampering rejection, instruction/revision mismatch | Unit passed; pretrained live/cache comparison pending |
| Training/checkpoints | Short synthetic cache training, episode-overlap rejection, checkpoint reload and deterministic policy inference | Unit passed; recorded-action training pending |
| Real ZEOS gate | Real `Kernel` + `CommandSeat` + `GateSpec`, held device write, allow/veto | Passed |
| Failure closed | Malformed verdict, timeout, guard crash and early exit | Passed |
| Head + gate smoke demo | Trained real head on synthetic fixtures, authored command, real ZEOS, joint-state simulation | Passed: allow 1 head call / 16 dispatched steps; veto 0 / 0 |
| Complete pretrained runtime and Gemma feedback | Implemented executable Modal stage | Pending |
| Isaac Sim | No robot USD/scene or verified units supplied | Not run |
| Robot hardware | No hardware accessed | Not run |

22 tests pass locally on Python 3.12.13 / macOS arm64. See the tests for fixture scope. No synthetic feature tensor or tiny random model is presented as a pretrained Gemma result.

## Measured GPU evidence

`head-verification.json` records the independent synthetic-conditioning head benchmark on NVIDIA L40S. The first measured run sampled a 16×7 chunk in 36.7 ms, with 165,560,320 bytes peak allocated (157.9 MiB) and 201,326,592 bytes reserved (192 MiB). A second run under the exported lock updates that JSON and may differ slightly. The synthetic benchmark uses batch 4, 1,024 feature tokens and feature dimension 2,560; sample timing uses batch 1 and 32 Euler steps after warmup. Peak memory includes the head, input tensors, optimizer and training allocations. It does not include Gemma.

Gemma GPU memory, multimodal prefill latency, semantic generation latency, real-feature cache timings and complete policy-loop latency are **unmeasured** because model access failed before loading. No claim is made about their acceptability or real-time capacity.

## Reproduction

Run the documented Modal stages after enabling access to the gated checkpoint for the account/token in the existing `huggingface` Modal secret. The private dataset is already readable by that secret. Initial access-failure run: `ap-7ndGXn0b2Qm0mMU39IbRIX`; actual dataset-validation run: `ap-JLBbJsUaXdPQTxGmqFp3RR`; initial GPU head benchmark: `ap-bQyBlv83Vwyz4ayZLOrqDP`.

The requested development order was followed for the initial Gemma verification attempt. After its external access block, independent head tests, dataset validation and gate integration proceeded. Recorded-feature training and the complete demo were not bypassed with substitute model features.
