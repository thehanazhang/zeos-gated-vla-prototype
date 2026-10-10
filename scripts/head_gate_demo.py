"""Offline synthetic head + real ZEOS demonstration; deliberately no VLM claims."""

import json
from pathlib import Path
import torch
from zeos_vla.backbone import MODEL_REVISION, PROMPT_VERSION
from zeos_vla.head import FlowHead, HeadConfig
from zeos_vla.training import save_checkpoint, Policy
from zeos_vla.gate import ZeosActionGate
from zeos_vla.semantic import SemanticCommand
from zeos_vla.runtime import JointStateSimulator

if __name__ == "__main__":
    torch.manual_seed(42)
    torch.set_num_threads(1)
    config = HeadConfig(feature_dim=24, width=32, layers=1)
    head = FlowHead(config)
    f = torch.randn(1, 8, 24)
    mask = torch.ones(1, 8, dtype=torch.bool)
    state = torch.zeros(1, 7)
    actions = (
        torch.linspace(0, 0.25, 16)[None, :, None]
        * torch.linspace(0.3, 1, 7)[None, None, :]
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.003)
    losses = []
    for _ in range(150):
        optimizer.zero_grad()
        loss = head.loss(actions, f, state, mask)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    stats = {
        "action_mean": torch.zeros(7),
        "action_std": torch.ones(7),
        "proprio_mean": torch.zeros(7),
        "proprio_std": torch.ones(7),
    }
    path = Path("checkpoints/synthetic-smoke.pt")
    save_checkpoint(
        path,
        head,
        stats,
        {
            "model_revision": MODEL_REVISION,
            "prompt_version": PROMPT_VERSION,
            "scope": "Synthetic smoke fixture; no Gemma features or recorded actions",
        },
    )
    policy = Policy(path)
    command = SemanticCommand("manipulate", "package", "slow")
    cases = {}
    for name, enabled in [("veto", False), ("allow", True)]:
        robot = JointStateSimulator(state[0])
        decision = ZeosActionGate(enabled).authorize(command)
        calls = 0
        if decision.allowed:
            calls += 1
            robot.dispatch(policy.predict(f, mask, state)[0])
        cases[name] = {
            "allowed": decision.allowed,
            "reason": decision.reason,
            "head_calls": calls,
            "dispatched_steps": len(robot.chunks) * 16,
            "final_state": robot.state.tolist(),
            "joint_delta_norm": float(torch.linalg.vector_norm(robot.state - state[0])),
            "gate_journal": decision.journal,
        }
    assert cases["veto"]["head_calls"] == 0 and cases["veto"]["dispatched_steps"] == 0
    assert (
        cases["allow"]["head_calls"] == 1 and cases["allow"]["dispatched_steps"] == 16
    )
    assert cases["allow"]["joint_delta_norm"] > 0
    report = {
        "scope": "Synthetic conditioning/action training + authored command + real ZEOS + joint-state simulator",
        "pretrained_gemma_used": False,
        "recorded_training_used": False,
        "gemma_feedback_tested": False,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "cases": cases,
    }
    Path("reports/head-gate-demonstration.json").write_text(
        json.dumps(report, indent=2)
    )
    print(
        json.dumps(
            {
                "scope": report["scope"],
                "cases": {
                    k: {f: v for f, v in x.items() if f != "gate_journal"}
                    for k, x in cases.items()
                },
            },
            indent=2,
        )
    )
