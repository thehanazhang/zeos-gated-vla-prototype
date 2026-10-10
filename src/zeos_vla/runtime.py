"""Gate-before-sampling orchestration with a deliberately minimal simulated robot."""

from dataclasses import asdict
import time
import torch
from zeos_vla.semantic import SemanticCommand


class JointStateSimulator:
    """Recorded-unit joint targets, no dynamics/contact model and no physical IO."""

    def __init__(self, proprio):
        self.state = proprio.detach().cpu().clone().flatten()
        self.chunks = []

    def dispatch(self, chunk):
        actions = chunk.detach().cpu()
        if actions.shape != (16, len(self.state)) or not torch.isfinite(actions).all():
            raise ValueError("Invalid simulated action chunk")
        self.chunks.append(actions.clone())
        self.state = actions[-1].clone()


def run_encoded(
    backbone, policy, gate, robot, encoding, image, instruction, proprio, seed=0
):
    result = {
        "command_text": encoding.command_text,
        "backbone_metrics": encoding.metrics,
        "head_called": False,
        "dispatched_steps": 0,
        "feedback": None,
        "allowed": False,
    }
    try:
        command = SemanticCommand.parse(encoding.command_text)
        gate_start = time.perf_counter()
        decision = gate.authorize(command)
        result["gate_seconds"] = time.perf_counter() - gate_start
        result["gate"] = asdict(decision)
    except Exception as exc:
        reason = f"Command validation/runtime failure: {type(exc).__name__}: {exc}"
        result["reason"] = reason
        try:
            result["feedback"] = backbone.feedback(image, instruction, reason)
        except Exception as feedback_error:
            result["feedback_error"] = type(feedback_error).__name__
        return result
    result["reason"] = decision.reason
    if not decision.allowed:
        try:
            feedback_start = time.perf_counter()
            result["feedback"] = backbone.feedback(image, instruction, decision.reason)
            result["feedback_seconds"] = time.perf_counter() - feedback_start
        except Exception as exc:
            result["feedback_error"] = type(exc).__name__
        return result
    # The exact original encoding is retained through authorization. No feedback
    # representations replace it and no denied branch calls the trained policy.
    device = policy.device
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    try:
        result["head_called"] = True
        chunk = policy.predict(
            encoding.features, encoding.mask, proprio.reshape(1, -1), seed=seed
        )[0]
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        result["head_seconds"] = time.perf_counter() - start
        if chunk.shape != (16, len(proprio)) or not torch.isfinite(chunk).all():
            raise ValueError("Policy produced an invalid chunk")
        robot.dispatch(chunk)
        result.update(
            allowed=True,
            dispatched_steps=16,
            actions=chunk.cpu().tolist(),
            joint_delta_norm=float(
                torch.linalg.vector_norm(chunk[-1].cpu() - proprio.cpu())
            ),
        )
    except Exception as exc:
        result["reason"] = f"Policy/dispatch failure: {type(exc).__name__}: {exc}"
    return result


def observe_and_act(backbone, policy, gate, robot, image, instruction, proprio, seed=0):
    try:
        encoding = backbone.encode(image, instruction)
    except Exception as exc:
        return {
            "allowed": False,
            "head_called": False,
            "dispatched_steps": 0,
            "reason": f"Backbone failed: {type(exc).__name__}",
        }
    return run_encoded(
        backbone, policy, gate, robot, encoding, image, instruction, proprio, seed
    )
