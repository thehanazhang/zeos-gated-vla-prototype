import json
import torch
from PIL import Image
import pytest
from zeos_vla.cache import FeatureCache, provenance, cache_key, collate
from zeos_vla.data import LeRobotAdapter
from zeos_vla.head import FlowHead, HeadConfig
from zeos_vla.backbone import ObservationEncoding, PROMPT_VERSION
from zeos_vla.training import save_checkpoint, Policy, fit_stats, train
from zeos_vla.runtime import JointStateSimulator, run_encoded, observe_and_act
from zeos_vla.gate import ZeosActionGate


@pytest.fixture
def sample():
    return {
        "image": Image.new("RGB", (16, 16), "red"),
        "instruction": "Manipulate the package slowly.",
        "episode_index": 0,
        "frame_index": 0,
        "actions": torch.randn(16, 7),
        "proprio": torch.randn(7),
        "action_mask": torch.ones(16, dtype=torch.bool),
    }


def test_cache_exact_and_provenance(tmp_path, sample):
    cache = FeatureCache(tmp_path)
    meta = provenance(sample, "fixture", "abc")
    f = torch.randn(5, 24, dtype=torch.bfloat16)
    mask = torch.ones(5, dtype=torch.bool)
    key = cache.write(meta, f, mask, sample)
    stored = cache.read(key, meta)
    assert torch.equal(f, stored["features"]) and not stored["features"].requires_grad
    assert cache_key(meta) != cache_key({**meta, "instruction": "different task"})
    assert cache_key(meta) != cache_key({**meta, "dataset_revision": "other"})
    with pytest.raises(ValueError):
        cache.read(key, {**meta, "prompt_version": "other"})
    stored["features"][0, 0] += 1
    torch.save(stored, tmp_path / (key + ".pt"))
    with pytest.raises(ValueError, match="checksum"):
        cache.read(key)


def test_adapter_actions_and_episode_padding(sample):
    item = {
        "observation.images.wrist": torch.rand(3, 12, 16),
        "observation.state": sample["proprio"],
        "action": sample["actions"],
        "action_is_pad": torch.tensor([False] * 12 + [True] * 4),
        "episode_index": 3,
        "frame_index": 24,
        "task": "move package",
    }
    adapted = LeRobotAdapter.adapt(item, "observation.images.wrist")
    assert adapted["actions"].shape == (16, 7) and adapted["action_mask"].sum() == 12
    assert adapted["episode_index"] == 3 and adapted["image"].size == (16, 12)


def test_checkpoint_policy_and_training(tmp_path, sample):
    torch.set_num_threads(1)
    torch.manual_seed(1)
    cache = FeatureCache(tmp_path)
    keys = []
    for ep in [0, 1]:
        s = {**sample, "episode_index": ep}
        meta = provenance(s, "fixture", "abc")
        keys.append(
            cache.write(meta, torch.randn(5, 24), torch.ones(5, dtype=torch.bool), s)
        )
    for name, ep, key in [("train", 0, keys[0]), ("val", 1, keys[1])]:
        (tmp_path / (name + ".json")).write_text(
            json.dumps(
                {
                    "keys": [key],
                    "episodes": [ep],
                    "dataset_id": "fixture",
                    "dataset_revision": "abc",
                }
            )
        )
    report = train(
        tmp_path,
        tmp_path / "train.json",
        tmp_path / "val.json",
        tmp_path / "model.pt",
        steps=2,
        batch_size=1,
        device="cpu",
    )
    policy = Policy(tmp_path / "model.pt")
    assert report["validation_flow_loss"] > 0
    record = cache.read(keys[0])
    batch = collate([record])
    a = policy.predict(
        batch["features"], batch["feature_mask"], batch["proprio"], steps=2
    )
    b = policy.predict(
        batch["features"], batch["feature_mask"], batch["proprio"], steps=2
    )
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert a.shape == (1, 16, 7)
    head = FlowHead(HeadConfig(feature_dim=24))
    stats = fit_stats([record])
    with pytest.raises(ValueError):
        save_checkpoint(
            tmp_path / "bad.pt",
            head,
            stats,
            {"model_revision": "bad", "prompt_version": PROMPT_VERSION},
        )
        Policy(tmp_path / "bad.pt")
    (tmp_path / "val.json").write_text(
        json.dumps(
            {
                "keys": [keys[0]],
                "episodes": [0],
                "dataset_id": "fixture",
                "dataset_revision": "abc",
            }
        )
    )
    with pytest.raises(ValueError, match="leakage"):
        train(
            tmp_path,
            tmp_path / "train.json",
            tmp_path / "val.json",
            tmp_path / "x.pt",
            steps=1,
            device="cpu",
        )


class BackboneSpy:
    def __init__(self):
        self.feedbacks = []

    def feedback(self, image, instruction, reason):
        self.feedbacks.append(reason)
        return "Acknowledged veto"


class PolicySpy:
    device = torch.device("cpu")

    def __init__(self):
        self.calls = 0
        self.features = None

    def predict(self, features, mask, proprio, seed=0):
        self.calls += 1
        self.features = features
        return proprio[:, None, :].expand(-1, 16, -1) + 0.1


@pytest.mark.parametrize("allow", [False, True])
def test_runtime_real_gate_precedes_head_and_dispatch(allow):
    encoding = ObservationEncoding(
        torch.randn(1, 5, 24),
        torch.ones(1, 5, dtype=torch.bool),
        '{"action":"manipulate","target":"package","speed":"slow"}',
        {},
    )
    model = BackboneSpy()
    policy = PolicySpy()
    robot = JointStateSimulator(torch.zeros(7))
    result = run_encoded(
        model,
        policy,
        ZeosActionGate(allow),
        robot,
        encoding,
        None,
        "task",
        torch.zeros(7),
    )
    assert result["allowed"] == allow
    assert policy.calls == int(allow) and len(robot.chunks) == int(allow)
    if allow:
        assert policy.features is encoding.features and not model.feedbacks
    else:
        assert model.feedbacks and result["dispatched_steps"] == 0


def test_runtime_invalid_command_and_broken_backbone():
    encoding = ObservationEncoding(
        torch.randn(1, 5, 24), torch.ones(1, 5, dtype=torch.bool), "bad", {}
    )
    model = BackboneSpy()
    policy = PolicySpy()
    robot = JointStateSimulator(torch.zeros(7))
    result = run_encoded(
        model,
        policy,
        ZeosActionGate(True),
        robot,
        encoding,
        None,
        "task",
        torch.zeros(7),
    )
    assert not result["allowed"] and policy.calls == 0 and not robot.chunks
    result = observe_and_act(
        model, policy, ZeosActionGate(True), robot, None, "task", torch.zeros(7)
    )
    assert not result["allowed"] and result["dispatched_steps"] == 0


def test_nonfinite_policy_output_never_dispatches():
    encoding = ObservationEncoding(
        torch.randn(1, 5, 24),
        torch.ones(1, 5, dtype=torch.bool),
        '{"action":"manipulate","target":"package","speed":"slow"}',
        {},
    )
    policy = PolicySpy()
    policy.predict = lambda *args, **kwargs: torch.full((1, 16, 7), float("nan"))
    robot = JointStateSimulator(torch.zeros(7))
    result = run_encoded(
        BackboneSpy(),
        policy,
        ZeosActionGate(True),
        robot,
        encoding,
        None,
        "task",
        torch.zeros(7),
    )
    assert not result["allowed"] and result["head_called"]
    assert result["dispatched_steps"] == 0 and not robot.chunks
