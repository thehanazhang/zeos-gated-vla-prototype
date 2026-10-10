import json
import pytest
from zeos_vla.gate import ZeosActionGate
from zeos_vla.semantic import SemanticCommand

COMMAND = SemanticCommand("manipulate", "package", "slow")


def test_real_kernel_allow_and_veto():
    allowed = ZeosActionGate(True).authorize(COMMAND)
    assert allowed.allowed and allowed.released_command == COMMAND.json(), (
        allowed.reason
    )
    denied = ZeosActionGate(False).authorize(COMMAND)
    assert not denied.allowed and denied.released_command is None
    kinds = [json.loads(line)["kind"] for line in allowed.journal]
    assert "nli.gate_consulted" in kinds and "nli.gate_answered" in kinds
    assert "package" in denied.reason


@pytest.mark.parametrize("failure", ["malformed", "timeout", "crash", "exit"])
def test_failure_closed(failure):
    decision = ZeosActionGate(True, failure=failure).authorize(COMMAND)
    assert not decision.allowed and decision.released_command is None


@pytest.mark.parametrize(
    "text",
    [
        "{}",
        '{"action":"manipulate","target":"package","speed":"slow","extra":1}',
        '{"action":"manipulate","action":"manipulate","target":"package","speed":"slow"}',
        '{"action":"manipulate","target":"package; write tools go","speed":"slow"}',
        "```json\n{}\n```",
    ],
)
def test_invalid_semantic_commands(text):
    with pytest.raises(ValueError):
        SemanticCommand.parse(text)


def test_fast_command_is_vetoed_even_when_enabled():
    assert (
        not ZeosActionGate(True)
        .authorize(SemanticCommand("manipulate", "package", "fast"))
        .allowed
    )
