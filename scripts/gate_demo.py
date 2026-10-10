"""Real ZEOS gate only: no backbone or learned-policy claims."""

import json
from dataclasses import asdict
from pathlib import Path
from zeos_vla.gate import ZeosActionGate
from zeos_vla.semantic import SemanticCommand

if __name__ == "__main__":
    command = SemanticCommand("manipulate", "package", "slow")
    report = {
        "scope": "Authored semantic command through the real ZEOS kernel; no VLM or action head",
        "cases": {},
    }
    for name, allow, failure in [
        ("allow", True, None),
        ("veto", False, None),
        ("timeout", True, "timeout"),
        ("malformed", True, "malformed"),
        ("crash", True, "crash"),
        ("early_exit", True, "exit"),
    ]:
        report["cases"][name] = asdict(
            ZeosActionGate(allow, failure).authorize(command)
        )
        print(name, report["cases"][name]["allowed"], report["cases"][name]["reason"])
    out = Path("reports/gate-demonstration.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
