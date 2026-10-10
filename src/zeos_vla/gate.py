"""Application guard job implemented through the real, pinned ZEOS CommandSeat."""

from dataclasses import dataclass
import time
from zeos.core.events import GateAnswered, GateConsulted, PipeWritten
from zeos.core.gates import GateSpec, GateTable, VETO, gate_problems
from zeos.core.ids import DescriptorName, PipeName, Principal, Ring
from zeos.core.kernel import Kernel, KernelConfig
from zeos.core.pipes import PipeSpec, PipeTable
from zeos.core.resources import ResourceTable
from zeos.core.vectors import VectorTable
from zeos.descriptor.schema import Descriptor
from zeos.machine.seat import CommandSeat
from zeos.world.store import WorldStore
from zeos.journal.codec import to_line
from zeos_vla.semantic import SemanticCommand

ACTUATOR = PipeName("actuators.simulated_robot")
REQUESTS = PipeName("gate.package.requests")
VERDICTS = PipeName("gate.package.verdicts")


@dataclass(frozen=True)
class GateDecision:
    allowed: bool
    reason: str
    released_command: str | None
    journal: tuple[str, ...]
    latency_seconds: float


class PackageGuardSource:
    """No reasoning model: policy is a deterministic, separately owned guard job.

    The proposal is read from the kernel request pipe via on_arrival; it is not
    chosen by a precomputed allow/veto tape. Kernel enforcement is never replaced.
    """

    def __init__(self, command, allow_package_manipulation=False, failure=None):
        self.command = command
        self.allow_package_manipulation = allow_package_manipulation
        self.failure = failure
        self.arrivals = {}

    def arrival(self, job, text):
        self.arrivals[job] = text

    def next_command(self, turn):
        if turn.descriptor == "planner":
            return f"write tools {self.command};" if turn.issued == 0 else "exit;"
        if turn.descriptor == "clock":
            return "say tick;" if turn.issued < 100 else "exit;"
        if turn.issued == 0:
            return "read stdin;"
        if turn.issued == 1:
            if self.failure == "timeout":
                return "read stdin;"
            if self.failure == "crash":
                raise RuntimeError("Injected guard failure")
            if self.failure == "exit":
                return "exit;"
            if self.failure == "malformed":
                return "write stdout uncertain;"
            try:
                proposal = SemanticCommand.parse(self.arrivals.get(turn.job, ""))
                if proposal.speed != "slow":
                    verdict = "veto: fast movement is disabled"
                elif not self.allow_package_manipulation:
                    verdict = "veto: package manipulation is disabled"
                else:
                    verdict = "allow: package manipulation enabled"
            except (ValueError, TypeError):
                verdict = "veto: invalid semantic command"
            return f"write stdout {verdict};"
        return "exit;"


class ZeosActionGate:
    def __init__(self, allow_package_manipulation=False, failure=None):
        if type(allow_package_manipulation) is not bool:
            raise ValueError("Gate flag must be a boolean")
        self.allow = allow_package_manipulation
        self.failure = failure

    def authorize(self, command: SemanticCommand):
        start = time.perf_counter()
        payload = command.json()
        source = PackageGuardSource(payload, self.allow, self.failure)
        descriptors = [
            {
                "name": "planner",
                "priority": 50,
                "pipes": {"tools": str(ACTUATOR)},
                "capabilities": [{"pipe": str(ACTUATOR), "min_integrity": 2}],
            },
            {
                "name": "package-guard",
                "priority": 10,
                "pipes": {"stdin": str(REQUESTS), "stdout": str(VERDICTS)},
                "capabilities": [{"pipe": str(VERDICTS), "min_integrity": 2}],
            },
            {"name": "clock", "priority": 60, "pipes": {"stdout": "clock.ticks"}},
        ]
        desc = {
            DescriptorName(d["name"]): Descriptor.from_frontmatter(
                d, body="Deterministic package guard demo"
            )
            for d in descriptors
        }
        pipes = PipeTable(
            [
                PipeSpec(ACTUATOR, ring=Ring.TRUSTED, principal=Principal.DEVICE),
                PipeSpec(REQUESTS, ring=Ring.KERNEL, principal=Principal.KERNEL),
                PipeSpec(VERDICTS, ring=Ring.TRUSTED, principal=Principal.PEER_JOB),
                PipeSpec(PipeName("clock.ticks")),
            ]
        )
        spec = GateSpec(
            ACTUATOR,
            DescriptorName("package-guard"),
            REQUESTS,
            VERDICTS,
            on_gate_failure=VETO,
            timeout_ticks=48,
        )
        gates = GateTable([spec])
        problems = gate_problems(
            gates,
            descriptors=desc,
            pipes=[ACTUATOR, REQUESTS, VERDICTS, PipeName("clock.ticks")],
        )
        if problems:
            raise RuntimeError("Invalid ZEOS wiring: " + "; ".join(problems))
        events = []
        kernel = Kernel(
            descriptors=desc,
            machine=CommandSeat(source=source, on_arrival=source.arrival),
            pipes=pipes,
            vectors=VectorTable(),
            world=WorldStore(),
            resources=ResourceTable(),
            gates=gates,
            journal_sink=events,
            config=KernelConfig(case="zeos-vla", max_ticks=800),
        )
        try:
            kernel.start()
            kernel.spawn(DescriptorName("planner"))
            kernel.spawn(DescriptorName("clock"))
            for tick in range(800):
                kernel.advance_time(tick * 1_000_000)
                if not kernel.tick():
                    break
        except Exception as exc:
            # Runtime exception is denial even if an earlier journal looked permissive.
            return GateDecision(
                False,
                f"ZEOS runtime failed: {type(exc).__name__}",
                None,
                tuple(to_line(i, e) for i, e in enumerate(events)),
                time.perf_counter() - start,
            )
        answered = [
            e for e in events if isinstance(e, GateAnswered) and e.pipe == ACTUATOR
        ]
        released = [
            e for e in events if isinstance(e, PipeWritten) and e.pipe == ACTUATOR
        ]
        consulted = [
            e for e in events if isinstance(e, GateConsulted) and e.pipe == ACTUATOR
        ]
        actual = "".join(released[0].text).strip() if len(released) == 1 else None
        # Exact payload, exactly one verdict and release, correct journal ordering.
        allowed = bool(
            len(answered) == 1
            and answered[0].allowed
            and len(consulted) == 1
            and actual == payload
            and events.index(consulted[0])
            < events.index(answered[0])
            < events.index(released[0])
        )
        reason = answered[0].reason if answered else "No gate verdict; deny"
        return GateDecision(
            allowed,
            reason,
            actual if allowed else None,
            tuple(to_line(i, e) for i, e in enumerate(events)),
            time.perf_counter() - start,
        )
