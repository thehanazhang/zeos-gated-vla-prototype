"""Run in Isaac's own Python process, never the Modal control interpreter.

Replay previously generated recorded-action-trained outputs under a fresh real
ZEOS gate. Gemma generation and head training remain in their locked environment.
This is illustrative articulation motion, not closed-loop package manipulation.
"""

from pathlib import Path
import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import subprocess
import time
import traceback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "replay"], default="smoke")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=Path("/cache/complete-demo.json"))
    parser.add_argument("--assets", type=Path, default=Path("/assets/b601-RS"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    from isaacsim import SimulationApp
    simulation = SimulationApp({
        "headless": True, "width": 640, "height": 480,
        "fast_shutdown": True, "multi_gpu": False,
    })
    import omni.usd
    from pxr import UsdGeom, Gf
    stage = omni.usd.get_context().get_stage()
    report = {
        "isaacsim_version": importlib.metadata.version("isaacsim"),
        "headless_started": True, "hardware_tested": False,
        "physical_calibration_verified": False,
        "startup_seconds": time.perf_counter() - started,
    }
    exit_code = 0
    try:
        if args.mode == "smoke":
            cube = UsdGeom.Cube.Define(stage, "/World/SmokeCube")
            cube.CreateSizeAttr(0.1)
            cube.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.1))
            for _ in range(10):
                simulation.update()
            report.update(usd_stage_created=bool(stage.GetPrimAtPath("/World/SmokeCube")), updates=10)
        else:
            report.update(replay(simulation, stage, args))
    except Exception:
        exit_code = 1
        report.update(completed=False, error=traceback.format_exc())
    report["elapsed_before_shutdown_seconds"] = time.perf_counter() - started
    report["gpu"] = subprocess.run([
        "nvidia-smi", "--query-gpu=name,driver_version,memory.used", "--format=csv,noheader",
    ], capture_output=True, text=True, check=True).stdout.strip()
    # Documented isolated-worker exit avoids the observed native cleanup crash.
    # The supervisor checks process exit status; no graceful-cleanup claim.
    report["shutdown_mode"] = "documented close(skip_cleanup=True) immediate exit"
    report["completed"] = exit_code == 0
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    simulation.close(skip_cleanup=True, exit_code=exit_code)


def replay(simulation, stage, args):
    import numpy as np
    import omni.timeline
    import omni.replicator.core as rep
    from PIL import Image
    from pxr import UsdGeom, UsdLux, UsdPhysics, Gf
    from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig
    import isaacsim.core.experimental.utils.stage as stage_utils
    from isaacsim.core.experimental.prims import Articulation
    from isaacsim.core.simulation_manager import SimulationManager
    from zeos_vla.display_mapping import JOINT_NAMES, NEUTRAL, display_targets
    from zeos_vla.gate import ZeosActionGate
    from zeos_vla.semantic import SemanticCommand

    source_bytes = args.source.read_bytes()
    source = json.loads(source_bytes)
    approved = source["cases"]["allow"]
    if not approved["allowed"] or approved["dispatched_steps"] != 16:
        raise ValueError("Source did not contain a completed authorized policy cycle")
    command = SemanticCommand.parse(approved["command_text"])
    targets = display_targets(approved["actions"], source["initial_proprio"])
    import_start = time.perf_counter()
    usd_path = URDFImporter(URDFImporterConfig(
        urdf_path=str(args.assets / "urdf/ReBot_Arm_RS.urdf"),
        usd_path=str(args.output / "robot"), fix_base=True,
        merge_fixed_joints=False, merge_mesh=False,
        joint_drive_type="force", joint_target_type="position",
        override_joint_stiffness=500.0, override_joint_damping=40.0,
    )).import_urdf()
    # The actual 6.1 output has no default Physics selection. Explicitly select
    # its observed physx variant before looking up the articulation root.
    stage_utils.add_reference_to_stage(usd_path=usd_path, path="/World/B601",
                                       variants=[("Physics", "physx")])
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    scene = UsdPhysics.Scene.Define(stage, "/World/physicsScene")
    scene.CreateGravityMagnitudeAttr(0.0)  # motion display, no grasp/contact experiment
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    light = UsdLux.DomeLight.Define(stage, "/World/Light")
    light.CreateIntensityAttr(1200)

    def box(path, center, size, color):
        cube = UsdGeom.Cube.Define(stage, path)
        cube.CreateSizeAttr(1.0)
        cube.AddTranslateOp().Set(Gf.Vec3d(*center))
        cube.AddScaleOp().Set(Gf.Vec3f(*size))
        cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])

    box("/World/Table", (0, 0, -0.03), (1.2, 1.2, 0.05), (0.30, 0.20, 0.12))
    # Approximate open box of small red packages; no branded/dataset assets.
    box("/World/Box/Base", (0.28, 0, 0.025), (0.22, 0.18, 0.03), (0.55, 0.38, 0.20))
    for i, (center, size) in enumerate([
        ((0.28, -0.09, 0.07), (0.22, 0.01, 0.10)),
        ((0.28, 0.09, 0.07), (0.22, 0.01, 0.10)),
        ((0.17, 0, 0.07), (0.01, 0.18, 0.10)),
        ((0.39, 0, 0.07), (0.01, 0.18, 0.10)),
    ]):
        box(f"/World/Box/Side{i}", center, size, (0.55, 0.38, 0.20))
    for i in range(6):
        box(f"/World/Package{i}", (0.23 + 0.05*(i % 3), -0.035 + 0.07*(i // 3), 0.055),
            (0.035, 0.05, 0.02), (0.65, 0.025, 0.02))
    camera = rep.functional.create.camera(position=(0.85, 0.8, 0.7), look_at=(0.10, 0, 0.18),
                                          parent="/World", name="Camera")
    product = rep.create.render_product(camera.GetPath(), (640, 480))
    rgb = rep.annotators.get("rgb")
    rgb.attach(product)
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(20):
        simulation.update()
    roots = [str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    (args.output / "articulation-inspection.json").write_text(json.dumps({
        "roots": roots, "usd_path": str(usd_path),
        "b601_prims": [{"path": str(p.GetPath()), "schemas": list(p.GetAppliedSchemas())}
                       for p in stage.Traverse() if str(p.GetPath()).startswith("/World/B601")],
    }, indent=2))
    stage.GetRootLayer().Export(str(args.output / "scene.usda"))
    if len(roots) != 1:
        raise ValueError(f"Expected one articulation root; found {roots}")
    robot = Articulation(roots[0])
    if set(robot.dof_names) != set(JOINT_NAMES):
        raise ValueError(f"Unexpected imported DOFs: {robot.dof_names}")
    indices = robot.get_dof_indices(JOINT_NAMES).numpy()
    robot.set_dof_gains(stiffnesses=[500], dampings=[40])
    robot.set_dof_max_efforts([100])
    import_seconds = time.perf_counter() - import_start
    cases = {}
    for name, allow in [("veto", False), ("allow", True)]:
        robot.set_dof_positions(NEUTRAL[None, :], dof_indices=indices)
        robot.set_dof_velocities([0])
        robot.set_dof_position_targets(NEUTRAL[None, :], dof_indices=indices)
        for _ in range(90):
            simulation.update()
        initial = robot.get_dof_positions().numpy()[0].copy()
        decision = ZeosActionGate(allow).authorize(command)
        dispatched = 0
        observed = []
        frames = []
        start = time.perf_counter()
        for target in targets:
            if decision.allowed:
                robot.set_dof_position_targets(target[None, :], dof_indices=indices)
                dispatched += 1
            for _ in range(12):
                simulation.update()
            observed.append(robot.get_dof_positions().numpy()[0].tolist())
            rep.orchestrator.step(delta_time=0.0, pause_timeline=False)
            pixels = rgb.get_data()
            if pixels.shape[:2] != (480, 640):
                raise RuntimeError(f"Invalid render dimensions: {pixels.shape}")
            frames.append(Image.fromarray(pixels[:, :, :3]).copy())
        measured = np.asarray(observed)
        if not np.isfinite(measured).all():
            raise RuntimeError("Non-finite articulation state")
        delta = float(np.linalg.norm(measured[-1] - initial))
        if decision.allowed and delta <= 0.01:
            raise RuntimeError("Authorized targets did not produce measurable motion")
        if not decision.allowed and dispatched != 0:
            raise RuntimeError("Denied case dispatched a target")
        frames[0].save(args.output / f"{name}.gif", save_all=True, append_images=frames[1:],
                       duration=200, loop=0)
        frames[-1].save(args.output / f"{name}-final.png")
        cases[name] = {"gate": asdict(decision), "dispatched_targets": dispatched,
                       "initial_dof_positions": initial.tolist(), "observed_positions": observed,
                       "final_delta_norm_radians_and_meters": delta,
                       "max_abs_drift_or_motion": float(np.abs(measured-initial).max()),
                       "replay_and_render_seconds": time.perf_counter() - start}
    timeline.stop()
    rgb.detach()
    product.destroy()
    return {"scope": "Isaac articulation replay of previous Gemma/head outputs with fresh real ZEOS gates",
            "live_vlm_in_isaac": False, "b601_imported": True,
            "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "asset_source": json.loads((args.assets / "SOURCE.json").read_text()),
            "mapping": "Illustrative bounded deltas; arm 0.05 rad/unit, fingers 1% travel/unit",
            "gravity": 0.0, "package_success_measured": False,
            "physics_engine": SimulationManager.get_active_physics_engine(),
            "articulation_root": roots[0],
            "dof_names": robot.dof_names, "import_seconds": import_seconds, "cases": cases}


if __name__ == "__main__":
    main()
