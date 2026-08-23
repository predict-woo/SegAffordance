"""GT sign-convention check: the dataset writer's trajectories obey the
right-hand rule w.r.t. the stored motion axis — so the midpoint
screw-direction term (PredPredArticulationLoss.dir_weight) is compatible
with the GT supervision and will never fight the GT axis loss.

Verified here against the REAL writer code: tools/sf3d_process.py's
compute_trajectory_3d_camera_coords is AST-extracted from source (the
module itself imports the pod-only SceneFun3D devkit, so it cannot be
imported directly) and executed with numpy only. If someone changes the
writer's sweep-sign convention, this test breaks loudly.

Code-path facts this locks in (checked 2026-08-23):
  * writer rot arcs: e2 = n x e1, points sweep cos(t) e1 + sin(t) e2 with
    t in [0, +pi/2]  ->  velocity == +n x r everywhere (positive
    right-hand rotation about the stored axis);
  * writer trans rays: origin + t * dir_unit, t >= 0  ->  along +axis;
  * tools/sf3d_build_v3.py trans rebuild: +linspace recompute or positive
    rescale about the start point — sign preserved; rot records untouched;
  * datasets/scenefun3d.py reader: linspace subsample (order-preserving),
    axis tensor passed through verbatim.
"""

import ast
import math
import pathlib

import numpy as np
import pytest
import torch

from model.losses.geometric import PredPredArticulationLoss
from model.outputs import ModelOutputs

_SRC = pathlib.Path(__file__).resolve().parents[1] / "tools" / "sf3d_process.py"


def _load_writer_fn():
    tree = ast.parse(_SRC.read_text())
    wanted = {
        "TRAJECTORY_NUM_POINTS", "TRAJECTORY_TRANS_LENGTH_M",
        "TRAJECTORY_ROT_ARC_RAD", "TRAJECTORY_MIN_ROT_RADIUS_M",
        "TRAJECTORY_DEGENERATE_SEGMENT_M",
        "compute_trajectory_3d_camera_coords",
    }
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in wanted for t in node.targets
        ):
            nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in wanted:
            nodes.append(node)
    ns = {"np": np}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(_SRC), "exec"), ns)
    return ns["compute_trajectory_3d_camera_coords"]


compute_trajectory = _load_writer_fn()


def score_dir_term(axis, origin, traj_abs, p_revolute):
    """Score a writer trajectory + its OWN stored axis with the loss term."""
    traj_abs = torch.as_tensor(np.asarray(traj_abs), dtype=torch.float32)
    rel = (traj_abs - traj_abs[0:1]).unsqueeze(0)
    b = 1
    logit = torch.full((b,), 20.0)
    sign = 1.0 if p_revolute == 1.0 else -1.0
    type_logits = torch.stack([-sign * logit, sign * logit], dim=1)
    dummy_map = torch.zeros(b, 1, 4, 4)
    out = ModelOutputs(
        mask_logits=dummy_map,
        point_logits=dummy_map,
        point_uv=torch.zeros(b, 2),
        motion_pred=torch.as_tensor(axis, dtype=torch.float32).reshape(1, 3),
        motion_type_logits=type_logits,
        trajectory_pred=rel,
        origin_pred=torch.as_tensor(origin, dtype=torch.float32).reshape(1, 3),
        point_3d_pred=traj_abs[0].reshape(1, 3),
    )
    loss = PredPredArticulationLoss(weight=0.0, dir_weight=1.0)
    _, terms = loss(out, None)
    return terms["L_geo_pp_dir"].item()


def _random_case(seed):
    g = np.random.default_rng(seed)
    axis = g.normal(size=3)
    axis /= np.linalg.norm(axis)
    origin = g.normal(size=3) * 0.5 + np.array([0.0, 0.0, 2.0])
    # visible element points: a small blob well off the axis
    center = origin + np.cross(axis, g.normal(size=3)) * 1.0
    pts = center[None, :] + g.normal(size=(40, 3)) * 0.03
    return axis, origin, pts


@pytest.mark.parametrize("seed", range(8))
def test_writer_rot_arcs_satisfy_right_hand_rule(seed):
    axis, origin, pts = _random_case(seed)
    traj = compute_trajectory("rot", origin, axis, pts)
    assert len(traj) >= 20
    assert score_dir_term(axis, origin, traj, 1.0) == pytest.approx(0.0, abs=1e-5)
    # and the FLIPPED axis maxes the term — the conventions are opposite
    assert score_dir_term(-axis, origin, traj, 1.0) == pytest.approx(2.0, abs=1e-4)


@pytest.mark.parametrize("seed", range(8))
def test_writer_trans_rays_run_along_positive_axis(seed):
    axis, origin, pts = _random_case(seed)
    traj = compute_trajectory("trans", origin, axis, pts)
    assert score_dir_term(axis, origin, traj, 0.0) == pytest.approx(0.0, abs=1e-5)
    assert score_dir_term(-axis, origin, traj, 0.0) == pytest.approx(2.0, abs=1e-4)


def test_writer_round_knob_fallback_branch_too():
    # Candidate-2 branch: element centroid ON the axis (round knob) — the
    # radius comes from the mean axis distance, arc phased at the farthest
    # point. The sweep sign convention must be the same there.
    g = np.random.default_rng(0)
    axis = np.array([0.0, 0.0, 1.0])
    origin = np.array([0.1, -0.2, 1.5])
    # points ring-symmetric about the axis -> centroid ~ on axis
    ang = g.uniform(0, 2 * math.pi, size=60)
    ring = origin[None, :] + 0.05 * np.stack(
        [np.cos(ang), np.sin(ang), np.zeros_like(ang)], axis=1
    )
    traj = compute_trajectory("rot", origin, axis, ring)
    assert score_dir_term(axis, origin, traj, 1.0) == pytest.approx(0.0, abs=1e-5)
