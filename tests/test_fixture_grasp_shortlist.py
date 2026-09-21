import fcl
import numpy as np
import trimesh
from trimesh.collision import mesh_to_BVH

from tools.build_fixture_grasp_shortlists import continuous_collision


def pose(z: float) -> np.ndarray:
    result = np.eye(4)
    result[2, 3] = z
    return result


def test_naive_mesh_ccd_detects_crossing_thin_support() -> None:
    moving = mesh_to_BVH(trimesh.creation.box([0.02, 0.02, 0.02]))
    support = trimesh.creation.box([0.04, 0.04, 0.004])
    obstacle = fcl.CollisionObject(mesh_to_BVH(support), fcl.Transform())

    assert continuous_collision(
        moving,
        pose(0.10),
        pose(-0.10),
        obstacle,
        motion_type=fcl.CCDMotionType.CCDM_TRANS,
        iterations=100,
        toc_tolerance=1.0e-6,
    )


def test_mesh_ccd_accepts_motion_that_stays_above_support() -> None:
    moving = mesh_to_BVH(trimesh.creation.box([0.02, 0.02, 0.02]))
    support = trimesh.creation.box([0.04, 0.04, 0.004])
    obstacle = fcl.CollisionObject(mesh_to_BVH(support), fcl.Transform())

    assert not continuous_collision(
        moving,
        pose(0.10),
        pose(0.05),
        obstacle,
        motion_type=fcl.CCDMotionType.CCDM_TRANS,
        iterations=100,
        toc_tolerance=1.0e-6,
    )
