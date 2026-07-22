"""Export clean watertight grasp meshes from the AprilCube voxel specs.

AprilCube's textured OBJ deliberately partitions marker faces for rendering.
GraspGenX only needs the physical outer surface, so this script repeats the
same rounded voxel-union operation without the color/material partition and
exports one watertight mesh in metres per part.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import manifold3d as manifold
import numpy as np
import trimesh
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC_ROOT = PROJECT_ROOT / "config/aprilcube_parts"
DEFAULT_OUTPUT = PROJECT_ROOT / "generated/aprilcube_parts"
DEFAULT_AUDIT = PROJECT_ROOT / "artifacts/aprilcube_parts/grasp_mesh_audit.json"
PARTS = {
    "t_body": "t_body.yaml",
    "u_legs": "u_legs.yaml",
    "cube_head": "cube_head.yaml",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def occupancy(shape: dict) -> set[tuple[int, int, int]]:
    result: set[tuple[int, int, int]] = set()
    for cuboid in shape["cuboids"]:
        origin = [int(value) for value in cuboid["origin"]]
        size = [int(value) for value in cuboid["size"]]
        for x in range(origin[0], origin[0] + size[0]):
            for y in range(origin[1], origin[1] + size[1]):
                for z in range(origin[2], origin[2] + size[2]):
                    result.add((x, y, z))
    if not result:
        raise ValueError("Voxel specification has no occupied cells")
    return result


def canonical_indices(mesh: manifold.Mesh) -> tuple[np.ndarray, np.ndarray]:
    positions = np.asarray(mesh.vert_properties, dtype=np.float64)[:, :3]
    triangles = np.asarray(mesh.tri_verts, dtype=np.int64)
    parent = np.arange(len(positions), dtype=np.int64)
    for source, target in zip(mesh.merge_from_vert, mesh.merge_to_vert):
        parent[int(source)] = int(target)

    def root(index: int) -> int:
        trail = []
        while parent[index] != index:
            trail.append(index)
            index = int(parent[index])
        for item in trail:
            parent[item] = index
        return index

    canonical = np.array([root(index) for index in range(len(positions))], dtype=np.int64)
    used = sorted({int(canonical[int(index)]) for index in triangles.flat})
    compact = {old: new for new, old in enumerate(used)}
    faces = np.asarray(
        [[compact[int(canonical[int(index)])] for index in face] for face in triangles],
        dtype=np.int64,
    )
    return positions[used], faces


def build_mesh(spec: dict) -> trimesh.Trimesh:
    shape = spec["shape"]
    voxel_size = float(shape["voxel_size_mm"])
    occupied = occupancy(shape)
    mins = np.min(np.asarray(list(occupied)), axis=0)
    maxs = np.max(np.asarray(list(occupied)), axis=0)
    center_abs = (mins + maxs + 1.0) * voxel_size / 2.0

    solids = []
    for voxel in sorted(occupied):
        origin = np.asarray(voxel, dtype=float) * voxel_size - center_abs
        solids.append(
            manifold.Manifold.cube((voxel_size, voxel_size, voxel_size)).translate(origin)
        )
    solid = manifold.Manifold.batch_boolean(solids, manifold.OpType.Add)
    radius = float(spec.get("geometry", {}).get("edge_radius_mm", 0.0))
    segments = int(spec.get("geometry", {}).get("edge_segments", 5))
    if radius > 0:
        sphere = manifold.Manifold.sphere(radius, max(8, 8 * segments))
        solid = solid.minkowski_difference(sphere).minkowski_sum(sphere)
    if solid.is_empty() or str(solid.status()) != "Error.NoError":
        raise RuntimeError(f"Manifold construction failed: {solid.status()}")
    vertices_mm, faces = canonical_indices(solid.to_mesh())
    # Manifold already returns a valid indexed solid.  Do not run trimesh's
    # generic vertex-merging process here: its tolerance can merge distinct
    # vertices across the small fillet and destroy topology after metre scaling.
    mesh = trimesh.Trimesh(vertices=vertices_mm / 1000.0, faces=faces, process=False)
    mesh.remove_unreferenced_vertices()
    if not mesh.is_watertight or not mesh.is_winding_consistent or mesh.volume <= 0:
        raise RuntimeError(
            "Physical grasp mesh is not a positive, consistently wound watertight solid"
        )
    return mesh


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    args = parser.parse_args()

    audit = {"schema_version": 1, "parts": {}}
    for name, spec_name in PARTS.items():
        spec_path = SPEC_ROOT / spec_name
        spec = yaml.safe_load(spec_path.read_text())
        mesh = build_mesh(spec)
        output = args.output_root / name / "grasp_mesh.obj"
        output.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(output)
        audit["parts"][name] = {
            "spec_sha256": sha256(spec_path),
            "mesh_sha256": sha256(output),
            "vertices": int(len(mesh.vertices)),
            "faces": int(len(mesh.faces)),
            "watertight": bool(mesh.is_watertight),
            "winding_consistent": bool(mesh.is_winding_consistent),
            "extents_m": mesh.extents.tolist(),
            "volume_m3": float(mesh.volume),
        }
        print(f"{name}: {output} extents={mesh.extents} m")

    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, indent=2) + "\n")
    print(f"audit: {args.audit}")


if __name__ == "__main__":
    main()
