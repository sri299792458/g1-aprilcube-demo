"""Build the sparse three-leg presenter for the printed 40 mm AprilCube.

All design dimensions are millimetres. The exported 3MF therefore opens at
the intended scale in Bambu Studio without a unit conversion.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import zipfile
from pathlib import Path

import manifold3d as manifold
import numpy as np
import trimesh
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = PROJECT_ROOT / "config/fixtures/cube_tripod_presenter.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "hardware/fixtures/cube_tripod_presenter"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cylinder_between(
    start: np.ndarray,
    end: np.ndarray,
    radius: float,
    segments: int = 64,
) -> manifold.Manifold:
    direction = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
    length = float(np.linalg.norm(direction))
    if length <= 0.0:
        raise ValueError("Cylinder endpoints must be distinct")
    transform = trimesh.geometry.align_vectors(
        np.array([0.0, 0.0, 1.0]), direction
    )
    transform[:3, 3] = start
    return manifold.Manifold.cylinder(
        length,
        radius,
        circular_segments=segments,
    ).transform(transform[:3, :4])


def canonical_indices(mesh: manifold.Mesh) -> tuple[np.ndarray, np.ndarray]:
    positions = np.asarray(mesh.vert_properties, dtype=np.float64)[:, :3]
    triangles = np.asarray(mesh.tri_verts, dtype=np.int64)
    parent = np.arange(len(positions), dtype=np.int64)
    for source, target in zip(mesh.merge_from_vert, mesh.merge_to_vert):
        parent[int(source)] = int(target)

    def root(index: int) -> int:
        trail: list[int] = []
        while parent[index] != index:
            trail.append(index)
            index = int(parent[index])
        for item in trail:
            parent[item] = index
        return index

    canonical = np.array([root(index) for index in range(len(positions))])
    used = sorted({int(canonical[int(index)]) for index in triangles.flat})
    compact = {old: new for new, old in enumerate(used)}
    faces = np.asarray(
        [
            [compact[int(canonical[int(index)])] for index in face]
            for face in triangles
        ],
        dtype=np.int64,
    )
    return positions[used], faces


def validate_spec(spec: dict) -> None:
    geometry = spec["geometry"]
    base = geometry["base"]
    legs = geometry["legs"]
    pads = geometry["contact_pads"]
    target = spec["target"]

    if int(legs["count"]) != 3:
        raise ValueError("This fixture's kinematic support contract requires three legs")
    if float(geometry["support_height_mm"]) <= float(base["height_mm"]):
        raise ValueError("Support height must be above the base")
    if float(legs["support_radius_mm"]) >= float(target["size_mm"]) / 2.0:
        raise ValueError("Support centres must remain beneath the target")
    planar_half_width = float(target["size_mm"]) / 2.0 - float(
        target["edge_radius_mm"]
    )
    if float(legs["support_radius_mm"]) + float(pads["diameter_mm"]) / 2.0 > planar_half_width:
        raise ValueError("Contact pads extend onto the cube's rounded edge blend")
    if float(legs["foot_radius_mm"]) + float(legs["diameter_mm"]) / 2.0 > float(
        base["diameter_mm"]
    ) / 2.0:
        raise ValueError("Leg feet extend beyond the base")


def build_mesh(spec: dict) -> tuple[trimesh.Trimesh, dict]:
    validate_spec(spec)
    geometry = spec["geometry"]
    base = geometry["base"]
    legs = geometry["legs"]
    pads = geometry["contact_pads"]

    support_height = float(geometry["support_height_mm"])
    base_radius = float(base["diameter_mm"]) / 2.0
    base_height = float(base["height_mm"])
    leg_radius = float(legs["diameter_mm"]) / 2.0
    foot_radius = float(legs["foot_radius_mm"])
    support_radius = float(legs["support_radius_mm"])
    pad_radius = float(pads["diameter_mm"]) / 2.0
    pad_height = float(pads["height_mm"])
    first_angle = float(legs["first_angle_deg"])

    solids = [
        manifold.Manifold.cylinder(
            base_height,
            base_radius,
            circular_segments=128,
        )
    ]
    contact_centres = []
    # The leg endpoints deliberately overlap the base and pads so the Boolean
    # result is one physical solid rather than four merely touching shells.
    leg_start_z = base_height / 2.0
    # End the slanted leg axis at the pad's lower plane. Its angled circular
    # cap then overlaps the pad but cannot protrude above the contact plane.
    leg_end_z = support_height - pad_height
    for index in range(3):
        angle_deg = first_angle + index * 120.0
        angle = math.radians(angle_deg)
        radial = np.array([math.cos(angle), math.sin(angle), 0.0])
        foot = radial * foot_radius
        foot[2] = leg_start_z
        contact = radial * support_radius
        contact[2] = support_height
        leg_end = contact.copy()
        leg_end[2] = leg_end_z
        solids.append(cylinder_between(foot, leg_end, leg_radius))

        pad = manifold.Manifold.cylinder(
            pad_height,
            pad_radius,
            circular_segments=64,
        ).translate(
            (
                float(contact[0]),
                float(contact[1]),
                support_height - pad_height,
            )
        )
        solids.append(pad)
        contact_centres.append(contact.tolist())

    solid = manifold.Manifold.batch_boolean(solids, manifold.OpType.Add)
    if solid.is_empty() or str(solid.status()) != "Error.NoError":
        raise RuntimeError(f"Tripod Boolean construction failed: {solid.status()}")

    vertices, faces = canonical_indices(solid.to_mesh())
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.remove_unreferenced_vertices()
    mesh.metadata["name"] = spec["name"]
    mesh.units = "mm"

    components = mesh.split(only_watertight=False)
    expected_extents = np.array([2.0 * base_radius, 2.0 * base_radius, support_height])
    expected_min = np.array([-base_radius, -base_radius, 0.0])
    expected_max = np.array([base_radius, base_radius, support_height])
    if not mesh.is_watertight:
        raise RuntimeError("Tripod mesh is not watertight")
    if not mesh.is_winding_consistent or mesh.volume <= 0.0:
        raise RuntimeError("Tripod mesh winding or signed volume is invalid")
    if len(components) != 1:
        raise RuntimeError(f"Tripod mesh has {len(components)} disconnected components")
    if not np.allclose(mesh.extents, expected_extents, atol=1e-6):
        raise RuntimeError(f"Unexpected extents: {mesh.extents}")
    if not np.allclose(mesh.bounds[0], expected_min, atol=1e-6):
        raise RuntimeError(f"Unexpected minimum bounds: {mesh.bounds[0]}")
    if not np.allclose(mesh.bounds[1], expected_max, atol=1e-6):
        raise RuntimeError(f"Unexpected maximum bounds: {mesh.bounds[1]}")

    vertical_span = leg_end_z - leg_start_z
    lean_deg = math.degrees(math.atan2(foot_radius - support_radius, vertical_span))
    triangle_inradius = support_radius / 2.0
    audit = {
        "contact_centres_mm": contact_centres,
        "contact_plane_z_mm": support_height,
        "contact_triangle_inradius_mm": triangle_inradius,
        "leg_lean_from_vertical_deg": lean_deg,
        "bounds_mm": mesh.bounds.tolist(),
        "extents_mm": mesh.extents.tolist(),
        "vertices": int(len(mesh.vertices)),
        "triangles": int(len(mesh.faces)),
        "connected_components": len(components),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "volume_mm3": float(mesh.volume),
        "full_solid_pla_mass_g": float(mesh.volume / 1000.0 * 1.24),
    }
    return mesh, audit


def verify_export(path: Path, expected_extents: np.ndarray) -> dict:
    # STL stores triangle soup, so normal loading must weld its coincident
    # vertices before the topology check. The 3MF remains indexed already.
    loaded = trimesh.load(path, force="mesh", process=True)
    if not isinstance(loaded, trimesh.Trimesh):
        raise RuntimeError(f"Could not reload {path.name} as one mesh")
    if not loaded.is_watertight or not loaded.is_winding_consistent:
        raise RuntimeError(f"Reloaded {path.name} is not a valid closed mesh")
    if not np.allclose(loaded.extents, expected_extents, atol=1e-5):
        raise RuntimeError(f"Reloaded {path.name} has wrong scale: {loaded.extents}")
    return {
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "watertight": bool(loaded.is_watertight),
        "connected_components": len(loaded.split(only_watertight=False)),
        "extents_mm": loaded.extents.tolist(),
    }


def verify_3mf_archive(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        corrupt_member = archive.testzip()
        if corrupt_member is not None:
            raise RuntimeError(f"Corrupt 3MF member in {path.name}: {corrupt_member}")
        model = archive.read("3D/3dmodel.model")
        if b'unit="millimeter"' not in model:
            raise RuntimeError(f"{path.name} does not declare millimetre units")


def height_token(height_mm: float) -> str:
    if float(height_mm).is_integer():
        return str(int(height_mm))
    return str(height_mm).replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    spec_path = args.spec.resolve()
    output = args.output.resolve()
    spec = yaml.safe_load(spec_path.read_text())
    mesh, audit = build_mesh(spec)

    output.mkdir(parents=True, exist_ok=True)
    threemf_path = output / "cube_tripod_presenter.3mf"
    stl_path = output / "cube_tripod_presenter.stl"
    audit_path = output / "audit.json"
    threemf_path.write_bytes(mesh.export(file_type="3mf"))
    stl_path.write_bytes(mesh.export(file_type="stl"))

    verify_3mf_archive(threemf_path)

    expected_extents = np.asarray(audit["extents_mm"])
    audit["schema_version"] = 1
    audit["design_spec"] = str(spec_path.relative_to(PROJECT_ROOT))
    audit["design_spec_sha256"] = sha256(spec_path)
    audit["exports"] = {
        "3mf": verify_export(threemf_path, expected_extents),
        "stl": verify_export(stl_path, expected_extents),
    }

    variant_heights = [
        float(value) for value in spec["manufacturing"]["height_test_variants_mm"]
    ]
    if sorted(set(variant_heights)) != variant_heights:
        raise ValueError("Height-test variants must be unique and sorted")
    spacing = float(spec["manufacturing"]["plate_centre_spacing_mm"])
    if spacing < float(spec["geometry"]["base"]["diameter_mm"]):
        raise ValueError("Plate centre spacing must exceed the base diameter")

    plate_meshes: list[trimesh.Trimesh] = []
    variant_audit = {}
    centre_offset = (len(variant_heights) - 1) / 2.0
    for index, height in enumerate(variant_heights):
        variant_spec = copy.deepcopy(spec)
        variant_spec["name"] = f"cube_tripod_presenter_h{height_token(height)}"
        variant_spec["geometry"]["support_height_mm"] = height
        variant_mesh, variant_geometry_audit = build_mesh(variant_spec)
        token = height_token(height)
        variant_3mf = output / f"cube_tripod_presenter_h{token}.3mf"
        variant_stl = output / f"cube_tripod_presenter_h{token}.stl"
        variant_3mf.write_bytes(variant_mesh.export(file_type="3mf"))
        variant_stl.write_bytes(variant_mesh.export(file_type="stl"))
        verify_3mf_archive(variant_3mf)
        variant_extents = np.asarray(variant_geometry_audit["extents_mm"])
        variant_audit[token] = {
            "support_height_mm": height,
            "geometry": variant_geometry_audit,
            "exports": {
                "3mf": verify_export(variant_3mf, variant_extents),
                "stl": verify_export(variant_stl, variant_extents),
            },
        }

        positioned = variant_mesh.copy()
        positioned.apply_translation([(index - centre_offset) * spacing, 0.0, 0.0])
        plate_meshes.append(positioned)

    plate = trimesh.util.concatenate(plate_meshes)
    plate_components = plate.split(only_watertight=False)
    if len(plate_components) != len(variant_heights):
        raise RuntimeError(
            f"Height-test plate has {len(plate_components)} components, expected "
            f"{len(variant_heights)}"
        )
    if not plate.is_watertight or not plate.is_winding_consistent:
        raise RuntimeError("Height-test plate contains invalid fixture geometry")
    plate_path = output / "cube_tripod_presenter_height_test_plate_40_50_60.3mf"
    plate_path.write_bytes(plate.export(file_type="3mf"))
    verify_3mf_archive(plate_path)
    expected_plate_extents = np.array(
        [
            spacing * (len(variant_heights) - 1)
            + float(spec["geometry"]["base"]["diameter_mm"]),
            float(spec["geometry"]["base"]["diameter_mm"]),
            max(variant_heights),
        ]
    )
    plate_export_audit = verify_export(plate_path, expected_plate_extents)
    if plate_export_audit["connected_components"] != len(variant_heights):
        raise RuntimeError("Reloaded height-test plate lost a fixture component")
    audit["height_test"] = {
        "variant_order_left_to_right_mm": variant_heights,
        "plate_centre_spacing_mm": spacing,
        "plate_extents_mm": expected_plate_extents.tolist(),
        "plate_3mf": plate_export_audit,
        "variants": variant_audit,
    }
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")

    print(f"3MF: {threemf_path}")
    print(f"STL: {stl_path}")
    print(f"audit: {audit_path}")
    print(f"height-test plate: {plate_path}")
    print(f"extents: {mesh.extents.tolist()} mm")
    print(f"watertight: {mesh.is_watertight}; components: {len(mesh.split())}")


if __name__ == "__main__":
    main()
