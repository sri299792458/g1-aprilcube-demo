# 40 mm AprilCube tripod presenter

This single-piece PLA fixture presents the released 40 mm AprilCube 50 mm
above the tabletop. Three independent angled legs terminate in three small,
coplanar pads; there is no platform beneath the cube. The open space between
the legs gives the Dex3 fingers substantially more clearance than a cube placed
directly on the table.

## Print file

Open `cube_tripod_presenter.3mf` in Bambu Studio and select the Bambu Lab H2D.
The file contains geometry in millimetres and is already oriented base-down.

For the first experiment, use
`cube_tripod_presenter_height_test_plate_40_50_60.3mf`. It places three
fixtures on one plate at 90 mm centre spacing. Their support heights are 40,
50, and 60 mm from left to right. The three individual 3MF files are also
included in case one version needs to be reprinted later.

Suggested first print:

- Generic PLA;
- 0.4 mm nozzle;
- 0.20 mm layer height;
- three wall loops;
- 15% gyroid infill; and
- supports disabled.

The three 5.5 mm legs are only about 13 degrees from vertical, so this model is
intended to print without supports. Fix the clean underside of the 72 mm base
to the tabletop using thin double-sided tape before robot operation.

## Offline right-hand grasp proposals

The exact tripod and tabletop meshes were checked against every pose in the
3,178-candidate retained GraspGenX pool. A candidate is included only if the
open hand, its complete local-Z approach, and the recorded Dex3 closing motion
clear the fixture and table. The sets assume the cube is centred and yaw-aligned
on the fixture; the measured real pose must still be rechecked by cuRobo.

| Height | Any 15/10/7 cm approach | 15 cm | 10 cm | 7 cm |
|---|---:|---:|---:|---:|
| 40 mm | 324 | 275 | 306 | 324 |
| 50 mm | 372 | 353 | 365 | 372 |
| 60 mm | 394 | 377 | 384 | 394 |

The deterministic shortlists are under
`artifacts/grasp_shortlists/cube_tripod_right_v1/` and rebuild with:

```bash
uv run python tools/build_fixture_grasp_shortlists.py
```

## Critical dimensions

| Feature | Dimension |
|---|---:|
| Cube-bottom height above table | 50 mm |
| Base | 72 mm diameter x 4 mm |
| Legs | 3 x 5.5 mm diameter |
| Contact pads | 3 x 4 mm diameter |
| Contact-pad centre radius | 14 mm |
| Static support-triangle inradius | 7 mm |

The exact source parameters are in
`config/fixtures/cube_tripod_presenter.yaml`. Rebuild and re-audit both print
formats with:

```bash
uv run python tools/build_cube_tripod_presenter.py
```

`audit.json` records scale, hashes, topology, component count, volume, and the
three pad locations.
