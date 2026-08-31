# HOI4D articulation annotator — design (2026-09-01)

**Goal:** a web-based tool to manually annotate articulation parameters
(prismatic/revolute axis, origin) on the HOI4D processed-2D dataset, by
moving an axis gizmo in a navigable 3D reconstruction of each sequence.
Built on viser (Apache-2.0, `pip install viser`) — no custom web stack.
Survey that selected viser over sstk/MultiScan's annotator and the
bbox-platforms (CVAT et al.): 2026-09-01 session; sstk's annotator is
the UX reference (axis gizmo + motion preview), not a code base.

**Decisions (user-approved):** annotate **per sequence in world frame**
(official poses project the one annotation into every sample);
scene = **fused multi-frame point cloud**; runs on the **dev pod**,
browsed from the Mac.

## Inputs (all on the main volume)

- `hoi4d_processed_2d/data.lmdb` — sample records (read-only here);
  keys group by sequence; records carry mask coords, wrist trajectory,
  window/event metadata, K.
- `hoi4d_processed_2d/frames.lmdb` — `jpeg` + `depth_png` (16-bit mm)
  + `orig_size` per key.
- `hands354/<seq>/camera/official_poses.npy` — (300,4,4) camera-to-world
  (verified present for all 354 seqs) + `intrinsic.npy` (3,3).

## Component: `tools/hoi4d_annotate_articulation.py`

Single file, three parts:

### 1. Scene builder (cached)
Group LMDB keys by sequence → choose ~15 sample frames spread across
windows → decode jpeg + depth, back-project with K (mm→m, drop zero
depth, respect `orig_size`→512 scaling), lift to world via
`official_poses[frame]` → subsample to ~300k points. Overlays:
current window's GT mask points tinted red (the moving part), wrist
trajectory polyline in world, small camera frusta. Cache per-sequence
npz under the annotations dir; revisits skip the build.

### 2. viser UI
- Transform-controls gizmo; **+z of the gizmo = articulation axis**;
  arrow + infinite dashed line rendered through it.
- Panel: sequence dropdown (annotated/unannotated badge), window
  selector, `prismatic | revolute` toggle, **align-to-trajectory**
  button (init axis from the wrist track's principal direction),
  **motion preview slider** (sweeps the red mask points along the
  current screw — translate along axis, or rotate about axis+origin;
  the immediate visual check for a wrong axis or sign), Save / Skip /
  Flag-bad.
- Revolute: gizmo position = origin. Prismatic: position display-only.
- Sign convention matches our GT: trans moves along +axis for the
  "open"-family events; rot is right-hand-positive opening. The
  preview makes sign errors visible.

### 3. Storage + export
Per-sequence JSON `annotations/<seq>.json` (atomic tmp+rename; LMDBs
never mutated):

```json
{"seq": "...", "category": "C4", "annotator": "andy", "time": "...",
 "parts": [{"window_events": ["open", "close"], "type": "trans|rot",
            "axis_world": [x, y, z], "origin_world": [x, y, z],
            "flag": null | "bad-poses" | "ambiguous"}]}
```

One part entry normally covers all windows of a sequence; the window
selector can add a second entry when two parts articulate.

`--export` mode: project world → camera per sample key via
`inv(pose[frame])`, emit SF3D-convention
`motion_dir_3d_camera_coords` / `motion_origin_3d_camera_coords` /
`motion_origin_2d_image_coords` + `motion_type`, as a sidecar pickle
keyed like `data.lmdb` (mergeable into records later; merge is out of
scope).

## Operation

Run on the dev pod (`viser` server, default port 8080); open from the
Mac through an SSH port-forward on the existing `segaff-dev` host
entry. Annotations land on the volume beside the LMDBs.

## Testing

- Unit (repo suite): back-projection round-trip against K;
  world↔camera export round-trip through a known pose; annotation
  JSON read/write + atomicity.
- Manual smoke: one C4 and one C6 sequence — preview sweep must match
  the video's actual motion.

## Out of scope

Multi-user, mask editing, motion-range annotation, LMDB record
merging, non-furniture categories (no hands/poses exist for them).
