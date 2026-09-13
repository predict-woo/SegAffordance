# Fig. 2 candidates: ground-truth records of the three hand-video sources

Paper figure material (ARTHUR, Fig. 2 "dataset examples"). No model involved:
each panel is a ground-truth record exactly as the loader returns it, drawn on
the source frame at its native aspect ratio (1024 px long side, upscaled from
the 512 x 512 training frame).

Per panel: moving-part mask (red fill + outline), hand track of the interacting
hand's middle-finger knuckle (green, white halo, ring at the first point = the
interaction point, arrowhead at the last point), a type badge (revolute orange
/ prismatic blue), and the instruction as the caption strip.

Samples: 10 per source from the held-out (val) scene split (`val_split_ratio`
0.15, seed 42), one record per scene, type-balanced 5/5 for HOI4D and EPIC
(ARCTIC is all revolute), pick seed 0. `<source>/samples.md` lists key, type
and instruction per panel; `<source>/contact_<source>.jpg` is a 5 x 2 sheet
for browsing. The user picks the panels that go into the figure.

Regenerate (dev pod). matplotlib is not in `requirements.lock`, and
`dev.sh run` re-syncs `/opt/venv` from the lock on every call, so the
install has to be in the same session as the render:

```bash
bash runpod/dev.sh run "uv pip install -q --python /opt/venv/bin/python matplotlib && \
  B=viz/20260913_fig2_data_samples && for s in hoi4d epic arctic; do \
  python tools/viz_data_samples.py --dataset \$s --num 10 --out \$B/\$s; done"
```

Datasets: `/workspace/datasets/hoi4d_processed_2d_v2`, `epic_processed_2d`,
`arctic_processed_2d` with their key caches under `/workspace/cache/`.

Reading notes (2026-09-13): HOI4D masks are clean part masks with short tracks
(laptop screens, cabinet doors, drawers, lamp arms, toy cars). EPIC masks are
the propagated VISOR polygons: fridge and drawer records are tight, the two
cupboard records (P18_02_28, P06_05_125) show the whole-carcass VISOR
convention noted in STATE.md. ARCTIC masks are pixel-accurate renders with the
hand not cut out; the tracks are long arcs (laptop, box, waffle iron) or short
strokes (phone, scissors, levers). The waffle-iron and capsule-machine tracks
wander because the nearest hand releases the part mid-stroke.
