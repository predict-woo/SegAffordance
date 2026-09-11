# 20260912_hand2d_articulation_100 — the four joint decoder models on ~100 random held-out hand-video records

94 random held-out records of the three 2D hand sources (scene split = the joint4 datamodule's: ratio 0.15,
seed 42, same key caches), `tools/hoi4d_predict_articulation.py` (generalised to `--dataset hoi4d|epic|arctic`,
no category filter, up to 3 picks per sequence, seed 0), 2x sharp style, 0.5 m rays:

- `hoi4d/` 34 windows (rot + trans, "none" records excluded as in training)
- `epic/` 27 records (the held-out pool ran out at 27 under the 3-per-sequence cap)
- `arctic/` 33 strokes — the only source with a REAL hinge: GT axis from the object model drawn green on the GT panel
- `sheets/` 5-row contact sheets per source (jpg, for reading through)
- `arctic_axis_probe.csv` — `tools/arctic_axis_probe.py` over the whole ARCTIC held-out split (329 revolute records)

Panels: GT (mask green, 2D knuckle track cyan, label; ARCTIC: green hinge line) | `L2_2pi` =
`20260911_joint4_decoder_rgb_scalefree` best-epoch16 | `L2_2pi_axis` = `20260912_joint4_decoder_l2anchor`
best-epoch17 | `H1_axis` = `20260912_joint4_decoder_h1anchor` best-epoch16 | `cf_frame` =
`20260912_joint4_decoder_cfframe` best-epoch17 (all through `config/sf3d_test_decoder_rgb_scalefree.yaml`,
predicted type at test). Per model panel: mask red, point ring, hinge line red through the predicted origin
(rot) or a direction ray from the point (trans), 90-deg orbit yellow, decoded trajectory light green.

## What the panels show

1. **Type is right everywhere.** HOI4D: safes / cabinet doors / laptops / trash lids rot, drawers / switches /
   toy cars trans. EPIC: fridge, cupboard and microwave doors rot, drawers trans. ARCTIC: rot 100 %.
2. **Vertical hinges (doors) mostly land.** Fridge / cabinet / safe / microwave doors get a near-vertical hinge
   line at or near the door's hinge edge in most panels (HOI4D safe 16 / 19, cabinet doors 02 / 08, EPIC fridge
   01 / 04 / 11 / 12 / 14). SF3D is full of vertical-hinged doors, and this transfers.
3. **Horizontal hinges (lids) get the direction, not the placement.** On ARCTIC laptops the predicted axis
   vector is close to GT for L2_2pi / L2_2pi_axis / H1_axis (cos > 0.95 in most panels) but the radius is
   1-5 m, so the hinge line is drawn parallel to the seam and offset from it. H1_axis collapses the radius to
   0.1-0.2 m and puts the line through the hand. Box lids similar. Notebooks: cf_frame nails them, the others
   flip or miss.
4. **Trans rays are sensible**: along the drawer pull direction on EPIC / HOI4D drawers, along the push on the
   toy car.
5. **Masks**: moving part on all three sources, with the known decoder-arm spill on EPIC (cupboard carcass,
   fridge front) and some hand inclusion on ARCTIC.
6. z_p is unsupervised on hand video: 0.1-4 m for the same object across models.

## ARCTIC held-out split, 329 revolute records, predicted axis vs GT hinge (`arctic_axis_probe.csv`)

| model | axis err mean / median (deg) | < 10 deg | < 20 deg | sign flips | hinge-line offset (median, image frac) | radius median (m) |
|---|---|---|---|---|---|---|
| L2_2pi | 41.1 / 38.2 | 10.3 % | 24.9 % | 32.5 % | 0.085 | 1.18 |
| L2_2pi_axis | 48.2 / 45.7 | 2.7 % | 18.8 % | 26.4 % | 0.070 | 0.69 |
| **H1_axis** | **18.0 / 13.2** | **36.2 %** | **69.6 %** | **12.8 %** | 0.164 | 0.08 |
| cf_frame | 46.0 / 46.4 | 7.0 % | 23.4 % | 48.3 % | 0.096 | 0.90 |

Per object, unsigned mean deg / flip % (n: box 11, capsule machine 14, espresso machine 40, laptop 31,
microwave 31, notebook 36, phone 68, scissors 80, waffle iron 18):

| model | box | capsule | espresso | laptop | microwave | notebook | phone | scissors | waffle |
|---|---|---|---|---|---|---|---|---|---|
| L2_2pi | 20.3 / 0 | 48.6 / 36 | 55.2 / 100 | 10.0 / 0 | 57.7 / 55 | 48.0 / 58 | 35.0 / 9 | 41.3 / 12 | 51.3 / 44 |
| L2_2pi_axis | 29.4 / 36 | 59.3 / 7 | 81.0 / 35 | 21.2 / 0 | 42.9 / 42 | 49.2 / 47 | 33.0 / 6 | 55.7 / 28 | 56.8 / 67 |
| H1_axis | 13.7 / 0 | 25.2 / 0 | 6.6 / 0 | 9.0 / 0 | 9.8 / 0 | 24.9 / 100 | 25.1 / 3 | 17.2 / 0 | 33.4 / 22 |
| cf_frame | 21.7 / 36 | 47.2 / 57 | 72.6 / 100 | 18.6 / 39 | 49.6 / 26 | 24.3 / 0 | 32.3 / 96 | 65.1 / 18 | 51.9 / 44 |

Reading. (a) **The H1 + axis model transfers hinge DIRECTION to hand video far better than the other three**
(18 deg mean, 70 % under 20 deg, 13 % flips), including objects SF3D never shows (espresso lever 6.6 deg,
microwave 9.8, laptop 9.0, scissors 17.2) — while it is the worst on SF3D's own MA (31.07 vs cf_frame 36.36)
and collapses the radius to ~8 cm (hence the worst hinge-line offset: the line goes through the hand). (b) The
SF3D-record model (cf_frame) is the worst transfer: 46 deg mean and 48 % sign flips — the scale-free axis
term that wins on SF3D does not carry its sign to the hand sources, whereas the H1 quadratic (velocity match
on a sweep, sign-aware by construction) does. The 2D projection loss alone does not fix sign (the L2-only
base flips 33 %). (c) Laptops are easy for direction in every arm (9-21 deg, no flips except cf_frame); their
visible failure in the laptop probe is the radius/origin, not the axis. (d) Notebook covers invert the
ordering (cf_frame 24 deg / 0 flips, H1 100 % flips): the two losses disagree on which way a book-cover hinge
points, i.e. the sign convention the 2D arc imposes is object-dependent. (e) Hinge PLACEMENT (origin) is bad
for all four (median offset 7-16 % of the image width): nothing on the 2D side supervises the origin, and the
radius is either huge (L2 / cf_frame) or collapsed (H1).

Consequence for the next step: ARCTIC's GT axes + origins under the closed-form loss (the parked candidate)
supervise exactly the two things that are missing here (sign on lids, origin), on the object classes the
hand sources care about. Which SF3D-side loss to keep is now a real question: cf_frame for SF3D MA, or
H1 + axis for transfer — worth an ARCTIC axis probe on every future joint arm (`tools/arctic_axis_probe.py`,
~4 min on the dev pod). Regen: manifest.yaml in each source dir.
