# 20260829_sf3d_cf_l2_noaxis — position L2 alone (IN FLIGHT)

**Question:** the mirror of cf_h1_noaxis — what does the position
quadratic do BY ITSELF? Completes the single-term corners of the
composition grid:

| arm | L2 | H1 | anchor | MA |
|---|---|---|---|---|
| closedform | 0.5 | 0.5 | 0.5 | 29.19 |
| cf_noaxis | 0.5 | 0.5 | — | 27.71 |
| cf_h1only | — | 1.0 | 0.5 | **30.64 record** |
| cf_h1_noaxis | — | 1.0 | — | in flight |
| **this arm** | 1.0 | — | — | ? |
| cf_l2only (+anchor) | 1.0 | — | 0.5 | unrun corner |

**Recipe:** cf_noaxis base with position 1.0 / derivative 0.0 (single-
term weight mirroring cf_h1only), vae 0, sweep pi/2, 30 epochs, seed 42.

**Predictions (registered before launch):** matched axis blurrier than
every H1 arm (position tolerates axis error when origin compensates);
origin competitive (~0.25 — it is the term's measured job); flips
moderate (position flip penalty 2.75 at pi/2 is the family's strongest,
but no scale-free anchor); MA 25-27, below cf_noaxis (sharpener gone).

Result: PENDING
