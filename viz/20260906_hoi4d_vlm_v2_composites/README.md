# HOI4D VLM sweep v2 — what the VLM receives

Sample Set-of-Mark composite(s) exactly as sent to gpt-5.6-luna (one
1920x1080 JPEG per interaction window + the text prompt in
`tools/hoi4d_vlm_select_all.py:PROMPT`), from the full-package sweep
(`/workspace/vlm_select_v2/` on the HOI4D volume; composites live in
`shard_*/comps/`).

Layout (`build_composite_v2`): top row = full first/last frame of the
window with the zoom rectangle drawn; bottom row = zoomed crop (around the
candidate segments only — the hand mask always reaches the frame border and
would defeat the crop) with every 2Dseg class tinted + outlined, numbered
at a distance-transform interior point; colliding labels are pushed apart
with a leader line; the primary hand (palette index 2) is labeled H.
Classes under 2000 px are not labeled. The title carries the HOI4D category
name and the CSV verb.

Prompt asks for `ANSWER: <number(s) or NONE>` (a single moving part for
open/close/press/switch; ALL parts of the object for pick-up/put-down/
pour/cut — the builder unions them) and `DESC: <imperative sentence>`
(OPD-style: part + qualifier, no motion mechanics/direction).

`ZY20210800001_H1_C14_N25_S209_s03_T2_w0.jpg`: trash can, `open`; 1 = bin
body, 2 = lid. Expected reply: `ANSWER: 2`.
