# 20260912_a3vlm_3doi_data_check — training targets rendered back onto their images

**What this shows.** A visual sanity check of the two new baseline converters before their
single-shot training runs: the encoded targets decoded from the generated dataset files and drawn
on the exact images each model will see.

- `a3vlm_<i>_<key>.jpg` — the 448 px padded square A3VLM consumes. Green = the 12 edges of the 3D
  box decoded from the REC answer string; red = the joint axis decoded from the REG-Joint answer,
  with a blue dot at its first endpoint; the joint type is printed top-left and the language query
  (our element description) along the bottom. Both are parsed from the answer strings with the
  same regexes the exporter uses, so a correct panel proves encode -> serialise -> parse -> decode.
- `3doi_<i>_<img_name>.jpg` — the 1024x768 frame 3DOI consumes. Green = the mask polygon, blue =
  the keypoint/affordance prompt, red = the 2D axis line (their format is a full-image line cut by
  the image rectangle, which is why it runs edge to edge), with the kinematic label next to each.

**How to read them.** The 3DOI panels should show tight polygons on the functional elements with
the keypoint inside the mask and each red line passing through its element. The A3VLM boxes are
small: at 448 px a SceneFun3D functional element is only a few pixels across, which is the same
scale problem the OPD baselines hit at 256x192 and is worth remembering when reading their scores.

**Regenerate.** The panels come from the staged datasets on the volume
(`/workspace/datasets/baselines/stage/{a3vlm,3doi}`), rendered with a throwaway script that
decodes with `tools/baselines_sf3d/sf3d_to_a3vlm.py` (`parse_box`, `parse_axis`, `unproject_uvd`)
and reads `3doi_sf3d/data_test.pt` directly. Note the A3VLM panels here were rendered from the
FIRST generation of the dataset (short, element-sized axis segments); the data was regenerated
afterwards with longer segments, so the red lines are longer in the data actually trained on.

**Numbers that came out of this check** (see `experiments/baselines_sf3d/20260913_a3vlm/notes.md`):
the A3VLM answer format quantises (u,v,d) to 2 decimals, and with the short segments a perfect
model would have scored only 89.5% within the 10 deg MA threshold; with the long segments the
encoding round-trips to 1.19 deg mean error over all 5,088 test elements.

Related: `experiments/baselines_sf3d/20260913_a3vlm`, `experiments/baselines_sf3d/20260913_3doi`.
