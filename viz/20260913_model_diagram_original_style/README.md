# Current architecture in the original diagram style

Editable SVG schematic for `20260913_joint4_decoder_l2anchor_dense` (seed 42), based on `config/joint4_decoder_l2anchor_dense.yaml` and `model/layers.py`. This is an architecture figure, not checkpoint inference.

Preserves the original Figma diagram's white background, gray network blocks, pink/red feature maps, mathematical labels, and thin connectors. The RGB example is reused from the original diagram screenshot. The architecture now shows frozen DINOv3/dino.txt, text-conditioned decoding, dense articulation voting, scalar depth/length heads, camera backprojection, and analytic motion decoding. The auxiliary training-only hinge heatmap is omitted; the deployed hinge origin comes from dense votes.

Regenerate locally (no model execution):

```sh
python3 viz/20260913_model_diagram_original_style/build.py
rsvg-convert -w 2100 -o viz/20260913_model_diagram_original_style/preview.png viz/20260913_model_diagram_original_style/model.svg
```

`model.svg` is the vector deliverable; `preview.png` is the rendered review copy. Figma import is pending because the MCP quota was exhausted and desktop input was unreliable. The earlier card-style proposal in Figma is not this revision.
