# 20260913_iphone_l2anchor_vs_field — the two candidate models on the user's four phone photos

Same four iPhone photos as `20260911_iphone_probe` (inputs NOT committed; 26 mm-equivalent, 1152x1536,
f = 1109 px), prompts "open door" / "close laptop" / "open closet" / "push chair forward",
`tools/predict_image.py --field-names field --f35 26`. Panels: photo | `l2anchor`
(`20260912_joint4_decoder_l2anchor` best-epoch17) | `field` (`20260913_field_joint4_l2anchor` best-epoch14).

| photo | l2anchor | field |
|---|---|---|
| door | rot 0.97, mask + point on the handle, z_p 2.1 m; hinge line diagonal across the lower door (axis (0.41,-0.90,0.12)), r 0.48 — NOT on the hinge edge (the 2026-09-11 DCT joint model had it on the right edge) | rot 0.95, handle mask + point, z_p 1.7 m; axis (-0.22,-0.97,-0.07) = near-vertical, the right direction, but the line runs through the door's middle-right, r 0.47; orbit a horizontal arc through the handle |
| laptop | rot 0.80 (the older models said trans), mask on parts of the screen with spill, hinge line VERTICAL through the keyboard (wrong direction), r 1.19 | rot 0.84, mask spills badly (ceiling blobs, keyboard, the vacuum), hinge line diagonal from the vacuum to the keyboard, r 0.08 |
| closet | rot 0.93, mask + point on the right handle, hinge line diagonal across the right door (not at its edge), r 0.37 | rot 0.97, handle mask + a spurious blob on the door, hinge line the other diagonal, r 0.22 |
| chair | trans 0.18; point on the bed (wrong object), a short trajectory along the bed frame | trans 0.05; partial mask on the chair back, point on the blanket, short ray — the better localisation, still not a chair |

Reading. In the wild both are weak on hinge PLACEMENT: neither puts the door or closet hinge on the
door's edge, and both get the laptop hinge direction wrong (l2anchor vertical through the keyboard,
field diagonal). The field model's axis DIRECTION is right on the door (near-vertical) and its type
calls are the most confident, but its masks spill on the cluttered laptop scene — the predicted-mask
weighting without GT teacher forcing may make out-of-distribution masks messier. Both now call the
laptop revolute (the HOI4D / ARCTIC labels at work). Hand-video training has not yet produced hinge
placement that survives a new apartment; the SF3D-side gains do not show up here. These four photos
are anecdotes, not a benchmark — a small labelled in-the-wild set would be the honest test.
