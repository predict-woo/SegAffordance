#!/usr/bin/env python3
"""Generalisation figure: three real hand-video records (one per 2D source) feed the model, and a
3D articulation prediction on an unseen category (the ARCTIC laptop of Fig. 5) comes out.

Pure SVG in the method-figure style (viz/20260914_arthur_method_figure). Inputs in src/ are
downscaled copies of the Fig. 2 picks (viz/20260913_fig2_data_samples/picks_nocap) and of the
Fig. 5 EgoArt panel (viz/20260914_fig5v1_handvideo/00_arctic_rot_270_dense.png).

    python3 build.py && rsvg-convert -w 2000 -o preview.png fig.svg && rsvg-convert -f pdf -o fig.pdf fig.svg
"""
import base64
from pathlib import Path

OUT = Path(__file__).resolve().parent
INK, LINE, MUTED, DIM = "#2b2b2b", "#555555", "#6b6b6b", "#8a8a8a"
TRUNK, GROUP, GROUPLINE = "#e9e9e9", "#f7f7f7", "#c9c9c9"
FONT = "Helvetica, Arial, sans-serif"
parts = []
add = parts.append


def b64(p):
    return base64.b64encode((OUT / "src" / p).read_bytes()).decode()


def text(x, y, s, fs=18, anchor="middle", fill=INK, weight="normal", italic=False):
    st = ' font-style="italic"' if italic else ""
    add(f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{fs}" fill="{fill}" text-anchor="{anchor}" '
        f'dominant-baseline="central" font-weight="{weight}"{st}>{s}</text>')


def arrow(pts, width=2.0, color=LINE):
    d = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    add(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" marker-end="url(#arr)" stroke-linejoin="round"/>')


def photo(x, y, w, h, img, cid, align="xMidYMid"):
    add(f'<clipPath id="{cid}"><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10"/></clipPath>')
    add(f'<image x="{x}" y="{y}" width="{w}" height="{h}" preserveAspectRatio="{align} slice" '
        f'clip-path="url(#{cid})" href="data:image/jpeg;base64,{b64(img)}"/>')
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="none" stroke="{LINE}" stroke-width="1.3"/>')


# ---------------------------------------------------------------- layout
import json
PW, PH, PG = 300, 200, 46                         # 2D panels, all cropped to 3:2, and their gap
rows = [   # (image, source, crop alignment)
    ("hoi4d.jpg", "HOI4D", "xMidYMid"),
    ("epic.jpg", "EPIC-KITCHENS", "xMidYMid"),
    ("arctic.jpg", "ARCTIC", "xMidYMin"),         # keep the top: the track's ring sits in the top-right corner
]
LX, LY0 = 40, 40
STACK = 3 * PH + 2 * PG
MID = LY0 + STACK / 2
MX, MW, MH = 470, 230, 188                        # model block, centred on the middle panel
MY = MID - MH / 2
SX, SW, SH = MX + MW / 2 - PW / 2, PW, PH         # SF3D panel, centred under the block
SY = LY0 + STACK - SH                             # bottom edge level with the 2D stack
OX, OW = 800, 460                                 # output panel, native 4:3
OH = OW * 3 / 4
OY = MID - OH / 2
W = OX + OW + 40
H = int(max(LY0 + STACK + 118, SY + SH + 118, OY + OH + 100))

add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
add('<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
    f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{LINE}"/></marker>'
    '<marker id="arrg" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="9" markerHeight="9" markerUnits="userSpaceOnUse" orient="auto">'
    '<path d="M 0 0 L 10 5 L 0 10 z" fill="#2ecc40" stroke="white" stroke-width="1.5"/></marker></defs>')
add(f'<rect width="{W}" height="{H}" fill="white"/>')

# ---- left: three 2D records, one per source, same size
for k, (img, src, al) in enumerate(rows):
    y = LY0 + k * (PH + PG)
    photo(LX, y, PW, PH, img, f"c{k}", al)
    text(LX + PW / 2, y + PH + 20, src, 20, fill=MUTED)
    arrow([(LX + PW, y + PH / 2), (MX, MID + (k - 1) * 70)])
text(LX + PW / 2, LY0 + STACK + 58, "2D human video", 30, weight="bold", fill="#444")
text(LX + PW / 2, LY0 + STACK + 92, "hand tracks, no 3D labels", 22, fill=MUTED)
text(MX - 18, MID - 20, "train", 22, anchor="end", fill=MUTED)

# ---- bottom: the 3D source (SceneFun3D record with its ground-truth articulation)
META = json.loads((OUT / "src" / "sf3d_meta.json").read_text())
photo(SX, SY, SW, SH, "sf3d_frame.jpg", "csf3d")            # square frame sliced to 3:2 (centre band)
add(f'<clipPath id="csf3d_ov"><rect x="{SX}" y="{SY}" width="{SW}" height="{SH}" rx="10"/></clipPath>')
add(f'<g clip-path="url(#csf3d_ov)">')
OY0 = SY - SW / 6                                             # the square frame's top, given the 3:2 centre crop
add(f'<image x="{SX}" y="{OY0:.1f}" width="{SW}" height="{SW}" href="data:image/png;base64,{b64("sf3d_mask.png")}"/>')


def P(uv):
    return (SX + uv[0] * SW, OY0 + uv[1] * SW)


ax = " ".join(f"{x:.1f},{y:.1f}" for x, y in (P(uv) for uv in META["axis_uv"]))
add(f'<polyline points="{ax}" fill="none" stroke="#333" stroke-opacity="0.6" stroke-width="7" stroke-linecap="round"/>')
add(f'<polyline points="{ax}" fill="none" stroke="#ffd43b" stroke-width="4" stroke-linecap="round"/>')
hx, hy = P(META["origin_uv"])
add(f'<circle cx="{hx:.1f}" cy="{hy:.1f}" r="7" fill="#ffd43b" stroke="#333" stroke-width="1.6"/>')
tr = [P(uv) for uv, ok in zip(META["traj_uv"], META["traj_valid"]) if ok]
d = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in tr)
add(f'<path d="{d}" fill="none" stroke="white" stroke-width="8" stroke-linecap="round" stroke-linejoin="round"/>')
add(f'<path d="{d}" fill="none" stroke="#2ecc40" stroke-width="4.5" stroke-linecap="round" stroke-linejoin="round" marker-end="url(#arrg)"/>')
px, py = P(META["point_uv"])
add(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="8" fill="white" stroke="#2ecc40" stroke-width="4"/>')
add('</g>')
add(f'<rect x="{SX}" y="{SY}" width="{SW}" height="{SH}" rx="10" fill="none" stroke="{LINE}" stroke-width="1.3"/>')
text(SX + SW / 2, SY + SH + 20, "SceneFun3D", 20, fill=MUTED)
text(SX + SW / 2, SY + SH + 58, "3D scans", 30, weight="bold", fill="#444")
text(SX + SW / 2, SY + SH + 92, "articulation labels, no hands", 22, fill=MUTED)
arrow([(SX + SW / 2, SY), (MX + MW / 2, MY + MH)])
text(MX + MW / 2 + 14, (SY + MY + MH) / 2, "train", 22, anchor="start", fill=MUTED)

# ---- middle: the model
add(f'<rect x="{MX}" y="{MY}" width="{MW}" height="{MH}" rx="18" fill="{TRUNK}" stroke="{LINE}" stroke-width="1.6"/>')
text(MX + MW / 2, MID, "EgoArt", 46, weight="bold")
arrow([(MX + MW, MID), (OX, MID)], width=2.4)
text((MX + MW + OX) / 2, MID - 22, "test", 22, fill=MUTED)

# ---- right: 3D prediction on an unseen category
photo(OX, OY, OW, OH, "pred_laptop.jpg", "cout")
text(OX + OW / 2, OY + OH + 38, "3D articulation", 30, weight="bold", fill="#444")
text(OX + OW / 2, OY + OH + 72, "on an unseen object category", 22, fill=MUTED)

add("</svg>")
(OUT / "fig.svg").write_text("\n".join(parts))
print("wrote", OUT / "fig.svg")
