#!/usr/bin/env python3
"""ARTHUR method figure (final dense-voting model, 20260913_joint4_decoder_l2anchor_dense).

Pure-SVG architecture schematic, no model execution. Style: flat pastel rounded blocks with
thin outlines (ViT / DiT / BLIP-2 conventions), left-to-right data flow, frozen towers marked
with a snowflake, feature maps drawn as small grids. Sized for a two-column \\textwidth figure:
at 7.16 in the 26 px labels print at ~7 pt and the 17 px annotations at ~4.6 pt.

    python3 build.py               -> model.svg
    rsvg-convert -w 2400 -o preview.png model.svg
    rsvg-convert -f pdf -o model.pdf model.svg

Mask overlay (sample/mask.png -> sample/mask_overlay.png; sRGB conversion is required or the
fill stays gray):
    magick sample/mask.png -colorspace sRGB -type TrueColor -fill "#ff2fa6" -opaque white \
        -transparent black -alpha set -channel A -evaluate multiply 0.9 +channel PNG32:_fill.png
    magick sample/mask.png -morphology EdgeOut Diamond:3 -colorspace sRGB -type TrueColor \
        -fill "#3a0620" -opaque white -transparent black PNG32:_edge.png
    magick _fill.png _edge.png -composite PNG32:sample/mask_overlay.png
"""
import base64
import math
from pathlib import Path

OUT = Path(__file__).resolve().parent

# ---------------------------------------------------------------- palette
INK = "#2b2b2b"
LINE = "#555555"
MUTED = "#6b6b6b"
DIM = "#8a8a8a"
FROZEN = "#d6e4f5"                  # frozen towers (blue tint)
TRUNK = "#e9e9e9"                   # trainable trunk blocks (gray)
HEAT, HEATC, HEATD = "#fbe3c7", "#f6cfa4", "#e5a463"    # heatmap head (orange)
VOTE, VOTEC, VOTED = "#d8ecd2", "#c3dfb9", "#93c283"    # voting head (green)
SCAL = "#e8dcf3"                    # scalar MLP heads (purple)
GEOM = "#fff3c4"                    # analytic decoder (yellow)
MAPC, MAPD = "#bccbe8", "#8ea6d8"   # generic feature map cells
GROUP, GROUPLINE = "#f7f7f7", "#c9c9c9"

FONT = "Helvetica, Arial, sans-serif"
MATH = "'Times New Roman', Times, serif"

# font sizes (px)
F_BOX, F_SUB, F_ANN, F_MATH, F_TITLE, F_SMALL = 28, 18, 20, 28, 28, 19

parts = []


def add(s):
    parts.append(s)


# ---------------------------------------------------------------- primitives
def box(x, y, w, h, label, fill, sub=None, rot=False, r=12, fs=F_BOX, sfs=F_SUB, stroke=LINE, subfill=MUTED, dashed=False):
    da = ' stroke-dasharray="7 5"' if dashed else ""
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="1.5"{da}/>')
    cx, cy = x + w / 2, y + h / 2
    lines = label.split("\n")
    subs = sub.split("\n") if sub else []
    lh, slh = fs * 1.15, sfs * 1.25
    gap = fs * 0.5 + sfs * 0.5 + 14 if subs else 0      # main -> sub: half heights + clear air
    total = (len(lines) - 1) * lh + gap + (len(subs) - 1) * slh if subs else (len(lines) - 1) * lh
    y0 = cy - total / 2
    tr = f' transform="rotate(-90 {cx} {cy})"' if rot else ""
    for i, ln in enumerate(lines):
        add(f'<text x="{cx}" y="{y0 + i * lh:.1f}" font-family="{FONT}" font-size="{fs}" fill="{INK}" '
            f'text-anchor="middle" dominant-baseline="central"{tr}>{ln}</text>')
    for j, ln in enumerate(subs):
        yy = y0 + (len(lines) - 1) * lh + gap + j * slh
        add(f'<text x="{cx}" y="{yy:.1f}" font-family="{FONT}" font-size="{sfs}" fill="{subfill}" '
            f'text-anchor="middle" dominant-baseline="central"{tr}>{ln}</text>')


def text(x, y, s, fs=F_ANN, anchor="middle", fill=INK, math=False, weight="normal", italic=False):
    fam = MATH if math else FONT
    st = ' font-style="italic"' if (italic or math) else ""
    add(f'<text x="{x}" y="{y}" font-family="{fam}" font-size="{fs}" fill="{fill}" text-anchor="{anchor}" '
        f'dominant-baseline="central" font-weight="{weight}"{st}>{s}</text>')


def v(s, sub=None):
    """bold upright vector symbol (paper: \\mathbf)"""
    out = f'<tspan font-weight="bold" font-style="normal">{s}</tspan>'
    if sub is not None:
        out += f'<tspan font-size="70%" baseline-shift="sub" font-style="italic">{sub}</tspan>'
    return out


def sc(s, sub=None):
    """italic scalar"""
    out = f'<tspan font-style="italic">{s}</tspan>'
    if sub is not None:
        out += f'<tspan font-size="70%" baseline-shift="sub" font-style="italic">{sub}</tspan>'
    return out


def arrow(pts, dashed=False, color=LINE, width=1.8, head=True):
    d = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    dash = ' stroke-dasharray="7 6"' if dashed else ""
    mk = ' marker-end="url(#arr)"' if head and color == LINE else (' marker-end="url(#arrg)"' if head else "")
    add(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}"{dash}{mk} stroke-linejoin="round"/>')


def grid_map(x, y, size, n=4, fill=MAPC, dark=MAPD, gap=2.5, seed=0):
    cell = (size - (n - 1) * gap) / n
    for i in range(n):
        for j in range(n):
            f = dark if (i * 7 + j * 3 + seed) % 5 == 0 else fill
            add(f'<rect x="{x + j * (cell + gap):.1f}" y="{y + i * (cell + gap):.1f}" width="{cell:.1f}" '
                f'height="{cell:.1f}" rx="2.5" fill="{f}"/>')


def card_map(x, y, size, n=4, fill=MAPC, dark=MAPD, bg="white", seed=0):
    """an opaque feature-map card: white separator edge, light body, grid cells on top —
    stacks of these occlude each other instead of blending"""
    add(f'<rect x="{x - 4}" y="{y - 4}" width="{size + 8}" height="{size + 8}" rx="6" fill="{bg}"/>')
    add(f'<rect x="{x - 1.5}" y="{y - 1.5}" width="{size + 3}" height="{size + 3}" rx="4" fill="{fill}" fill-opacity="0.55"/>')
    grid_map(x, y, size, n=n, fill=fill, dark=dark, seed=seed)


def cell_grid(x, y, size, n, color_fn, gap=2.5):
    """n x n cell grid; color_fn(i, j) -> fill colour (None = skip)."""
    cell = (size - (n - 1) * gap) / n
    for i in range(n):
        for j in range(n):
            f = color_fn(i, j)
            if f:
                add(f'<rect x="{x + j * (cell + gap):.1f}" y="{y + i * (cell + gap):.1f}" width="{cell:.1f}" '
                    f'height="{cell:.1f}" rx="2.5" fill="{f}"/>')
    return cell


def tokens(x, y, n, cell=16, gap=5, fill=MAPC):
    for i in range(n):
        add(f'<rect x="{x + i * (cell + gap)}" y="{y}" width="{cell}" height="{cell}" rx="3.5" fill="{fill}" stroke="{LINE}" stroke-width="0.9"/>')


def snowflake(cx, cy, r=10):
    c = "#3b6ea8"
    for k in range(3):
        a = math.radians(60 * k + 90)
        dx, dy = r * math.cos(a), r * math.sin(a)
        add(f'<line x1="{cx - dx:.1f}" y1="{cy - dy:.1f}" x2="{cx + dx:.1f}" y2="{cy + dy:.1f}" stroke="{c}" stroke-width="2.2" stroke-linecap="round"/>')
        for sgn in (1, -1):
            tx, ty = cx + sgn * dx * 0.55, cy + sgn * dy * 0.55
            for b in (a + math.radians(40), a - math.radians(40)):
                ex, ey = tx + sgn * 0.4 * r * math.cos(b), ty + sgn * 0.4 * r * math.sin(b)
                add(f'<line x1="{tx:.1f}" y1="{ty:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" stroke="{c}" stroke-width="1.8" stroke-linecap="round"/>')


def group(x, y, w, h, title):
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="18" fill="{GROUP}" stroke="{GROUPLINE}" '
        f'stroke-width="1.3" stroke-dasharray="9 7"/>')
    text(x + w / 2, y + 36, title, F_TITLE, weight="bold", fill="#444")


def otimes(cx, cy, r=12):
    add(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="white" stroke="{LINE}" stroke-width="1.5"/>')
    k = r * 0.58
    add(f'<line x1="{cx - k:.1f}" y1="{cy - k:.1f}" x2="{cx + k:.1f}" y2="{cy + k:.1f}" stroke="{LINE}" stroke-width="1.5"/>')
    add(f'<line x1="{cx - k:.1f}" y1="{cy + k:.1f}" x2="{cx + k:.1f}" y2="{cy - k:.1f}" stroke="{LINE}" stroke-width="1.5"/>')


def photo(x, y, s, clip_id, data=None):
    add(f'<clipPath id="{clip_id}"><rect x="{x}" y="{y}" width="{s}" height="{s}" rx="10"/></clipPath>')
    add(f'<image x="{x}" y="{y}" width="{s}" height="{s}" preserveAspectRatio="xMidYMid slice" '
        f'clip-path="url(#{clip_id})" href="data:image/png;base64,{data or IMG}"/>')


def frame(x, y, s, dashed=False):
    da = ' stroke-dasharray="7 5"' if dashed else ""
    add(f'<rect x="{x}" y="{y}" width="{s}" height="{s}" rx="10" fill="none" stroke="{LINE}" stroke-width="1.3"{da}/>')


import json
SAMPLE = OUT / "sample"                      # dumped by dump_sample.py on the dev pod (val 1773)
IMG = base64.b64encode((SAMPLE / "frame.png").read_bytes()).decode()
MASK = base64.b64encode((SAMPLE / "mask_overlay.png").read_bytes()).decode()
DEPTH = base64.b64encode((SAMPLE / "depth.png").read_bytes()).decode()
META = json.loads((SAMPLE / "meta.json").read_text())

# ================================================================ canvas
# --- layout constants: everything is placed with a running x cursor and named gaps
PAD = 36               # inner padding of a group box
GGAP = 22              # gap between neighbouring groups
MID = 372              # vertical centre of the main data row
GY0 = 30               # top of the group boxes
TROW = 628             # centre of the instruction / text row
VH = 310               # height of the tall trunk blocks
VY = MID - VH / 2
BH = 108               # height of the readout head blocks
R1, R2, R3 = MID - 182, MID, MID + 182

# ---- pre-compute x positions (left to right) --------------------------------
IX, IS = 26, 192                          # input photo
G1X = IX + IS + 26                        # encoder group
VX, VW = G1X + PAD, 150                   # DINOv3 tower / dino.txt tower
PX, PW = VX + VW + 72, 66                # pyramid adapter
LVX = PX + PW + 40                        # pyramid levels column (left edge of the widest map)
LVW = 66
FX = PX + PW + 118                        # text-gated FPN (exploded): per-level convs start here
JX = PX + PW + 62                         # depth-concat junction column (middle of the gap, clear of the FPN border)
CW, CH = 40, 78                           # per-level "conv 3x3" boxes (rotated, tall)
GX_ = FX + CW + 30                        # gate (x) on the /32 row
CBX, CBW = GX_ + 14 + 70, 20              # concat bar
AGX, AGW = CBX + CBW + 22, 40             # 1x1 aggregation conv (rotated)
CCX, CCW = AGX + AGW + 22, 40             # CoordConv (rotated)
CH2 = 150                                 # height of the two rotated fusion convs
FW = CCX + CCW - FX                       # width of the exploded FPN
FMX, FMS = FX + FW + 44, 78               # fused map F
TDX, TDW = FMX + FMS + 30, 96             # transformer decoder
G1W = TDX + TDW + PAD - G1X
G2X = G1X + G1W + GGAP                    # readout group
QX, QS = G2X + 62, 104                    # decoded map F_q
BX, BW = QX + QS + 48, 200                # head blocks
HX, HS, HG = BX + BW + 28, 56, 52         # grids (mask, point heatmap)
UX = HX + 2 * HS + HG                     # end of the point map
FLX, VS = BX + BW + 28, 96                # vote-field grid
FEND = FLX + VS
PMX, PMW = FEND + 26, 118                 # part-weighted mean
G2W = PMX + PMW + PAD - G2X
G3X = G2X + G2W + GGAP                    # decoder group
BUS = G3X + 42                            # vertical bus for the fan-in
ADX, ADW, ADH = BUS + 52, 160, 200        # analytic decoder
ADY = MID - ADH / 2
OX, OS = ADX + ADW + 44, 200              # output panel
OY = MID - OS / 2
G3W = OX + OS + PAD - G3X
W = G3X + G3W + 26
# supervision boxes
SY, SH, SW = 678, 88, 216
S1X = ADX + ADW / 2 - SW / 2 - 18
S2X = OX + OS / 2 - SW / 2 + 18
GH = SY + SH / 2 + 60 - GY0
H = GY0 + GH + 22

add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
add('<defs>'
    '<marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse">'
    f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{LINE}"/></marker>'
    '<marker id="arrg" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse">'
    f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{DIM}"/></marker>'
    '<marker id="arrw" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" markerUnits="userSpaceOnUse" orient="auto">'
    '<path d="M 0 0 L 10 5 L 0 10 z" fill="white"/></marker>'
    '<marker id="arry" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="8" markerHeight="8" markerUnits="userSpaceOnUse" orient="auto">'
    '<path d="M 0 0 L 10 5 L 0 10 z" fill="#ffd43b" stroke="#333" stroke-width="1"/></marker>'
    '</defs>')
add(f'<rect width="{W}" height="{H}" fill="white"/>')

# ---- groups
group(G1X, GY0, G1W, GH, "Language-conditioned feature encoder")
group(G2X, GY0, G2W, GH, "Articulation prediction heads")
group(G3X, GY0, G3W, GH, "Analytic trajectory generator")

# ================================================================ inputs
IY = MID - IS / 2
photo(IX, IY, IS, "clipin")
frame(IX, IY, IS)
text(IX + IS / 2, IY + IS + 26, f'RGB image {v("I")}', F_ANN + 1)

# optional depth input (RGB-D variant): panel above the RGB image, dashed
DS = 116
DPX, DPY = IX + (IS - DS) / 2, 58
photo(DPX, DPY, DS, "clipdepth", data=DEPTH)
frame(DPX, DPY, DS, dashed=True)
text(IX + IS / 2, DPY + DS + 22, f'depth {v("D")} (optional)', F_SMALL, fill=MUTED)

TX, TW, TH = IX, IS, 112
TY = TROW - TH / 2
add(f'<rect x="{TX}" y="{TY}" width="{TW}" height="{TH}" rx="14" fill="white" stroke="{LINE}" stroke-width="1.3"/>')
for k, ln in enumerate(("“Open the left", "door of the closet", "near the window”")):
    text(TX + TW / 2, TROW + (k - 1) * 27, ln, F_ANN - 1, italic=True)
text(TX + TW / 2, TY + TH + 26, f'instruction {v("T")}', F_ANN + 1)

# ================================================================ frozen towers
box(VX, VY, VW, VH, "DINOv3\nViT-L/16", FROZEN, sub="frozen")
snowflake(VX + VW - 22, VY + 22)
arrow([(IX + IS, MID), (VX, MID)])

DH = 136
DY = TROW - DH / 2
box(VX, DY, VW, DH, "dino.txt\ntext encoder", FROZEN, sub="frozen", fs=22)
snowflake(VX + VW - 22, DY + 22)
arrow([(TX + TW, TROW), (VX, TROW)])

# pyramid adapter (single final-layer map -> strides 8/16/32)
box(PX, VY, PW, VH, "Pyramid adapter", TRUNK, rot=True, fs=23)
arrow([(VX + VW, MID), (PX, MID)])

LV = [(PX + PW, MID - 96 - 33, 66, "/8"), (PX + PW, MID - 22, 44, "/16"), (PX + PW, MID + 96 - 12, 24, "/32")]   # rows at MID -96 / 0 / +96

# text-gated FPN, exploded: each level -> conv 3x3; the coarsest level is gated by the sentence
# state; coarser levels are brought to /16 (up x2 / pool x2) and concatenated with the finer one;
# a 1x1 aggregation conv and a coordinate-augmented conv give F.  (model/layers.py FPN)
ROWS = [(yy + s_ / 2) for (_x, yy, s_, _l) in LV]          # row centres: /8, /16, /32
SGX0, SGX1 = FX - 18, CCX + CCW + 18
SGY0, SGY1 = ROWS[0] - CH / 2 - 58, ROWS[2] + CH / 2 + 28
add(f'<rect x="{SGX0}" y="{SGY0}" width="{SGX1 - SGX0}" height="{SGY1 - SGY0}" rx="14" fill="none" '
    f'stroke="{GROUPLINE}" stroke-width="1.2" stroke-dasharray="8 6"/>')
text((SGX0 + SGX1) / 2, SGY0 + 26, "Text-gated FPN", F_SUB + 3, weight="bold", fill="#444")
for k, ((xx, yy, s_, lab), ry) in enumerate(zip(LV, ROWS)):
    arrow([(PX + PW, ry), (FX, ry)])
    text((PX + PW + JX) / 2 - 4, ry - 15, lab, F_SMALL - 2, fill=MUTED)
    box(FX, ry - CH / 2, CW, CH, "conv 3×3", TRUNK, fs=16, r=8, rot=True)
# /32 row: gate by the sentence state, then up x2 into the concat bar
otimes(GX_, ROWS[2])
arrow([(FX + CW, ROWS[2]), (GX_ - 12, ROWS[2])])
arrow([(GX_ + 12, ROWS[2]), (CBX, ROWS[2])])
text((GX_ + 12 + CBX) / 2, ROWS[2] - 17, "up ×2", F_SMALL - 2, fill=MUTED)
# /16 row: straight into the bar
arrow([(FX + CW, ROWS[1]), (CBX, ROWS[1])])
# /8 row: pool x2 into the bar
arrow([(FX + CW, ROWS[0]), (CBX, ROWS[0])])
text((FX + CW + CBX) / 2, ROWS[0] - 17, "pool ×2", F_SMALL - 2, fill=MUTED)
# concat bar spanning the three rows
add(f'<rect x="{CBX}" y="{ROWS[0] - CH / 2}" width="{CBW}" height="{ROWS[2] - ROWS[0] + CH}" rx="6" '
    f'fill="{TRUNK}" stroke="{LINE}" stroke-width="1.4"/>')
text(CBX + CBW / 2, ROWS[1], "concat", F_SMALL - 2, fill=INK, weight="normal")
parts[-1] = parts[-1].replace('<text ', f'<text transform="rotate(-90 {CBX + CBW / 2} {ROWS[1]})" ', 1)
# aggregation 1x1 conv, coordinate-augmented conv
arrow([(CBX + CBW, ROWS[1]), (AGX, ROWS[1])])
box(AGX, ROWS[1] - CH2 / 2, AGW, CH2, "conv 1×1", TRUNK, fs=16, r=8, rot=True)
arrow([(AGX + AGW, ROWS[1]), (CCX, ROWS[1])])
box(CCX, ROWS[1] - CH2 / 2, CCW, CH2, "CoordConv", TRUNK, fs=16, r=8, rot=True)
# optional depth encoder: small conv branch, features concatenated to the /8 and /16 levels before the convs
DEW, DEH = 156, 54
DEX, DEY = JX - DEW / 2, DPY + DS / 2 - DEH / 2
box(DEX, DEY, DEW, DEH, "Depth encoder", TRUNK, fs=21, dashed=True, stroke=DIM)
arrow([(DPX + DS, DPY + DS / 2), (DEX, DPY + DS / 2)], dashed=True, color=DIM)
# dashed trunk down the gap, ending on the /16 row; a concat node on the /8 and /16 level arrows
add(f'<path d="M {JX},{DEY + DEH} L {JX},{ROWS[1] - 9}" fill="none" stroke="{DIM}" stroke-width="1.8" stroke-dasharray="7 6"/>')
for ry in ROWS[:2]:
    add(f'<circle cx="{JX}" cy="{ry}" r="9" fill="white" stroke="{LINE}" stroke-width="1.4"/>')
    add(f'<line x1="{JX - 5}" y1="{ry}" x2="{JX + 5}" y2="{ry}" stroke="{LINE}" stroke-width="1.4"/>')
    add(f'<line x1="{JX}" y1="{ry - 5}" x2="{JX}" y2="{ry + 5}" stroke="{LINE}" stroke-width="1.4"/>')
text(JX + 14, (DEY + DEH + SGY0) / 2, "concat at /8, /16", F_SMALL - 2, anchor="start", fill=MUTED)
# sentence state s from dino.txt up into the gate
arrow([(VX + VW, TROW - 36), (GX_, TROW - 36), (GX_, ROWS[2] + 12)])
text(GX_ + 16, SGY1 + 24, v("s"), F_MATH, anchor="start", math=True)
text(GX_ + 42, SGY1 + 24, "sentence state", F_SMALL, anchor="start", fill=MUTED)

# fused map F
grid_map(FMX, MID - FMS / 2, FMS, n=5, seed=2)
arrow([(CCX + CCW, MID), (FMX, MID)])
text(FMX + FMS / 2, MID + FMS / 2 + 24, v("F"), F_MATH, math=True)

# transformer decoder
box(TDX, VY, TDW, VH, "Transformer decoder", TRUNK, sub="3 layers, cross-attention to words", rot=True, fs=23, sfs=15)
arrow([(FMX + FMS, MID), (TDX, MID)])
WY = TROW + 46
arrow([(VX + VW, WY), (TDX + TDW / 2, WY), (TDX + TDW / 2, VY + VH)])
TKX = TDX + TDW / 2 - 150
tokens(TKX, WY - 58, 5)
text(TKX - 16, WY - 50, v("w"), F_MATH, anchor="end", math=True)
text(TKX + 50, WY - 22, "word tokens", F_SMALL, fill=MUTED)

# decoded map F_q
QY = MID - QS / 2
grid_map(QX, QY, QS, n=6, seed=1)
arrow([(TDX + TDW, MID), (QX, MID)])
text(QX + QS / 2, QY + QS + 24, "decoded map", F_ANN)
text(QX + QS / 2, QY + QS + 50, f'{v("F", "q")}&#8201; 512 × 32 × 32', F_SMALL, fill=MUTED)

# ================================================================ readouts of F_q
# --- row 1: dynamic-kernel projector -> heatmaps -> u_p
box(BX, R1 - BH / 2, BW, BH, "Dynamic-kernel\nprojector", HEAT, sub="kernels from s", fs=24, sfs=15)
arrow([(QX + QS / 2, QY), (QX + QS / 2, R1), (BX, R1)])
# mask M: the same cell grid, with the part's cells (a bar at the handle's position) dark
MU, MV = META["point_uv"]
NM = 6
mj, mi = int(MU * NM), int(MV * NM)                    # cell of the interaction point
mask_cells = {(mi - 2, mj), (mi - 1, mj), (mi, mj), (mi + 1, mj)}
cell_grid(HX, R1 - HS / 2, HS, NM, lambda i, j: HEATD if (i, j) in mask_cells else HEATC)
# point heatmap: cells shaded by distance from the point (a blob of heat in the same grid)
PX2, PY2 = HX + HS + HG, R1 - HS / 2
HEAT_RAMP = ["#fbe3c7", "#f6cfa4", "#eeb277", "#e5934a", "#c9722a"]
def _heat(i, j):
    d = ((i + 0.5 - MV * NM) ** 2 + (j + 0.5 - MU * NM) ** 2) ** 0.5
    k = max(0, min(4, int(4 - d * 1.35)))
    return HEAT_RAMP[k]
cell_grid(PX2, PY2, HS, NM, _heat)
arrow([(BX + BW, R1), (HX, R1)])
text(HX + HS / 2, R1 + HS / 2 + 20, f'mask {sc("M")}', F_SMALL)
text(HX + HS + HG + HS / 2, R1 + HS / 2 + 20, "point heatmap", F_SMALL)
arrow([(UX, R1), (BUS, R1), (BUS, ADY + 34), (ADX, ADY + 34)])
text(UX + 70, R1 + 20, "soft-argmax", F_SMALL, fill=MUTED)
text(UX + 150, R1 - 22, v("u", "p"), F_MATH, math=True)

# --- row 2: dense voting head -> per-pixel fields -> part-weighted mean
box(BX, R2 - BH / 2, BW, BH, "Dense voting\nhead", VOTE, sub="conv, one vote per pixel", fs=24, sfs=15)
arrow([(QX + QS, R2), (BX, R2)])
# vote field: the same cell grid; the part's cells are dark and each carries a tiny arrow, its
# offset vote, pointing at the hinge cell (red)
VX0, VY0 = FLX, R2 - VS / 2
NV = 5
part = [(1, 3), (2, 3), (3, 3), (1, 4), (2, 4), (3, 4)]
hinge = (2, 0)
cellv = cell_grid(VX0, VY0, VS, NV, lambda i, j: VOTED if (i, j) in part else VOTEC)
gv = 2.5
def _cc(i, j):
    return VX0 + j * (cellv + gv) + cellv / 2, VY0 + i * (cellv + gv) + cellv / 2
hx_, hy_ = _cc(*hinge)
for (i, j) in part:
    cx_, cy_ = _cc(i, j)
    dx, dy = hx_ - cx_, hy_ - cy_
    L_ = (dx * dx + dy * dy) ** 0.5
    r_ = cellv * 0.38
    add(f'<line x1="{cx_ - dx / L_ * r_:.1f}" y1="{cy_ - dy / L_ * r_:.1f}" x2="{cx_ + dx / L_ * r_:.1f}" '
        f'y2="{cy_ + dy / L_ * r_:.1f}" stroke="white" stroke-width="2" marker-end="url(#arrw)"/>')
add(f'<circle cx="{hx_:.1f}" cy="{hy_:.1f}" r="{cellv * 0.28:.1f}" fill="#e03131"/>')
arrow([(BX + BW, R2), (FLX, R2)])
text((FLX + FEND) / 2, R2 + VS / 2 + 22, "per-pixel votes", F_SMALL)
text((FLX + FEND) / 2, R2 + VS / 2 + 48, f'{v("a", "i")}, {v("d", "i")}, {sc("c", "i")}, {v("o", "i")}', F_MATH - 3, math=True)
text((FLX + FEND) / 2, R2 + VS / 2 + 72, "axis, direction, type, offset to hinge", F_SMALL - 1, fill=MUTED)
box(PMX, R2 - BH / 2, PMW, BH, "part-\nweighted\nmean", VOTE, fs=21, r=10)
arrow([(FEND, R2), (PMX, R2)])
# weights = the mask
WYY = R2 - BH / 2 - 52
arrow([(HX + HS / 2, R1 + HS / 2 + 36), (HX + HS / 2, WYY), (PMX + PMW / 2, WYY), (PMX + PMW / 2, R2 - BH / 2)], dashed=True, color=DIM)
text(PMX + PMW / 2 - 12, WYY + 19, f'weights {sc("w", "i")} = {sc("M")}', F_SMALL, anchor="end", fill=MUTED)
arrow([(PMX + PMW, R2), (ADX, R2)])
text((PMX + PMW + ADX) / 2, R2 - 24, f'{v("n")}, {v("d")}, {sc("c")}, {v("&#293;")}', F_MATH - 2, math=True)

# --- row 3: scalar MLPs
box(BX, R3 - BH / 2, BW, BH, "Part-pooled\nMLPs", SCAL, sub="depths and sweep", fs=24, sfs=15)
arrow([(QX + QS / 2, QY + QS + 70), (QX + QS / 2, R3), (BX, R3)])
arrow([(BX + BW, R3), (BUS, R3), (BUS, ADY + ADH - 34), (ADX, ADY + ADH - 34)])
text(PMX + PMW / 2, R3 - 24, f'{sc("z", "p")}, {sc("z", "q")}, {sc("L")}', F_MATH, math=True)

# ================================================================ decoder group
box(ADX, ADY, ADW, ADH, "Analytic\ndecoder", GEOM, sub="lift with K,\nrender the trajectory", sfs=15)
text(ADX + ADW / 2, ADY - 34, f'intrinsics {v("K")}', F_SMALL, fill=MUTED)
arrow([(ADX + ADW / 2, ADY - 18), (ADX + ADW / 2, ADY)], color=DIM)

# output panel
photo(OX, OY, OS, "clipout")


def P(uv):
    return (OX + uv[0] * OS, OY + uv[1] * OS)


# GT part mask (magenta, dark outline), hinge axis (red), interaction point ring + sweep arc (yellow)
add(f'<image x="{OX}" y="{OY}" width="{OS}" height="{OS}" clip-path="url(#clipout)" href="data:image/png;base64,{MASK}"/>')
add(f'<clipPath id="clipout2"><rect x="{OX}" y="{OY}" width="{OS}" height="{OS}" rx="10"/></clipPath>')
add('<g clip-path="url(#clipout2)">')
axis_pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in (P(uv) for uv in META["axis_uv"]))
add(f'<polyline points="{axis_pts}" fill="none" stroke="white" stroke-width="5.5" stroke-linecap="round" stroke-opacity="0.7"/>')
add(f'<polyline points="{axis_pts}" fill="none" stroke="#e03131" stroke-width="3" stroke-linecap="round"/>')
tr = [P(uv) for uv, ok in zip(META["traj_uv"], META["traj_valid"]) if ok]
arc = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in tr)
add(f'<path d="{arc}" fill="none" stroke="#333" stroke-width="5.5" stroke-linecap="round" stroke-linejoin="round"/>')
add(f'<path d="{arc}" fill="none" stroke="#ffd43b" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" marker-end="url(#arry)"/>')
px, py = P(META["point_uv"])
add(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5" fill="none" stroke="#333" stroke-width="4"/>')
add(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5" fill="none" stroke="#ffd43b" stroke-width="2"/>')
add('</g>')
frame(OX, OY, OS)
arrow([(ADX + ADW, MID), (OX, MID)])
text(OX + OS / 2, OY - 74, "mask, point, type, axis,", F_SMALL, fill=MUTED)
text(OX + OS / 2, OY - 52, "hinge, sweep, trajectory", F_SMALL, fill=MUTED)
text(OX + OS / 2, OY - 24, f'{sc("M")}, {v("p")}, {sc("c")}, {v("n")}, {v("q")}, {sc("L")}, {sc("γ")}(s)', F_MATH - 2, math=True)

# supervision of the decoded trajectory
box(S1X, SY - SH / 2, SW, SH, "2D projection loss", "white", sub="hand tracks, human video", fs=21, sfs=15, stroke=DIM)
box(S2X, SY - SH / 2, SW, SH, "3D closed-form loss", "white", sub="full-sweep arcs, SceneFun3D", fs=21, sfs=15, stroke=DIM)
arrow([(S1X + SW / 2, SY - SH / 2), (S1X + SW / 2, ADY + ADH)], dashed=True, color=DIM)
arrow([(S2X + SW / 2, SY - SH / 2), (S2X + SW / 2, OY + OS)], dashed=True, color=DIM)
text((S1X + S2X + SW) / 2, SY + SH / 2 + 26, "supervision of the decoded trajectory", F_SMALL, fill=MUTED)

add("</svg>")
(OUT / "model.svg").write_text("\n".join(parts))
print("wrote", OUT / "model.svg")

import datetime, subprocess, sys
commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=OUT).stdout.strip() or None
(OUT / "manifest.yaml").write_text(
    f"command: {' '.join(sys.argv)}\n"
    f"tool_commit: {commit}\n"
    f"generated: {datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')}\n"
    "experiment: 20260913_joint4_decoder_l2anchor_dense\n"
    "config: config/joint4_decoder_l2anchor_dense.yaml\n"
    "checkpoint: null  # architecture schematic, no inference\n"
    "example_image: sample/frame.png (SceneFun3D val 1773, dump_sample.py)\n"
    "outputs: [model.svg, model.pdf, preview.png]\n"
    f"canvas_px: [{W}, {H}]\n"
)
