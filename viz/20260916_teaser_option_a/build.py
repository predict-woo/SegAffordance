#!/usr/bin/env python3
"""EgoArt teaser, option A ("failure contrast"), real data.

Top row, what the model learns from: a loose spread of hand-video records drawn with only the hand
track (2D), and a SceneFun3D record as an oblique point cloud with its annotated hinge axis and sweep (3D).
Bottom row, a new object that the 3D data never contains: the ARCTIC laptop frame with its instruction,
the prior methods as one fanned stack under a single cross, and EgoArt's answer under a check.

Inputs (src/): downscaled copies of the pod renders in panels/ (see README.md).

    python3 build.py && rsvg-convert -w 2000 -o preview.png fig.svg && rsvg-convert -f pdf -o fig.pdf fig.svg
"""
import base64
import random
from pathlib import Path

OUT = Path(__file__).resolve().parent
INK, LINE, MUTED = "#2b2b2b", "#555555", "#6b6b6b"
OK, BAD = "#2f9e5b", "#d24a43"
FONT = "Helvetica, Arial, sans-serif"
parts = []
add = parts.append


def b64(name):
    return base64.b64encode((OUT / "src" / name).read_bytes()).decode()


def text(x, y, s, fs=20, anchor="middle", fill=INK, weight="normal", italic=False):
    st = ' font-style="italic"' if italic else ""
    add(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{fs}" fill="{fill}" text-anchor="{anchor}" '
        f'dominant-baseline="central" font-weight="{weight}"{st}>{s}</text>')


def photo(x, y, w, h, img, cid, align="xMidYMid", rx=10, border=True, parent_attrs=""):
    add(f'<clipPath id="{cid}"><rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}"/></clipPath>')
    mime = "image/png" if img.endswith(".png") else "image/jpeg"
    add(f'<image x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" preserveAspectRatio="{align} slice" '
        f'clip-path="url(#{cid})" href="data:{mime};base64,{b64(img)}"/>')
    if border:
        add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="none" stroke="{LINE}" stroke-width="1.4"/>')


def arrow(x1, y1, x2, y2, width=2.2):
    add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{LINE}" stroke-width="{width}" marker-end="url(#arr)"/>')


def mark(cx, cy, ok, r=17):
    add(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="{OK if ok else BAD}" stroke="white" stroke-width="2.5"/>')
    k = r * 0.45
    if ok:
        add(f'<path d="M {cx - k:.1f} {cy:.1f} L {cx - k * 0.25:.1f} {cy + k * 0.8:.1f} L {cx + k * 1.1:.1f} {cy - k * 0.85:.1f}" '
            f'fill="none" stroke="white" stroke-width="3.6" stroke-linecap="round" stroke-linejoin="round"/>')
    else:
        add(f'<path d="M {cx - k:.1f} {cy - k:.1f} L {cx + k:.1f} {cy + k:.1f} M {cx + k:.1f} {cy - k:.1f} L {cx - k:.1f} {cy + k:.1f}" '
            f'stroke="white" stroke-width="3.6" stroke-linecap="round"/>')


# ---------------------------------------------------------------- layout (1000 x 720 design units)
W, H = 1000, 810
F_HEAD, F_LAB, F_SUB = 26, 28, 24      # ~6.5 / 7 / 6 pt at column width
add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
add('<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
    f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{LINE}"/></marker></defs>')
add(f'<rect width="{W}" height="{H}" fill="white"/>')

# ---- top row: what the model learns from
text(30, 28, "LEARNS FROM", F_HEAD, anchor="start", fill=MUTED, weight="600")
CY0, CH = 52, 250
# 2D card: six records, tracks only, scattered like prints on a table
CX, CW = 30, 445
add(f'<rect x="{CX}" y="{CY0}" width="{CW}" height="{CH}" rx="12" fill="none" stroke="{LINE}" stroke-width="1.2" stroke-dasharray="7 6"/>')
rng = random.Random(3)
fw, fh = 168, 168 * 9 / 16
slots = [(0.0, 0.05), (0.5, 0.0), (1.0, 0.12), (0.02, 0.95), (0.52, 1.0), (0.98, 0.9)]
order = [0, 2, 4, 1, 3, 5]                       # draw order, so the overlaps alternate
frames = ["hand_0.jpg", "hand_2.jpg", "hand_4.jpg", "hand_1.jpg", "hand_3.jpg", "hand_5.jpg"]
aligns = {"hand_4.jpg": "xMidYMin", "hand_5.jpg": "xMidYMin"}   # ARCTIC records: keep the top of the frame
for k, i in enumerate(order):
    sx, sy = slots[i]
    fx = CX + 14 + sx * (CW - fw - 28) + rng.uniform(-3, 3)
    fy = CY0 + 16 + sy * (CH - fh - 32)
    rot = rng.uniform(-6, 6)
    add(f'<g transform="rotate({rot:.1f} {fx + fw / 2:.1f} {fy + fh / 2:.1f})">')
    add(f'<rect x="{fx - 4:.1f}" y="{fy - 4:.1f}" width="{fw + 8}" height="{fh + 8:.1f}" rx="7" fill="white" stroke="{LINE}" stroke-width="1"/>')
    photo(fx, fy, fw, fh, frames[k], f"h{k}", aligns.get(frames[k], "xMidYMid"), rx=4, border=False)
    add('</g>')
text(CX + CW / 2, CY0 + CH + 26, "2D human video", F_LAB, weight="600")
text(CX + CW / 2, CY0 + CH + 56, "hand tracks, no 3D labels", F_SUB, fill=MUTED)

# 3D card: the point cloud
PX, PW = 525, 445
photo(PX, CY0, PW, CH, "pointcloud.png", "pc", rx=12)
# a small axis triad in the corner as the "this is 3D" cue
tx, ty = PX + PW - 44, CY0 + 40
for c, dx, dy in (("#e05555", 26, 0), ("#5cc66a", 0, -26), ("#5b8def", -15, 14)):
    add(f'<line x1="{tx}" y1="{ty}" x2="{tx + dx}" y2="{ty + dy}" stroke="{c}" stroke-width="2.6" stroke-linecap="round"/>')
text(PX + PW / 2, CY0 + CH + 26, "3D scans", F_LAB, weight="600")
text(PX + PW / 2, CY0 + CH + 56, "articulation labels, limited variety", F_SUB, fill=MUTED)

# ---- divider
add(f'<line x1="30" y1="{CY0 + CH + 82}" x2="{W - 30}" y2="{CY0 + CH + 82}" stroke="#d5d5d5"/>')

# ---- bottom row: a new object
RY = CY0 + CH + 110
text(30, RY, "NEW OBJECT, SEEN ONLY IN HUMAN VIDEO", F_HEAD, anchor="start", fill=MUTED, weight="600")
TY = RY + 26
TW, TH = 200, 150
photo(30, TY + 24, TW, TH, "laptop_frame.jpg", "test", rx=8)
cxb, by0, bw, bh, r = 30 + TW / 2, TY + 24 + TH + 18, 226, 76, 12
bx0 = cxb - bw / 2
add(f'<path d="M {bx0 + r} {by0} L {cxb - 11} {by0} L {cxb} {by0 - 13} L {cxb + 11} {by0} L {bx0 + bw - r} {by0} '
    f'A {r} {r} 0 0 1 {bx0 + bw} {by0 + r} L {bx0 + bw} {by0 + bh - r} A {r} {r} 0 0 1 {bx0 + bw - r} {by0 + bh} '
    f'L {bx0 + r} {by0 + bh} A {r} {r} 0 0 1 {bx0} {by0 + bh - r} L {bx0} {by0 + r} A {r} {r} 0 0 1 {bx0 + r} {by0} Z" '
    f'fill="white" stroke="{LINE}" stroke-width="1.6" stroke-linejoin="round"/>')
text(cxb, by0 + 26, "“open the", F_SUB + 2, italic=True)
text(cxb, by0 + 54, "laptop screen”", F_SUB + 2, italic=True)
arrow(30 + TW + 12, TY + 24 + TH / 2, 30 + TW + 60, TY + 24 + TH / 2)

# scattered pile of prior methods: five smaller cards tossed across the slot so each one shows,
# 3DOI on top and nearly square-on
SW, SH = 148, 148 * 3 / 4
RX0, RY0 = 300, TY + 22                            # pile region origin (between the arrow and "vs.")
pile = [  # (image, dx, dy, rot, name, chip corner), drawn back to front; the corner is one each card keeps visible
    ("laptop_OPDFormer-P.jpg", 112, -20, 9, "OPDFormer-P", "tr"),
    ("laptop_MOPD.jpg", 0, 26, -8, "MOPD", "tr"),
    ("laptop_OPDFormer-C.jpg", 172, 54, -5, "OPDFormer-C", "tr"),
    ("laptop_A3VLM.jpg", 10, 86, 6, "A3VLM", "tr"),
    ("laptop_3DOI.jpg", 152, 126, -2, "3DOI", "tr"),
]
SX, SY = RX0 + pile[-1][1], RY0 + pile[-1][2]     # the front card, for the label
for i, (img, dx, dy, rot, name, corner) in enumerate(pile):
    gx, gy = RX0 + dx, RY0 + dy
    add(f'<g transform="rotate({rot} {gx + SW / 2:.1f} {gy + SH / 2:.1f})">')
    add(f'<rect x="{gx - 3}" y="{gy - 3}" width="{SW + 6}" height="{SH + 6:.1f}" rx="8" fill="white" stroke="{LINE}" stroke-width="1"/>')
    photo(gx, gy, SW, SH, img, f"s{i}", rx=6, border=False)
    # method name chip, kept in the corner of the card that stays uncovered
    fs = 13; cw, chh = len(name) * fs * 0.6 + 12, fs + 8
    cx = gx + 6 if corner in ("tl", "bl") else (gx + 74 if corner == "tm" else gx + SW - 6 - cw)
    cy = gy + SH - 6 - chh if corner == "bl" else (gy + 6 if corner in ("tr", "tm") else gy + 30)
    add(f'<rect x="{cx:.1f}" y="{cy:.1f}" width="{cw:.0f}" height="{chh}" rx="5" fill="white" fill-opacity="0.92" stroke="{LINE}" stroke-width="0.8"/>')
    text(cx + cw / 2, cy + chh / 2, name, fs, weight="600")
    add('</g>')
text(RX0 + 186, SY + SH + 34, "prior methods", F_LAB, weight="600")
mark(RX0 + 186 - 96 - 24, SY + SH + 34, False, r=15)
text(RX0 + 186, SY + SH + 64, "prismatic, wrong part", F_SUB, fill=MUTED)

text(688, TY + 24 + TH / 2, "vs.", F_LAB, fill=MUTED)

# EgoArt
EX, EY, EW = 715, TY + 4, 255
EH = EW * 3 / 4
photo(EX, EY, EW, EH, "laptop_dense.jpg", "ego", rx=8)
text(EX + EW / 2 + 12, EY + EH + 26, "EgoArt", F_LAB, weight="600")
mark(EX + EW / 2 + 12 - 50 - 24, EY + EH + 26, True, r=15)
text(EX + EW / 2, EY + EH + 56, "revolute, hinge on edge", F_SUB, fill=MUTED)


add("</svg>")
(OUT / "fig.svg").write_text("\n".join(parts))
print("wrote", OUT / "fig.svg")
