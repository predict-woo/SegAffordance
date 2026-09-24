"""Render the final architecture using the original figure's graphic vocabulary."""
from pathlib import Path
import base64
import html
import re
import xml.etree.ElementTree as ET

OUT = Path(__file__).parent
parts = []
W, H = 3500, 1140

def text(x, y, s, size=28, bold=False, math=False, anchor='middle', rotate=None):
    transform = f' transform="rotate({rotate} {x} {y})"' if rotate else ''
    family = 'Times New Roman, serif' if math else 'Arial, Helvetica, sans-serif'
    if math:
        for a,b in [('v₂','v_2'),('v₃','v_3'),('v₄','v_4'),('nᵣ','n_r'),('nₜ','n_t'),('uₚ','u_p'),('zₚ','z_p'),('uq','u_q'),('zq','z_q'),('F′q','F′_q'),('Fq','F_q'),('τrot','τ_rot'),('τtrans','τ_trans')]:
            s=s.replace(a,b)
    content=html.escape(s)
    if math:
        content=re.sub(r'_([A-Za-z0-9]+)',r'<tspan baseline-shift="sub" font-size="70%">\1</tspan>',content)
    parts.append(f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" '
                 f'font-weight="{700 if bold else 400}" font-style="{"italic" if math else "normal"}" '
                 f'text-anchor="{anchor}" dominant-baseline="middle"{transform}>{content}</text>')

def rect(x,y,w,h,fill='#EEEEEE',rx=10,stroke='none',sw=1):
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')

def block(x,y,w,h,label,vertical=False,size=27,fill='#EEEEEE'):
    rect(x,y,w,h,fill)
    text(x+w/2,y+h/2,label,size,True,rotate=-90 if vertical else None)

def path(d,arrow=True,color='#1B1B1B',sw=2,dash=False):
    parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw}" '
                 f'stroke-linejoin="round" stroke-linecap="round"'
                 + (' marker-end="url(#arrow)"' if arrow else '')
                 + (' stroke-dasharray="7 6"' if dash else '') + '/>')

def maps(x,cy,h=100,n=3,width=15):
    for i in range(n):
        rect(x+i*width,cy-h/2,width,h,['#F6CACA','#FA9292','#FF1414'][min(i,2)],0)
    return x+n*width

def circle(x,y,r=5,fill='#202020'):
    parts.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}"/>')

# The source photo is reused from the original Figma screenshot, clipped by SVG.
image = base64.b64encode((OUT/'original-reference.png').read_bytes()).decode()
parts.append(f'<svg x="40" y="400" width="220" height="220" viewBox="494 76 153 153" overflow="hidden"><image width="1800" height="541" href="data:image/png;base64,{image}"/></svg>')
rect(40,400,220,220,'none',5,'#111',3)
text(150,668,'RGB image: I',32,math=True)
text(150,708,'3 × H × W',24,math=True)
text(145,108,'Text: T',33,math=True)
text(145,156,'“Open the right door”',23)
block(330,67,220,80,'dino.txt',False,30)
text(440,176,'Frozen text encoder',21)
path('M250 107H330')
# The sentence feature conditions FPN, dynamic filters and the scalar context.
path('M550 107H1980',False)
text(915,75,'Sentence feature s',25,math=True)
path('M791 107V333')
path('M1595 107V258')
path('M1970 107V185H2135V915H1775V934')
text(2040,171,'s',29,math=True)
# Word features enter the transformer on their own lane.
path('M550 134H580V215H1350V300')
text(1185,191,'Word features W',25,math=True)

block(305,300,80,435,'DINOv3 ViT-L/16',True,31)
text(345,782,'Frozen',23)
path('M260 510H305')
block(435,300,66,435,'Pyramid adapter',True,26)
path('M385 510H435')
path('M501 510H527',False)
path('M527 360V675',False)

# The three-scale CRIS FPN, using the same pink/red feature slabs and gray Conv bars.
rect(548,275,696,475,'#F5F5F5',12)
path('M791 275V346')
for cy,h,label in [(360,44,'v₄'),(515,82,'v₃'),(675,118,'v₂')]:
    path(f'M527 {cy}H578')
    maps(580,cy,h)
    text(603,cy-h/2-23,label,31,math=True)
    block(662,cy-60,40,120,'Conv',True,22,'#C9C9C9')
    path(f'M625 {cy}H662')
    path(f'M702 {cy}H743')
    maps(745,cy,h)
for cy,h in [(515,82),(675,118)]:
    block(850,cy-60,40,120,'Conv',True,22,'#C9C9C9')
    path(f'M790 {cy}H850')
    maps(930,cy,h)
    path(f'M890 {cy}H930')
circle(791,360,13,'#FFF')
parts.append('<circle cx="791" cy="360" r="13" fill="none" stroke="#111" stroke-width="2"/>')
text(791,360,'×',24)
path('M791 374V454H827V496')
text(810,425,'Up',21)
path('M969 554V606H827V656')
text(1004,591,'↓',24)
path('M808 360H970')
block(973,300,40,120,'Conv',True,22,'#C9C9C9')
block(993,455,40,120,'Conv',True,22,'#C9C9C9')
block(993,615,40,120,'Conv',True,22,'#C9C9C9')
path('M975 515H993')
path('M975 675H993')
block(1080,305,43,415,'Concat',True,25,'#C9C9C9')
path('M1013 360H1080')
path('M1033 515H1080')
path('M1033 675H1080')
maps(1140,515,126)
path('M1123 515H1140')
block(1190,450,38,130,'Conv',True,22,'#C9C9C9')
path('M1185 515H1190')
block(730,776,330,60,'Text-gated FPN',False,30)
text(1230,411,'Fq',32,math=True)
path('M1228 515H1300')
block(1300,300,100,435,'Transformer decoder',True,31)
text(1350,782,'3 layers',23)
path('M1400 515H1433')
maps(1435,515,170)
text(1457,397,'F′q',34,math=True)
path('M1480 515H1510',False)
path('M1510 390V978',False)

# Dynamic-filter localization head, retaining the original diagram's Y stack.
block(1560,260,70,265,'Dynamic Conv',True,27)
path('M1510 390H1560')
maps(1690,390,156,2)
path('M1630 390H1690')
text(1712,275,'Y',35,math=True)
path('M1697 309V232H3070V302')
text(2900,205,'Mask M',31,math=True)
path('M1712 468V577H1770')
block(1770,551,227,56,'Soft-argmax',False,26,'#C9C9C9')
path('M1997 579H2200V412H2300')
text(2100,550,'uₚ',32,math=True)
text(1820,525,'Point map',22)
path('M1697 468V513H1745V630H1958V635')
text(1770,494,'M',29,math=True)

# Dense articulation fields: separate rotation and translation vectors, type, hinge offsets.
block(1560,635,70,230,'Conv × 2 + 1×1',True,24)
path('M1510 748H1560')
path('M1630 748H1675')
for x,n,label in [(1675,3,'nᵣ'),(1734,3,'nₜ'),(1793,2,'c'),(1838,2,'u+Δu')]:
    maps(x,748,142,n,12)
    text(x+n*6,642,label,23 if label=='u+Δu' else 28,math=True)
block(1920,635,75,230,'Mask pooling',True,27)
path('M1871 748H1920')

text(1777,886,'Dense articulation voting',27,True)
path('M1995 699H2040V622H2650')
text(2410,592,'Axis n  ·  type c',31,math=True)
path('M1995 795H2226V509H2300')
text(2142,772,'uq',31,math=True)

# Scalar heads use pooled part/global/text context; pixel-local sampling supports depths.
block(1540,935,230,83,'Pool + context',False,26)
path('M1510 978H1540')
block(1815,935,172,83,'MLP heads',False,27)
path('M1770 978H1815')
text(1760,1062,'Part + global + text + pixel / local features',21)
path('M1987 961H2261V550H2300')
text(2085,934,'zₚ, zq',31,math=True)
path('M1987 996H2580V770H2650')
text(2410,968,'Path length ℓ',30,math=True)

# Geometric lifting and an analytic, rather than learned, trajectory decoder.
block(2300,346,94,223,'Back-project',True,29)
text(2350,272,'K',36,math=True)
path('M2347 300V346')
path('M2394 459H2650')
text(2515,421,'p, q ∈ ℝ³',32,math=True)
block(2650,365,108,456,'Analytic decoder',True,31)
text(2704,865,'Rotate / translate',24)
path('M2758 510H2890')

# Structured outputs, schematic geometry only.
rect(2910,310,270,246,'none',0,'#BFBFBF',2)
rect(2940,330,90,202,'#FAFAFA',0,'#A8A8A8',2)
parts.append('<path d="M3050 330L3218 393V551L3050 532Z" fill="#F5B5B5" fill-opacity="0.75" stroke="#222" stroke-width="2.5"/>')
path('M3050 588V287',True,sw=3)
circle(3188,457,7,'#E21A1A')
path('M3188 457C3297 480 3320 376 3270 332',True,color='#111',sw=2.5)
text(3300,450,'τrot',33,math=True)
text(3006,277,'n',32,math=True)
text(3081,602,'q',32,math=True)
circle(3050,532,5)
text(3300,574,'Rotation',27,True)
rect(2927,703,247,113,'none',0,'#BDBDBD',2)
parts.append('<path d="M2950 724H3153L3218 785H3015Z" fill="#F9DEDE" stroke="#222" stroke-width="2.5"/><path d="M3015 785H3218V864H3015Z" fill="#F5B5B5" stroke="#222" stroke-width="2.5"/><path d="M2950 724V802L3015 864" fill="none" stroke="#222" stroke-width="2.5"/>')
circle(3115,824,7,'#E21A1A')
path('M3115 824L3311 892',True,sw=2.5)
text(3320,807,'τtrans',33,math=True)
text(3293,936,'Translation',27,True)
text(3100,1019,'3D axis, hinge, interaction point & trajectory',24)
text(1750,1110,'Mask pooling uses ground-truth masks during training and predicted masks at inference.  •  Final plain-dense model; RGB-only input.',23)

svg = '<svg xmlns="http://www.w3.org/2000/svg" width="3500" height="1140" viewBox="0 0 3500 1140"><defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto" markerUnits="userSpaceOnUse"><path d="M1 1L8 5L1 9" fill="none" stroke="#1B1B1B" stroke-width="1.8"/></marker></defs><rect width="3500" height="1140" fill="white"/>' + ''.join(parts) + '</svg>'
root=ET.fromstring(svg)
assert root.attrib['viewBox']=='0 0 3500 1140'
assert len(root.findall('.//{http://www.w3.org/2000/svg}text')) > 35
assert all(s in svg for s in ['DINOv3','dino.txt','Mask pooling','Analytic decoder'])
(OUT/'model.svg').write_text(svg)
print(OUT/'model.svg')
