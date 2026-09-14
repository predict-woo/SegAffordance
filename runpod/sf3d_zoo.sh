#!/bin/bash
# SF3D validation model zoo (2026-09-14): every joint / control / ablation checkpoint on the SAME 20 random
# val frames (sf3d_vis_val.py, seed 3), one model per run (a 24 GB card cannot hold 27 DINOv3 copies), then
# per-frame grids: GT panel + every model's prediction panel, 7 columns, labelled.
# Run detached ON THE POD:  nohup bash runpod/sf3d_zoo.sh > experiments/sf3d_zoo.log 2>&1 &
# Marker: ZOO_DONE.
cd /workspace/SegAffordance
export PATH=/opt/venv/bin:$PATH
export TORCHINDUCTOR_CACHE_DIR=/root/inductor_cache TRITON_CACHE_DIR=/root/triton_cache HF_HOME=/root/hfcache HF_HUB_OFFLINE=1
OUT=viz/20260914_sf3d_model_zoo
DENSE=config/sf3d_test_decoder_rgb_scalefree_dense.yaml
# name | config | ckpt  (order = grid order; families grouped)
MODELS="
joint4_dct|experiments/20260910_joint4_dct_rgb_scalefree/config.yaml|experiments/20260910_joint4_dct_rgb_scalefree/checkpoints/best-epoch12-sf3dval0.9688.ckpt
joint4_dctv2|experiments/20260910_joint4_dctv2_rgb_scalefree/config.yaml|experiments/20260910_joint4_dctv2_rgb_scalefree/checkpoints/best-epoch13-sf3dval1.1888.ckpt
dct_chain|experiments/20260910_sf3d_g19_dct_rgb_scalefree_ft_multi3dct/config.yaml|experiments/20260910_sf3d_g19_dct_rgb_scalefree_ft_multi3dct/checkpoints/best-epoch24-valloss1.0178.ckpt
dec_base|experiments/20260911_joint4_decoder_rgb_scalefree/config.yaml|experiments/20260911_joint4_decoder_rgb_scalefree/checkpoints/best-epoch16-sf3dval1.0089.ckpt
dec_seed7|experiments/20260912_joint4_decoder_seed7/config.yaml|experiments/20260912_joint4_decoder_seed7/checkpoints/best-epoch19-sf3dval1.0964.ckpt
l2anchor|experiments/20260912_joint4_decoder_l2anchor/config.yaml|experiments/20260912_joint4_decoder_l2anchor/checkpoints/best-epoch17-sf3dval1.1495.ckpt
h1anchor|experiments/20260912_joint4_decoder_h1anchor/config.yaml|experiments/20260912_joint4_decoder_h1anchor/checkpoints/best-epoch16-sf3dval1.1649.ckpt
cfframe|experiments/20260912_joint4_decoder_cfframe/config.yaml|experiments/20260912_joint4_decoder_cfframe/checkpoints/best-epoch17-sf3dval1.1178.ckpt
cfframe_seed7|experiments/20260912_joint4_decoder_cfframe_seed7/config.yaml|experiments/20260912_joint4_decoder_cfframe_seed7/checkpoints/best-epoch17-sf3dval1.0880.ckpt
cfframe_a3|experiments/20260912_joint4_decoder_cfframe_a3/config.yaml|experiments/20260912_joint4_decoder_cfframe_a3/checkpoints/best-epoch19-sf3dval1.2559.ckpt
cfframe_query|config/joint4_decoder_cfframe_query.yaml|experiments/20260913_joint4_decoder_cfframe_query/checkpoints/best-epoch16-sf3dval1.1911.ckpt
attnpool|config/joint4_decoder_l2anchor_attnpool.yaml|experiments/20260913_joint4_decoder_l2anchor_attnpool/checkpoints/best-epoch19-sf3dval1.1630.ckpt
query|config/joint4_decoder_l2anchor_query.yaml|experiments/20260913_joint4_decoder_l2anchor_query/checkpoints/best-epoch16-sf3dval1.1168.ckpt
query_seed7|config/joint4_decoder_l2anchor_query_seed7.yaml|experiments/20260913_joint4_decoder_l2anchor_query_seed7/checkpoints/best-epoch19-sf3dval1.1176.ckpt
query_l4|config/joint4_decoder_l2anchor_query_l4.yaml|experiments/20260913_joint4_decoder_l2anchor_query_l4/checkpoints/best-epoch16-sf3dval1.1337.ckpt
query_eps01|config/joint4_decoder_l2anchor_query_eps01.yaml|experiments/20260913_joint4_decoder_l2anchor_query_eps01/checkpoints/best-epoch13-sf3dval1.1076.ckpt
query_pos|config/joint4_decoder_l2anchor_query_pos.yaml|experiments/20260913_joint4_decoder_l2anchor_query_pos/checkpoints/best-epoch19-sf3dval1.1439.ckpt
query_w1024|config/joint4_decoder_l2anchor_query_w1024.yaml|experiments/20260913_joint4_decoder_l2anchor_query_w1024/checkpoints/best-epoch19-sf3dval1.0775.ckpt
mlp1024|config/joint4_decoder_l2anchor_mlp1024.yaml|experiments/20260913_joint4_decoder_l2anchor_mlp1024/checkpoints/best-epoch13-sf3dval1.1247.ckpt
dense|$DENSE|experiments/20260913_joint4_decoder_l2anchor_dense/checkpoints/best-epoch13-sf3dval0.9789.ckpt
dense_seed7|config/joint4_decoder_l2anchor_dense_seed7.yaml|experiments/20260913_joint4_decoder_l2anchor_dense_seed7/checkpoints/best-epoch19-sf3dval1.0205.ckpt
dense_off|config/joint4_decoder_l2anchor_dense_off.yaml|experiments/20260913_joint4_decoder_l2anchor_dense_off/checkpoints/best-epoch17-sf3dval1.1404.ckpt
dense_d2d|config/joint4_decoder_l2anchor_dense_d2d.yaml|experiments/20260913_joint4_decoder_l2anchor_dense_d2d/checkpoints/best-epoch13-sf3dval1.0920.ckpt
sf3d_only|$DENSE|experiments/20260913_sf3d_decoder_l2anchor_dense/checkpoints/best-epoch13-sf3dval1.1144.ckpt
directloss|$DENSE|experiments/20260914_joint4_decoder_dense_directloss/checkpoints/best-epoch16-sf3dval0.7608.ckpt
sampledtraj|$DENSE|experiments/20260914_joint4_decoder_dense_sampledtraj/checkpoints/best-epoch19-sf3dval1.1797.ckpt
field|config/joint4_decoder_field.yaml|experiments/20260913_field_joint4_l2anchor/checkpoints/best-epoch14-sf3dval1.0536.ckpt
"
mkdir -p $OUT
echo "$MODELS" | grep -v '^\s*$' | while IFS='|' read -r name cfg ck; do
  [ -d "$OUT/$name" ] && [ "$(ls $OUT/$name/*.png 2>/dev/null | wc -l)" -ge 20 ] && { echo "skip $name (done)"; continue; }
  extra=""; [ "$name" = "field" ] && extra="--field-names field"
  echo "=== $name $(date +%H:%M:%S)"
  python tools/sf3d_vis_val.py --model $name $cfg $ck $extra --out $OUT/$name --num 20 --seed 3 2>&1 | grep -E 'done|Error|Traceback|missing' | tail -2
done
echo "=== grids $(date +%H:%M:%S)"
python - <<'EOF'
import cv2, glob, os, numpy as np
OUT='viz/20260914_sf3d_model_zoo'
names=[l.split('|')[0] for l in open('runpod/sf3d_zoo.sh').read().split('MODELS="')[1].split('"')[0].strip().splitlines() if l.strip()]
frames=sorted(os.path.basename(f) for f in glob.glob(f'{OUT}/dense/*.png'))
ncol=7
for fr in frames:
    ref=cv2.imread(f'{OUT}/dense/{fr}'); h=ref.shape[0]
    panels=[('GT', ref[:, :h])]
    for n in names:
        p=f'{OUT}/{n}/{fr}'
        if os.path.exists(p):
            im=cv2.imread(p); panels.append((n, im[:, h:2*h]))
    ph=360; sc=ph/h
    tiles=[(n, cv2.resize(im,(ph,ph),interpolation=cv2.INTER_AREA)) for n,im in panels]
    rows=(len(tiles)+ncol-1)//ncol; pad=6; hdr=26
    W=ncol*(ph+pad)+pad; H=rows*(ph+hdr+pad)+pad
    canvas=np.full((H,W,3),245,np.uint8)
    for k,(n,t) in enumerate(tiles):
        r,c=divmod(k,ncol); x=pad+c*(ph+pad); y=pad+r*(ph+hdr+pad)
        cv2.putText(canvas,n,(x+4,y+19),cv2.FONT_HERSHEY_SIMPLEX,0.6,(20,20,20),2,cv2.LINE_AA)
        canvas[y+hdr:y+hdr+ph, x:x+ph]=t
    out=f'{OUT}/grid_{fr[:-4]}.jpg'; cv2.imwrite(out, canvas, [cv2.IMWRITE_JPEG_QUALITY, 85]); print('grid', out)
EOF
echo "ZOO_DONE $(date)"
