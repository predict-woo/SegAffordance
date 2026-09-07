#!/bin/bash
# Fetch VISOR sparse-annotation JSONs (train+val), frame_mapping.json and the
# noun-class table into /workspace/datasets/visor (main volume). Parallel wget.
set -u
B=https://data.bris.ac.uk/datasets/2v6cgv1x04ol22qp9rm9x2j6a7
D=/workspace/datasets/visor; mkdir -p $D/annotations/train $D/annotations/val $D/lists
cd $D
curl -s -m 60 "$B/frame_mapping.json" -o frame_mapping.json
curl -s -m 60 "$B/EPIC_100_noun_classes_v2.csv" -o EPIC_100_noun_classes_v2.csv
curl -s -m 60 "$B/README.txt" -o README.txt
for split in train val; do
  curl -s -m 60 "$B/GroundTruth-SparseAnnotations/annotations/$split/" | grep -oE 'href="[^"]+\.json"' | sed 's/href="//; s/"$//' > lists/ann_$split.txt
  echo "$split: $(wc -l < lists/ann_$split.txt) json files"
  sed "s#^#$B/GroundTruth-SparseAnnotations/annotations/$split/#" lists/ann_$split.txt \
    | xargs -P 16 -I{} sh -c 'f=$(basename {}); [ -s annotations/'$split'/$f ] || curl -s -m 600 -o annotations/'$split'/$f {}'
  echo "$split: $(ls annotations/$split | wc -l) downloaded, $(du -sh annotations/$split | cut -f1)"
done
for split in train val test; do
  curl -s -m 60 "$B/GroundTruth-SparseAnnotations/rgb_frames/$split/" | grep -oE 'href="P[0-9]+/"' | sed 's/href="//; s/"$//' > lists/rgb_${split}_participants.txt
done
ls -la frame_mapping.json EPIC_100_noun_classes_v2.csv README.txt
echo VISOR_ANN_DONE
