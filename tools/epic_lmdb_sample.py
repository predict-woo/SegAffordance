"""Render a review sheet of random records from an EPIC 2D LMDB
(tools/epic_process_2d.py output): stored 512x512 frame, stored mask coords
(red), 2D knuckle trajectory (green; cyan = onset, magenta = end), header =
noun / verb / hand / d / T, second line = description.
Usage: python tools/epic_lmdb_sample.py --lmdb /workspace/datasets/epic_processed_2d \
           --out viz/20260908_epic_v1_lmdb_sample --n 20 [--seed 3]
"""
import argparse, lmdb, os, pickle, random, json
import cv2, numpy as np

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--lmdb", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=20); ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--uniform", action="store_true", help="plain random sample instead of stratified by noun")
    ap.add_argument("--work", default=None, help="work dir: copy each sampled record's QA panel.jpg into --out")
    a = ap.parse_args()
    random.seed(a.seed); os.makedirs(a.out, exist_ok=True)
    env = lmdb.open(f"{a.lmdb}/data.lmdb", readonly=True, lock=False); envf = lmdb.open(f"{a.lmdb}/frames.lmdb", readonly=True, lock=False)
    with env.begin() as t: keys = [k for k, _ in t.cursor()]
    # stratify by noun so rare fixtures show up
    bynoun = {}
    with env.begin() as t:
        for k in keys: bynoun.setdefault(pickle.loads(t.get(k))["epic"]["noun"], []).append(k)
    picks = []
    nouns = sorted(bynoun, key=lambda n: -len(bynoun[n]))
    if a.uniform: picks = random.sample(keys, min(a.n, len(keys)))
    while len(picks) < min(a.n, len(keys)):
        for n in nouns:
            if len(picks) >= a.n: break
            rest = [k for k in bynoun[n] if k not in picks]
            if rest: picks.append(random.choice(rest))
    tiles = []; manifest = []
    with env.begin() as t, envf.begin() as tf:
        for k in picks:
            r = pickle.loads(t.get(k)); f = pickle.loads(tf.get(k))
            im = cv2.imdecode(np.frombuffer(f["jpeg"], np.uint8), cv2.IMREAD_COLOR); S = im.shape[0]
            W, H = f["orig_size"]; sx, sy = S / W, S / H
            ov = im.copy()
            for y, x in r["mask_coordinates_yx"]: cv2.circle(ov, (int(x * sx), int(y * sy)), 1, (0, 0, 255), -1)
            im = cv2.addWeighted(im, 0.5, ov, 0.5, 0)
            pts = [(int(x * sx), int(y * sy)) for x, y in r["trajectory_2d_image_coords"]]
            for p, q in zip(pts, pts[1:]): cv2.line(im, p, q, (0, 255, 0), 2)
            cv2.circle(im, pts[0], 5, (255, 255, 0), -1); cv2.circle(im, pts[-1], 5, (255, 0, 255), -1)
            e = r["epic"]; cv2.rectangle(im, (0, 0), (S, 40), (0, 0, 0), -1)
            cv2.putText(im, f"{e['noun']} {e['verb']} {e['hand']} d={e['d']:+d} T={len(pts)} {e['scale_regime']}", (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.putText(im, f"{k.decode()}  '{r['description'][:30]}'", (4, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
            tiles.append(im); manifest.append(k.decode())
            if a.work:
                import shutil; nid = k.decode().split("/")[1]
                src = f"{a.work}/{nid}/panel.jpg"
                if os.path.exists(src): shutil.copy(src, f"{a.out}/panel_{len(manifest):02d}_{e['noun'].replace(':', '-')}_{nid}.jpg")
    cols = 4; rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles) - len(tiles) % cols, cols)]
    if len(tiles) % cols: last = tiles[-(len(tiles) % cols):]; last += [np.zeros_like(tiles[0])] * (cols - len(last)); rows.append(np.hstack(last))
    cv2.imwrite(f"{a.out}/records_sample.jpg", np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
    json.dump({"keys": manifest, "lmdb": a.lmdb, "seed": a.seed, "total_records": len(keys)}, open(f"{a.out}/sample_keys.json", "w"), indent=1)
    print("sample written:", len(tiles), "of", len(keys), "records ->", f"{a.out}/records_sample.jpg")

if __name__ == "__main__": main()
