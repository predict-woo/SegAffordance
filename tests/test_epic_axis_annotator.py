"""Smoke test for tools/epic_axis_annotator.py (no GPU, no pod): a synthetic cloud npz, a fake annotation, and the
export schema (axis normalised, keys present, unannotated records valid=false). Also round-trips the binary cloud
encoding the page consumes.   python3 -m pytest tests/test_epic_axis_annotator.py  (or run it directly)
"""
import json
import os
import struct
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
from epic_axis_annotator import build_export, encode_cloud, list_records, load_annotations, safe_name  # noqa: E402


def _write_cloud(d, key, n=1000):
    rng = np.random.default_rng(0)
    np.savez_compressed(os.path.join(d, safe_name(key) + ".npz"), xyz=rng.normal(size=(n, 3)).astype(np.float32),
                        rgb=rng.integers(0, 255, (n, 3), dtype=np.uint8), in_mask=rng.random(n) < 0.1,
                        point_uv=np.array([100.0, 50.0], np.float32), desc="open the drawer", key=key, type_label=0,
                        K_render=np.eye(3, dtype=np.float32), size=np.array([1024, 576], np.int32))


def test_export_schema():
    with tempfile.TemporaryDirectory() as td:
        clouds, out = os.path.join(td, "clouds"), os.path.join(td, "ann")
        os.makedirs(clouds); os.makedirs(out)
        _write_cloud(clouds, "P01_03/P01_03_27"); _write_cloud(clouds, "P02_01/P02_01_05")
        recs = list_records(clouds)  # no index.json -> scanned from the npz files
        assert [r["key"] for r in recs] == ["P01_03/P01_03_27", "P02_01/P02_01_05"]
        json.dump(dict(key="P01_03/P01_03_27", p1=[0.1, 0.2, 1.0], p2=[0.1, 0.5, 1.4], type="rot", note="hinge"),
                  open(os.path.join(out, safe_name("P01_03/P01_03_27") + ".json"), "w"))
        gt = build_export(recs, load_annotations(out, recs))
        g = gt["P01_03/P01_03_27"]
        assert g["valid"] and g["type"] == "rot" and g["note"] == "hinge"
        for k in ("axis_cam", "origin_cam", "type", "valid", "seq", "desc"):
            assert k in g
        assert abs(np.linalg.norm(g["axis_cam"]) - 1) < 1e-9
        assert np.allclose(g["axis_cam"], [0.0, 0.6, 0.8]) and g["origin_cam"] == [0.1, 0.2, 1.0]
        u = gt["P02_01/P02_01_05"]
        assert u["valid"] is False and u["type"] == "trans" and "axis_cam" not in u
        json.dumps(gt)  # serialisable


def test_cloud_encoding_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        _write_cloud(td, "P01_03/P01_03_27", n=50)
        rec = list_records(td)[0]
        buf = encode_cloud(os.path.join(td, rec["file"]), rec)
        ml = struct.unpack_from("<I", buf, 0)[0]
        meta = json.loads(buf[4:4 + ml])
        assert meta["n"] == 50 and meta["key"] == "P01_03/P01_03_27" and len(meta["K_render"]) == 3
        off = 4 + ml
        xyz = np.frombuffer(buf[off:off + 50 * 12], np.float32).reshape(50, 3); off += 50 * 12
        msk = np.frombuffer(buf[off + 150:off + 200], np.uint8)
        z = np.load(os.path.join(td, rec["file"]))
        assert np.array_equal(xyz, z["xyz"]) and np.array_equal(msk.astype(bool), z["in_mask"])


if __name__ == "__main__":
    test_export_schema(); test_cloud_encoding_roundtrip(); print("ok")
