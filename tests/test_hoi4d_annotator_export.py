import importlib.util, pathlib, pickle
import numpy as np
from test_hoi4d_annotator_scene import data_root, SEQ, KEYS   # fixture reuse

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "ann", ROOT / "tools" / "hoi4d_annotate_articulation.py")
ann = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ann)


def test_export_projects_world_to_each_camera(data_root, tmp_path):
    ds = ann.Dataset(data_root, data_root / "hands")
    store = ann.AnnotationStore(tmp_path / "annos")
    store.save(SEQ, "C4", [{"window_events": ["open"], "type": "trans",
                            "axis_world": [0.0, 0.0, 1.0],
                            "origin_world": [0.5, 0.0, 1.0], "flag": None}])
    out = tmp_path / "export_sf3d.pkl"
    n = ann.export_sf3d(ds, store, out)
    assert n == len(KEYS)
    data = pickle.loads(out.read_bytes())
    for key in KEYS:
        e = data[key]
        assert e["motion_type"] == "trans"
        np.testing.assert_allclose(e["motion_dir_3d_camera_coords"], [0, 0, 1], atol=1e-12)
        f = int(key[-3:])
        np.testing.assert_allclose(
            e["motion_origin_3d_camera_coords"], [0.5 - 0.01 * f, 0.0, 1.0], atol=1e-9)


def test_flagged_sequences_skipped(data_root, tmp_path):
    ds = ann.Dataset(data_root, data_root / "hands")
    store = ann.AnnotationStore(tmp_path / "annos")
    store.save(SEQ, "C4", [{"window_events": ["open"], "type": "rot",
                            "axis_world": [0, 0, 1.0], "origin_world": [0, 0, 0],
                            "flag": "bad-poses"}])
    n = ann.export_sf3d(ds, store, tmp_path / "e.pkl")
    assert n == 0
