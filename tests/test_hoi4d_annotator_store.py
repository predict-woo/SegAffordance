import importlib.util, json, pathlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "ann", ROOT / "tools" / "hoi4d_annotate_articulation.py")
ann = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ann)

PART = {"window_events": ["open", "close"], "type": "rot",
        "axis_world": [0.0, 0.0, 1.0], "origin_world": [1.0, 2.0, 3.0], "flag": None}


def test_save_load_roundtrip(tmp_path):
    store = ann.AnnotationStore(tmp_path)
    p = store.save("SEQX", "C6", [PART])
    assert p.exists() and not list(tmp_path.glob("*.tmp"))
    got = store.load("SEQX")
    assert got["seq"] == "SEQX" and got["category"] == "C6"
    assert got["parts"] == [PART]
    assert "time" in got and "annotator" in got


def test_missing_returns_none(tmp_path):
    assert ann.AnnotationStore(tmp_path).load("NOPE") is None


def test_status(tmp_path):
    store = ann.AnnotationStore(tmp_path)
    store.save("A", "C4", [PART])
    bad = dict(PART); bad["flag"] = "bad-poses"
    store.save("B", "C4", [bad])
    assert store.status() == {"A": "annotated", "B": "flagged"}


def test_save_is_atomic_json(tmp_path):
    store = ann.AnnotationStore(tmp_path)
    path = store.save("A", "C4", [PART])
    json.load(open(path))   # valid JSON on disk
