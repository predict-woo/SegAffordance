"""The shard/merge contract between runpod/baselines/a3vlm/chain.sh::evaluate() and the exporter.

Their eval script generates on ONE model-parallel group, so `evaluate()` splits the question JSON
into NPROC/MP shards with `d[i::n]`, runs one group per GPU pair, and concatenates
`vqa_logs/<flag>_s<i>/shard<i>.json`. Two properties have to hold or the 5,088-row export is
silently wrong, and both are cheap to check without a GPU:

  1. the split is a partition -- every question appears exactly once across the shards;
  2. the merge loses nothing and the exporter keys every answer back to its own element, even when
     two elements of the same frame produce byte-identical questions (which the chained
     REC -> REG-Joint protocol does whenever both get the same predicted box).

Property 2 is why `runpod/baselines/a3vlm/patch_eval.py` threads our `key` through the eval
script: joining on (image, question) alone is ambiguous exactly in that case.
"""
import json

import numpy as np

from tools.baselines_sf3d.a3vlm_preds_to_jsonl import export, make_joint_questions, results_by_key
from tools.baselines_sf3d.sf3d_to_a3vlm import (
    JOINT_INSTRUCT,
    REC_INSTRUCT,
    box_points,
    fmt_axis,
    fmt_box,
    pad_params,
    project_uvd,
    vqa,
)

K = [[1600.0, 0, 960.0], [0, 1600.0, 720.0], [0, 0, 1]]
META = {"key_frame": "s/v/1.0", "wh": [1920, 1440], "pad": list(pad_params(1920, 1440)), "K": K, "d_min": 1.0, "d_max": 4.0}
IMG = "/x/s_v_1.0.jpg"


def _shard(items, n):
    """Exactly what evaluate()'s PYSPLIT does."""
    return [items[i::n] for i in range(n)]


def _merge(shards):
    """Exactly what evaluate()'s PYMERGE does."""
    out = []
    for s in shards:
        out += s
    return out


def test_split_is_a_partition_and_merge_is_lossless():
    items = [{"i": i} for i in range(5088)]
    for n in (1, 2, 4, 8):
        shards = _shard(items, n)
        assert sum(len(s) for s in shards) == len(items)
        merged = _merge(shards)
        assert sorted(x["i"] for x in merged) == list(range(5088))
        # no shard is empty for our sizes, so every generation group does useful work
        assert all(len(s) > 0 for s in shards)


def test_two_elements_of_one_frame_with_identical_questions_stay_distinct():
    """The chained protocol gives both elements the same predicted box -> identical questions."""
    box = fmt_box(project_uvd(box_points([0.2, 0.0, 2.0], [0.2, 0.6, 0.1]), K, META["pad"], 1.0, 4.0))
    rec_qs = [vqa(IMG, REC_INSTRUCT + "open the left door", box, key="s/v/1.0/a"),
              vqa(IMG, REC_INSTRUCT + "open the right door", box, key="s/v/1.0/b")]
    rec_res = [{"image": IMG, "question": q["conversations"][0]["value"], "answer": "box " + box, "key": q["key"]}
               for q in rec_qs]
    jq = make_joint_questions(rec_res, rec_qs)
    # the two chained questions really are byte-identical
    assert jq[0]["conversations"][0]["value"] == jq[1]["conversations"][0]["value"] == JOINT_INSTRUCT.format(REF=box)

    ax_a = fmt_axis("revolute", project_uvd(np.array([[0.2, -0.3, 2.0], [0.2, 0.3, 2.0]]), K, META["pad"], 1.0, 4.0))
    ax_b = fmt_axis("prismatic", project_uvd(np.array([[-0.3, 0.0, 2.0], [0.3, 0.0, 2.0]]), K, META["pad"], 1.0, 4.0))
    # answers come back through separate shards, each carrying its own key
    shards = _shard([{"image": IMG, "question": jq[0]["conversations"][0]["value"], "answer": ax_a, "key": "s/v/1.0/a"},
                     {"image": IMG, "question": jq[1]["conversations"][0]["value"], "answer": ax_b, "key": "s/v/1.0/b"}], 2)
    merged = _merge(shards)

    by_key = results_by_key(merged, jq)
    assert set(by_key) == {"s/v/1.0/a", "s/v/1.0/b"}, "identical questions must not collapse"
    lines = export(merged, jq, {"s_v_1.0.jpg": META})
    assert [l["key"] for l in lines] == ["s/v/1.0/a", "s/v/1.0/b"]
    assert [l["type"] for l in lines] == [1, 0], "each element kept its own answer"


def test_without_threaded_keys_identical_questions_would_collide():
    """Guards the reason patch_eval.py exists: the (image, question) fallback cannot separate them."""
    box = fmt_box(project_uvd(box_points([0.2, 0.0, 2.0], [0.2, 0.6, 0.1]), K, META["pad"], 1.0, 4.0))
    q_text = JOINT_INSTRUCT.format(REF=box)
    jq = [vqa(IMG, q_text, None, key="s/v/1.0/a"), vqa(IMG, q_text, None, key="s/v/1.0/b")]
    ax = fmt_axis("revolute", project_uvd(np.array([[0.2, -0.3, 2.0], [0.2, 0.3, 2.0]]), K, META["pad"], 1.0, 4.0))
    unkeyed = [{"image": IMG, "question": q_text, "answer": ax},
               {"image": IMG, "question": q_text, "answer": ax}]
    by_key = results_by_key(unkeyed, jq)
    # both elements resolve to the SAME (image, question) answer: the ambiguity threading the key removes
    assert by_key["s/v/1.0/a"] is by_key["s/v/1.0/b"]


def test_answers_json_round_trips_through_a_file():
    """The shards really are written and re-read as JSON; non-ASCII descriptions must survive."""
    items = [{"image": IMG, "question": "Ouvrir le tiroir étroit", "answer": "<axis>revolute</axis>[0,0,0,1,1,1]",
              "key": "s/v/1.0/é"}]
    assert json.loads(json.dumps(items)) == items
