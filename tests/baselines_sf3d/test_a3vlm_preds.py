import numpy as np

from tools.baselines_sf3d import common as C
from tools.baselines_sf3d.a3vlm_preds_to_jsonl import export, make_joint_questions, question_text, to_prediction
from tools.baselines_sf3d.sf3d_to_a3vlm import parse_box
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


def _prompt(q):
    return "sys\n\n### Human: " + q + "\n### Assistant:"


def test_question_text_recovers_the_human_turn():
    assert question_text(_prompt("hello there")) == "hello there"
    assert question_text("plain question") == "plain question"


def test_to_prediction_round_trip():
    axis_pts = np.array([[0.2, -0.3, 2.0], [0.2, 0.3, 2.0]])
    ans = fmt_axis("revolute", project_uvd(axis_pts, K, META["pad"], 1.0, 4.0))
    box = project_uvd(box_points([0.2, 0.0, 2.0], [0.2, 0.6, 0.1]), K, META["pad"], 1.0, 4.0)
    p = to_prediction("k", ans, box, META)
    assert p["matched"] and p["type"] == 1
    assert np.allclose(p["axis_cam"], [0, 1, 0], atol=0.05)
    assert abs(p["origin_cam"][0] - 0.2) < 0.03 and abs(p["origin_cam"][2] - 2.0) < 0.05
    m = C.rle_decode(p["mask_rle"])
    assert m.shape == (1440, 1920) and m.sum() > 1000
    ys, xs = np.nonzero(m)
    assert abs(xs.mean() - (960 + 1600 * 0.2 / 2.0)) < 20 and abs(ys.mean() - 720) < 20
    # prismatic: origin at the segment midpoint
    p2 = to_prediction("k", fmt_axis("prismatic", project_uvd(axis_pts, K, META["pad"], 1.0, 4.0)), box, META)
    assert p2["type"] == 0 and abs(p2["origin_cam"][1]) < 0.05
    # garbage answers -> unmatched but keep the box mask
    p3 = to_prediction("k", "I cannot tell.", box, META)
    assert not p3["matched"] and p3["type"] is None and p3["mask_rle"] is not None
    assert to_prediction("k", "<axis>fixed</axis>[0.1,0.1,0.1,0.2,0.2,0.2]", None, META)["matched"] is False


def test_make_joint_and_export_chain():
    box = fmt_box(project_uvd(box_points([0.2, 0.0, 2.0], [0.2, 0.6, 0.1]), K, META["pad"], 1.0, 4.0))
    rec_qs = [vqa(IMG, REC_INSTRUCT + "open the door", box, key="s/v/1.0/a"), vqa(IMG, REC_INSTRUCT + "press it", box, key="s/v/1.0/b")]
    rec_res = [{"image": IMG, "question": _prompt(REC_INSTRUCT + "open the door"), "answer": "The box is " + box, "key": "s/v/1.0/a"},
               {"image": IMG, "question": _prompt(REC_INSTRUCT + "press it"), "answer": "nonsense", "key": "s/v/1.0/b"}]
    jq = make_joint_questions(rec_res, rec_qs)
    assert len(jq) == 2 and jq[0]["rec_failed"] is False and jq[1]["rec_failed"] is True
    assert jq[0]["conversations"][0]["value"] == JOINT_INSTRUCT.format(REF=box) and jq[0]["key"] == "s/v/1.0/a"
    ans = fmt_axis("revolute", project_uvd(np.array([[0.2, -0.3, 2.0], [0.2, 0.3, 2.0]]), K, META["pad"], 1.0, 4.0))
    # both chained questions are identical (same predicted box): only the key carried by the patched eval tells them apart
    joint_res = [{"image": IMG, "question": _prompt(jq[0]["conversations"][0]["value"]), "answer": ans, "key": "s/v/1.0/a"},
                 {"image": IMG, "question": _prompt(jq[1]["conversations"][0]["value"]), "answer": ans, "key": "s/v/1.0/b"}]
    lines = export(joint_res, jq, {"s_v_1.0.jpg": META})
    assert [l["key"] for l in lines] == ["s/v/1.0/a", "s/v/1.0/b"]
    assert lines[0]["matched"] and lines[0]["mask_rle"] is not None
    assert not lines[1]["matched"] and lines[1]["mask_rle"] is None  # REC failed -> unmatched even though the axis parsed


def test_export_gtbox_uses_question_box():
    box = fmt_box(project_uvd(box_points([0.2, 0.0, 2.0], [0.2, 0.6, 0.1]), K, META["pad"], 1.0, 4.0))
    ans = fmt_axis("prismatic", project_uvd(np.array([[0.2, -0.3, 2.0], [0.2, 0.3, 2.0]]), K, META["pad"], 1.0, 4.0))
    qs = [vqa(IMG, JOINT_INSTRUCT.format(REF=box), ans, key="s/v/1.0/a")]
    res = [{"image": IMG, "question": _prompt(qs[0]["conversations"][0]["value"]), "answer": ans}]
    lines = export(res, qs, {"s_v_1.0.jpg": META}, gt_box=True)
    assert lines[0]["matched"] and lines[0]["type"] == 0 and lines[0]["mask_rle"] is not None



def test_dot_stripped_answers_are_recovered():
    """eval_affordance_v2.py saves answer.replace('.', ''); our numbers are all "{:.2f}" in [0, 1]."""
    from tools.baselines_sf3d.a3vlm_preds_to_jsonl import answer_text
    stripped = {"answer": "<axis>prismatic</axis>[026,037,029,028,058,035]"}
    assert answer_text(stripped) == "<axis>prismatic</axis>[0.26,0.37,0.29,0.28,0.58,0.35]"
    box = {"answer": "[[021,047,023],[029,047,023],[021,049,023],[023,047,027],[030,049,027],[023,049,027],[030,047,027],[029,049,027]]"}
    assert parse_box(answer_text(box)) is not None and abs(parse_box(answer_text(box))[0, 0] - 0.21) < 1e-9
    assert answer_text({"answer": "[[100,000,050]]"}).startswith("[[1.00,0.00,0.50]]")
    # the patched script's raw answer wins, and normal answers pass through untouched
    assert answer_text({"answer": "021", "raw_answer": "<axis>revolute</axis>[0.21,0.5,0.5,0.2,0.6,0.5]"}).startswith("<axis>")
    assert answer_text({"answer": "<axis>revolute</axis>[0.21,0.50,0.50,0.20,0.60,0.50]"}) == "<axis>revolute</axis>[0.21,0.50,0.50,0.20,0.60,0.50]"
    assert answer_text({"answer": "I cannot tell"}) == "I cannot tell"



def test_accumulated_results_file_is_disambiguated_by_question_text():
    """Their eval script appends older answers to newer ones under the same key; the join must
    pick the answer to THIS question set, not the last one written."""
    from tools.baselines_sf3d.a3vlm_preds_to_jsonl import results_by_key
    box = fmt_box(project_uvd(box_points([0.2, 0.0, 2.0], [0.2, 0.6, 0.1]), K, META["pad"], 1.0, 4.0))
    ax = fmt_axis("revolute", project_uvd(np.array([[0.2, -0.3, 2.0], [0.2, 0.3, 2.0]]), K, META["pad"], 1.0, 4.0))
    joint_q = [vqa(IMG, JOINT_INSTRUCT.format(REF=box), ax, key="s/v/1.0/a")]
    results = [  # new answers first, then the appended older REC answer for the same key
        {"image": IMG, "question": _prompt(JOINT_INSTRUCT.format(REF=box)), "answer": ax, "key": "s/v/1.0/a"},
        {"image": IMG, "question": _prompt(REC_INSTRUCT + "open the door"), "answer": box, "key": "s/v/1.0/a"},
    ]
    r = results_by_key(results, joint_q)
    assert r["s/v/1.0/a"]["answer"] == ax
    lines = export(results, joint_q, {"s_v_1.0.jpg": META}, gt_box=True)
    assert lines[0]["matched"] and lines[0]["type"] == 1
