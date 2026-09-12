"""Derive a joint-decoder experiment arm (config + chain script + notes stub)
from an existing arm by overriding model/loss params.

  python tools/make_joint_arm.py --base config/joint4_decoder_l2anchor.yaml \
      --tag query_cfframe --exp 20260913_joint4_decoder_cfframe_query \
      --what "query readout on the cf_frame SF3D side" \
      --set model_params.articulation_readout=query --set loss_params.closed_form_frame_weight=1.0 \
      [--profile 3d:closed_form_frame_weight=1.0]   # per-source loss-profile override (model.source_profiles)

Writes config/joint4_decoder_<tag>.yaml, run_joint4dec_<tag>_chain.sh (from the base's chain
script, test calls carry the model-param overrides), experiments/<exp>/notes.md. Nothing is
launched. Values are parsed as YAML scalars.
"""
import argparse
import os
import re

import yaml


def _set(d, dotted, value):
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = yaml.safe_load(value)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--base-chain", default=None, help="chain script to derive from (default: guessed from base)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--what", required=True)
    ap.add_argument("--set", action="append", default=[], metavar="model_params.KEY=VAL | loss_params.KEY=VAL")
    ap.add_argument("--profile", action="append", default=[], metavar="PROFILE:KEY=VAL")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.base))
    base_exp = None
    for cb in cfg["trainer"]["callbacks"]:
        d = cb.get("init_args", {}).get("dirpath")
        if d:
            base_exp = d.split("/experiments/")[1].split("/")[0]
    assert base_exp, "no ModelCheckpoint dirpath in base"
    mp_over = {}
    for s in a.set:
        k, v = s.split("=", 1)
        if k.startswith("model_params."):
            _set(cfg["model"]["model_params"], k[len("model_params."):], v)
            mp_over[k[len("model_params."):]] = v
        elif k.startswith("loss_params."):
            _set(cfg["model"]["loss_params"], k[len("loss_params."):], v)
        else:
            _set(cfg, k, v)
    for p in a.profile:
        prof, kv = p.split(":", 1)
        k, v = kv.split("=", 1)
        _set(cfg["model"]["loss_profiles"][prof], k, v)
    text = yaml.safe_dump(cfg, sort_keys=False, width=120)
    text = text.replace(base_exp, a.exp)
    header = f"# JOINT decoder arm `{a.tag}` ({a.exp[:8]}): {a.what}. Derived from {a.base} by tools/make_joint_arm.py; overrides: {' '.join(a.set + a.profile) or 'none'}.\n"
    out_cfg = f"config/joint4_decoder_{a.tag}.yaml"
    open(out_cfg, "w").write(header + text)

    chain_src = a.base_chain or ("run_joint4dec_" + os.path.basename(a.base).replace("joint4_decoder_", "").replace(".yaml", "") + "_chain.sh")
    if not os.path.exists(chain_src):
        chain_src = "run_joint4dec_l2anchor_chain.sh"
    c = open(chain_src).read()
    c = re.sub(r"^# Pod .*\n", f"# Pod jdec-{a.tag} ({a.exp[:8]}): joint decoder arm {a.tag} — {a.what}.\n", c, count=1)
    c = re.sub(r"^E=\S+; CFG=\S+", f"E={a.exp}; CFG={out_cfg}", c, flags=re.M)
    c = re.sub(r"/dev/shm/joint4dec_\w+_local\.yaml", f"/dev/shm/joint4dec_{a.tag}_local.yaml", c)
    if mp_over:
        # drop any earlier override of the same key inherited from the base chain (last one wins anyway)
        for k in mp_over:
            c = re.sub(rf"--model\.model_params\.{re.escape(k)} \S+ ", "", c)
        mp = " ".join(f"--model.model_params.{k} {v}" for k, v in mp_over.items())
        c = c.replace("--model.model_params.compile_model false --data.lmdb_path /dev/shm/data.lmdb",
                      f"{mp} --model.model_params.compile_model false --data.lmdb_path /dev/shm/data.lmdb")
        c = c.replace('DEC="', f'DEC="{mp} ')
    out_chain = f"run_joint4dec_{a.tag}_chain.sh"
    open(out_chain, "w").write(c)

    os.makedirs(f"experiments/{a.exp}", exist_ok=True)
    notes = f"experiments/{a.exp}/notes.md"
    if not os.path.exists(notes):
        open(notes, "w").write(
            f"# {a.exp} — joint decoder arm `{a.tag}`\n\n**Goal.** {a.what}\n\n"
            f"**Setup.** Derived from `{a.base}` (exp `{base_exp}`) with overrides `{' '.join(a.set + a.profile) or 'none'}`; "
            f"`{out_cfg}`, `{out_chain}`, pod jdec-{a.tag}.\n\n**Result.** (pending)\n"
        )
    print(out_cfg, out_chain, notes)


if __name__ == "__main__":
    main()
