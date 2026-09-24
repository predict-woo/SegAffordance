"""Screw-loss slot-in for Particulate (particulate/models.py forward + train.py
weights). Idempotent. Adds two loss keys, both weight 0 by default, so the
'theirs' arm is bit-identical to upstream:
  screw_h1_loss     : per revolute point, closed-form H1 quadratic with
                      p = point, q = predicted foot point x~_j, n = predicted
                      part direction; GT from gt_closest_point_on_axis + plucker.
  screw_anchor_loss : per revolute part, 1 - cos(d~, d*) (sign-sensitive).
  (prismatic: screw_prismatic_loss = |d^ - d^*|^2 = 2(1-cos), per part)
Ours arm: axis_revolute L1 -> 0, axis_prismatic L1 -> 0, screw_* on; the per-point
foot L1 (their absolute origin term) is KEPT."""
import sys, pathlib
R = pathlib.Path(".")
m = R / "particulate/models.py"; t = R / "train.py"
s = m.read_text()
if "SCREW_PATCH" not in s:
    s = s.replace("import torch\n", "import torch\nfrom screw_loss import closed_form_screw_loss  # SCREW_PATCH\n", 1)
    old = "        # Combine all losses\n        losses = dict(\n"
    new = '''        # SCREW_PATCH: closed-form screw terms (weights live in train.py config)
        screw_h1_loss = self.parameters().__next__().sum() * 0
        screw_anchor_loss = self.parameters().__next__().sum() * 0
        screw_prismatic_loss = self.parameters().__next__().sum() * 0
        if forward_motion_params and closest_point_on_axis is not None and gt_closest_point_on_axis is not None \\
                and revolute_plucker is not None and gt_revolute_plucker is not None:
            B, N = part_ids.shape
            valid_parts = torch.arange(revolute_plucker.shape[1], device=xyz.device)[None] < num_valid_parts[:, None]
            rev_parts = valid_parts & torch.any(gt_revolute_plucker[..., :3] != 0, dim=-1)
            d_p = revolute_plucker[..., :3]; d_g = gt_revolute_plucker[..., :3]
            if rev_parts.any():
                cos = torch.nn.functional.cosine_similarity(d_p[rev_parts].float(), d_g[rev_parts].float(), dim=-1, eps=1e-8)
                screw_anchor_loss = (1.0 - cos).mean()
                # per-point: gather the part direction for every point
                d_p_pt = torch.gather(d_p, 1, part_ids.unsqueeze(-1).expand(-1, -1, 3))
                d_g_pt = torch.gather(d_g, 1, part_ids.unsqueeze(-1).expand(-1, -1, 3))
                rev_pt = torch.gather(rev_parts, 1, part_ids)
                if rev_pt.any():
                    p = xyz[rev_pt].float(); q = closest_point_on_axis[rev_pt].float(); qg = gt_closest_point_on_axis[rev_pt].float()
                    n = d_p_pt[rev_pt].float(); ng = d_g_pt[rev_pt].float()
                    lever = (p - qg) - ((p - qg) * torch.nn.functional.normalize(ng, dim=-1)).sum(-1, keepdim=True) * torch.nn.functional.normalize(ng, dim=-1)
                    ok = lever.norm(dim=-1) >= float(getattr(self, "screw_min_radius", 0.02))
                    if ok.any():
                        pos, der = closed_form_screw_loss(torch.ones(ok.sum(), device=xyz.device), n[ok], n[ok], q[ok], p[ok], ng[ok], qg[ok], p[ok])
                        screw_h1_loss = der.mean()
            pri_parts = valid_parts & torch.any(gt_prismatic_axis[..., :3] != 0, dim=-1) if gt_prismatic_axis is not None else None
            if pri_parts is not None and pri_parts.any():
                a = torch.nn.functional.normalize(prismatic_axis[pri_parts].float(), dim=-1)
                b = torch.nn.functional.normalize(gt_prismatic_axis[pri_parts][..., :3].float(), dim=-1)
                screw_prismatic_loss = (a - b).pow(2).sum(-1).mean()

        # Combine all losses
        losses = dict(
            screw_h1_loss=screw_h1_loss,
            screw_anchor_loss=screw_anchor_loss,
            screw_prismatic_loss=screw_prismatic_loss,
'''
    assert old in s; s = s.replace(old, new, 1)
    m.write_text(s); print("patched models.py")
else:
    print("models.py already patched")
ts = t.read_text()
if "SCREW_PATCH" not in ts:
    old = '        "point_closest_point_on_axis_loss": config.loss_weight_point_closest_point_on_axis,\n    }'
    new = ('        "point_closest_point_on_axis_loss": config.loss_weight_point_closest_point_on_axis,\n'
           '        "screw_h1_loss": config.get("loss_weight_screw_h1", 0.0),  # SCREW_PATCH\n'
           '        "screw_anchor_loss": config.get("loss_weight_screw_anchor", 0.0),\n'
           '        "screw_prismatic_loss": config.get("loss_weight_screw_prismatic", 0.0),\n    }')
    assert old in ts; ts = ts.replace(old, new, 1)
    # weights-only init from the released model.pt
    old = "    model = build_model(config)\n"
    new = ('    model = build_model(config)\n'
           '    if config.get("init_weights"):  # SCREW_PATCH\n'
           '        sd = torch.load(config.init_weights, map_location="cpu")\n'
           '        sd = sd.get("state_dict", sd.get("model", sd))\n'
           '        missing, unexpected = model.load_state_dict(sd, strict=False)\n'
           '        print(f"[init_weights] loaded; missing={len(missing)} unexpected={len(unexpected)}")\n'
           '        assert len(unexpected) == 0, unexpected[:5]\n')
    assert old in ts; ts = ts.replace(old, new, 1)
    t.write_text(ts); print("patched train.py")
else:
    print("train.py already patched")
