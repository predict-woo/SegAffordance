"""Apply the screw-loss slot-in to USDNet (models/criterion.py + loss conf).
Idempotent: skips if the marker is present. Run from the USDNet repo root.
Arm 'theirs' = loss.screw_weight=0 -> code path identical to upstream."""
import re, sys, pathlib
R = pathlib.Path(".")
crit = R / "models/criterion.py"; conf = R / "conf/loss/set_criterion_articulation.yaml"
s = crit.read_text()
if "SCREW_PATCH" in s:
    print("criterion already patched"); sys.exit(0)

# 1. import
s = s.replace("import torch\n", "import torch\nfrom screw_loss import closed_form_screw_loss  # SCREW_PATCH\n", 1)

# 2. constructor args + attrs
old = "        regular_arti_loss = True,\n        use_mov_mask_for_interaction_loss = False,\n    ):"
new = ("        regular_arti_loss = True,\n        use_mov_mask_for_interaction_loss = False,\n"
       "        screw_weight = 0.0, screw_mode = 'replace', screw_min_radius = 0.1, screw_term = 'h1', screw_cap = 0.0,\n    ):")
assert old in s; s = s.replace(old, new, 1)
old = "        self.regular_arti_loss = regular_arti_loss\n"
new = (old + "        self.screw_weight, self.screw_mode = float(screw_weight), screw_mode\n"
       "        self.screw_min_radius, self.screw_term = float(screw_min_radius), screw_term\n"
       "        self.screw_cap = float(screw_cap)\n"
       "        self._screw_acc = [0.0, 0.0, 0.0, 0]  # sum_screw, sum_theirs, sum_axis1cos, n\n"
       "        print('SCREW_PATCH: weight', self.screw_weight, 'mode', self.screw_mode, 'min_radius', self.screw_min_radius, 'term', self.screw_term)\n")
assert old in s; s = s.replace(old, new, 1)

# 3. rotation branch
old = "                        loss_translations_item.append(dist_gt_axis + dist_pred_axis)\n"
new = '''                        theirs_item = dist_gt_axis + dist_pred_axis
                        if self.screw_weight > 0 and 'mov_parts_centers' in targets[batch_id]:
                            p_star = targets[batch_id]['mov_parts_centers'][target_id].to(pred_origin).float()
                            n_gt = F.normalize(target_axis.float(), dim=0)
                            rel = p_star - target_origin.float()
                            lever = rel - (rel * n_gt).sum() * n_gt
                            if lever.norm() >= self.screw_min_radius:
                                pos, der = closed_form_screw_loss(
                                    torch.ones(1, device=pred_origin.device), pred_axis[None], pred_axis[None],
                                    pred_origin[None], p_star[None], target_axis[None], target_origin[None], p_star[None])
                                term = der[0] if self.screw_term == 'h1' else (pos[0] if self.screw_term == 'pos' else 0.5 * (pos[0] + der[0]))
                                if self.screw_cap > 0: term = term.clamp(max=self.screw_cap)
                                item = self.screw_weight * term + (theirs_item if self.screw_mode == 'add' else 0.0)
                                a = self._screw_acc; a[0] += float(term); a[1] += float(theirs_item); a[3] += 1
                                if a[3] % 500 == 0:
                                    print(f"SCREW_STATS n={a[3]} mean_screw_term={a[0]/a[3]:.4f} mean_theirs_linedist={a[1]/a[3]:.4f}")
                            else:
                                item = theirs_item
                            loss_translations_item.append(item)
                        else:
                            loss_translations_item.append(theirs_item)
'''
assert old in s; s = s.replace(old, new, 1)
crit.write_text(s)

c = conf.read_text()
if "screw_weight" not in c:
    c = c.rstrip("\n") + "\nscrew_weight: 0.0\nscrew_mode: replace\nscrew_min_radius: 0.1\nscrew_term: h1\nscrew_cap: 0.0\n"
    conf.write_text(c)
print("patched criterion.py and loss conf")
