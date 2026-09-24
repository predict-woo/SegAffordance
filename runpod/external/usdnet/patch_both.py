"""Make screw_term='both' = pos + der (each capped independently when screw_cap>0).
Idempotent; applies on top of apply_patch.py + patch_cap.py. Run from the USDNet root."""
import pathlib
crit = pathlib.Path("models/criterion.py"); s = crit.read_text()
old = "                                term = der[0] if self.screw_term == 'h1' else (pos[0] if self.screw_term == 'pos' else 0.5 * (pos[0] + der[0]))\n                                if self.screw_cap > 0: term = term.clamp(max=self.screw_cap)\n"
new = ("                                capf = (lambda v: v.clamp(max=self.screw_cap)) if self.screw_cap > 0 else (lambda v: v)\n"
       "                                term = capf(der[0]) if self.screw_term == 'h1' else (capf(pos[0]) if self.screw_term == 'pos' else capf(pos[0]) + capf(der[0]))\n")
if "capf = " in s:
    print("already patched")
else:
    assert old in s, "anchor not found"
    crit.write_text(s.replace(old, new, 1)); print("patched: both = pos + der, per-term cap")
