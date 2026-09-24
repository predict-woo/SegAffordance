"""Add screw_cap to an already-patched USDNet checkout (idempotent)."""
import pathlib
crit = pathlib.Path("models/criterion.py"); conf = pathlib.Path("conf/loss/set_criterion_articulation.yaml")
s = crit.read_text()
if "screw_cap" not in s:
    s = s.replace("screw_min_radius = 0.1, screw_term = 'h1',\n    ):", "screw_min_radius = 0.1, screw_term = 'h1', screw_cap = 0.0,\n    ):", 1)
    s = s.replace("        self.screw_min_radius, self.screw_term = float(screw_min_radius), screw_term\n",
                  "        self.screw_min_radius, self.screw_term = float(screw_min_radius), screw_term\n        self.screw_cap = float(screw_cap)\n", 1)
    old = "                                term = der[0] if self.screw_term == 'h1' else (pos[0] if self.screw_term == 'pos' else 0.5 * (pos[0] + der[0]))\n"
    assert old in s
    s = s.replace(old, old + "                                if self.screw_cap > 0: term = term.clamp(max=self.screw_cap)\n", 1)
    crit.write_text(s); print("cap added")
c = conf.read_text()
if "screw_cap" not in c:
    conf.write_text(c.rstrip("\n") + "\nscrew_cap: 0.0\n"); print("conf cap added")
