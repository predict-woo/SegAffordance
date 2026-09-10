"""Numerical verification for knowledge/2026-09-11_revolute_loss_with_axis_term.md

Every closed form claimed in the note is checked here against Simpson-rule
quadrature (N = 201 samples) of the sampled (curve-space) definition, on
random articulations including flipped axes, small levers and large
along-axis offsets.  Also: flipped-axis floors (min over origin/point at
n = -n*), and Riemannian gradient / Hessian of each loss w.r.t. the axis.

Run:  python3 revolute_loss_check.py
"""
import numpy as np

rng = np.random.default_rng(0)
N = 201  # Simpson needs odd


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def unit(v):
    return v / np.linalg.norm(v)


def simpson(f, a, b, n=N):
    x = np.linspace(a, b, n)
    y = f(x)
    h = (b - a) / (n - 1)
    w = np.ones(n); w[1:-1:2] = 4; w[2:-1:2] = 2
    return h / 3 * np.einsum("i,i...->...", w, y)


def lever(n, p0, o):
    v = p0 - o
    r = v - (v @ n) * n
    return r, np.cross(n, r)


def gram(Th):
    A = 1.5 * Th + np.sin(2 * Th) / 4 - 2 * np.sin(Th)      # int (cos-1)^2
    C = 0.5 * Th - np.sin(2 * Th) / 4                       # int sin^2
    B2 = np.sin(Th) ** 2 - 2 * (1 - np.cos(Th))             # 2 int (cos-1) sin
    C2 = 0.5 * Th + np.sin(2 * Th) / 4                      # int cos^2
    I1 = Th - np.sin(Th)                                    # int (1-cos)
    return A, C, B2, C2, I1


def rodrigues(n, th):
    K = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def frame_M(n, r, ng, rg):
    """M_ij = (GT frame col i) . (pred frame col j), cols = (r_hat, t_hat)."""
    rh, th_ = unit(r), unit(np.cross(n, r))
    rgh, tgh = unit(rg), unit(np.cross(ng, rg))
    return rgh @ rh, rgh @ th_, tgh @ rh, tgh @ th_     # M11 M12 M21 M22


# ----------------------------------------------------------------------------
# sampled (curve-space) definitions
# ----------------------------------------------------------------------------
def curve(n, r, t, th):          # relative position, (len(th), 3)
    return (np.cos(th) - 1)[:, None] * r + np.sin(th)[:, None] * t


def dcurve(n, r, t, th):         # d/dtheta
    return -np.sin(th)[:, None] * r + np.cos(th)[:, None] * t


def sampled_pos(P, G, Th, norm):
    n, r, t = P; ng, rg, tg = G
    num = simpson(lambda th: np.sum((curve(n, r, t, th) - curve(ng, rg, tg, th)) ** 2, -1), 0, Th)
    return num / norm


def sampled_der(P, G, Th, norm):
    n, r, t = P; ng, rg, tg = G
    num = simpson(lambda th: np.sum((dcurve(n, r, t, th) - dcurve(ng, rg, tg, th)) ** 2, -1), 0, Th)
    return num / norm


def sampled_unit_tangent(P, G, Th):
    n, r, t = P; ng, rg, tg = G
    R, Rg = np.linalg.norm(r), np.linalg.norm(rg)
    return simpson(lambda th: np.sum((dcurve(n, r, t, th) / R - dcurve(ng, rg, tg, th) / Rg) ** 2, -1), 0, Th) / Th


def sampled_frenet(P, G, Th):
    n, r, t = P; ng, rg, tg = G
    R, Rg = np.linalg.norm(r), np.linalg.norm(rg)
    def f(th):
        T = dcurve(n, r, t, th) / R; Tg = dcurve(ng, rg, tg, th) / Rg
        Nn = -(np.cos(th)[:, None] * r + np.sin(th)[:, None] * t) / R
        Ng = -(np.cos(th)[:, None] * rg + np.sin(th)[:, None] * tg) / Rg
        B = np.cross(T, Nn); Bg = np.cross(Tg, Ng)
        return np.sum((T - Tg) ** 2 + (Nn - Ng) ** 2 + (B - Bg) ** 2, -1)
    return simpson(f, 0, Th) / Th


def sampled_so3(n, ng, Th):
    def f(th):
        return np.array([np.sum((rodrigues(n, x) - rodrigues(ng, x)) ** 2) for x in th])
    return simpson(f, 0, Th)


# ----------------------------------------------------------------------------
# closed forms (the claims)
# ----------------------------------------------------------------------------
def cf_pos_gt(P, G, Th):                       # current code, position
    A, C, B2, _, _ = gram(Th)
    dr, dt = P[1] - G[1], P[2] - G[2]
    return (A * dr @ dr + C * dt @ dt + B2 * dr @ dt) / ((A + C) * G[1] @ G[1])


def cf_der_gt(P, G, Th):                       # current code, derivative
    _, C, _, C2, _ = gram(Th)
    dr, dt = P[1] - G[1], P[2] - G[2]
    return (C * dr @ dr + C2 * dt @ dt - np.sin(Th) ** 2 * dr @ dt) / (Th * G[1] @ G[1])


def phase_terms(P, G, Th):
    """Returns lam, k, S=(M11+M22), D=(M11-M22), X=(M12+M21)."""
    M11, M12, M21, M22 = frame_M(P[0], P[1], G[0], G[1])
    lam = np.linalg.norm(P[1]) / np.linalg.norm(G[1])
    return lam, P[0] @ G[0], M11 + M22, M11 - M22, M12 + M21


def cf_der_gm(P, G, Th):                       # H1 / (Th |r||r*|)
    lam, k, S, D, X = phase_terms(P, G, Th)
    return (lam + 1 / lam) - S + np.sin(Th) / Th * (np.cos(Th) * D + np.sin(Th) * X)


def cf_der_gt_phase(P, G, Th):                 # H1 / (Th |r*|^2) in phase form
    lam, k, S, D, X = phase_terms(P, G, Th)
    return (lam ** 2 + 1) - lam * S + lam * np.sin(Th) / Th * (np.cos(Th) * D + np.sin(Th) * X)


def cf_pos_gt_phase(P, G, Th):                 # L2/((A+C)|r*|^2) in master-formula form
    A, C, B2, _, _ = gram(Th)
    lam, k, S, D, X = phase_terms(P, G, Th)
    rho = np.hypot(A - C, B2) / (A + C); chi0 = np.arctan2(B2, A - C)
    cos_chi_m = (D * np.cos(chi0) + X * np.sin(chi0))          # (1-k) cos(chi - chi0)
    cos_psi_m = S                                               # (1+k) cos psi
    return (lam - 1) ** 2 + lam * ((1 - k) - rho * cos_chi_m) + lam * ((1 + k) - cos_psi_m)


def cf_pos_gm(P, G, Th):                       # L2 / ((A+C)|r||r*|)
    A, C, B2, _, _ = gram(Th)
    lam, k, S, D, X = phase_terms(P, G, Th)
    return (lam + 1 / lam) - S - ((A - C) * D + B2 * X) / (A + C)


def cf_unit_tangent(P, G, Th):
    lam, k, S, D, X = phase_terms(P, G, Th)
    return 2 - S + np.sin(Th) / Th * (np.cos(Th) * D + np.sin(Th) * X)


def cf_frenet(P, G, Th):
    lam, k, S, D, X = phase_terms(P, G, Th)
    return 6 - 2 * (S + k)                     # = 4(1-k) + 2(1+k)(1-cos psi)


def cf_so3(n, ng, Th):
    A, _, _, _, I1 = gram(Th)
    x = 1 - n @ ng
    return 8 * I1 * x - 2 * A * x ** 2


# ----------------------------------------------------------------------------
# 1. closed-form verification on random articulations
# ----------------------------------------------------------------------------
def random_case(kind):
    ng = unit(rng.normal(size=3)); og = rng.normal(size=3)
    # GT point: lever radius + along-axis offset
    m = unit(np.cross(ng, rng.normal(size=3)))
    Rg = {"small": 10 ** rng.uniform(-3, -2)}.get(kind, 10 ** rng.uniform(-1, 0.5))
    ag = rng.normal() * (3.0 if kind == "along" else 0.3)
    pg = og + Rg * m + ag * ng
    if kind == "flip":
        n = -ng
    elif kind == "near":
        n = unit(ng + 0.05 * rng.normal(size=3))
    else:
        n = unit(rng.normal(size=3))
    o = og + 0.3 * rng.normal(size=3)
    p = pg + 0.3 * rng.normal(size=3)
    if kind == "flip" and rng.random() < 0.5:
        o, p = og.copy(), pg.copy()   # exact origin/point, flipped axis
    rg, tg = lever(ng, pg, og); r, t = lever(n, p, o)
    return (n, r, t), (ng, rg, tg)


print("=" * 78)
print("1. closed forms vs Simpson quadrature (N=201): max |closed - sampled|")
print("=" * 78)
sweeps = {"pi/2": np.pi / 2, "pi": np.pi, "2pi": 2 * np.pi, "0.7": 0.7}
kinds = ["generic"] * 40 + ["flip"] * 20 + ["small"] * 15 + ["along"] * 15 + ["near"] * 10
cases = [random_case(k) for k in kinds]
checks = {
    "pos  /|r*|^2  (current)": (cf_pos_gt, lambda P, G, Th: sampled_pos(P, G, Th, (gram(Th)[0] + gram(Th)[1]) * G[1] @ G[1])),
    "H1   /|r*|^2  (current)": (cf_der_gt, lambda P, G, Th: sampled_der(P, G, Th, Th * G[1] @ G[1])),
    "H1   /|r*|^2  phase form": (cf_der_gt_phase, lambda P, G, Th: sampled_der(P, G, Th, Th * G[1] @ G[1])),
    "pos  /|r*|^2  master form": (cf_pos_gt_phase, lambda P, G, Th: sampled_pos(P, G, Th, (gram(Th)[0] + gram(Th)[1]) * G[1] @ G[1])),
    "H1   /|r||r*| (geo-mean)": (cf_der_gm, lambda P, G, Th: sampled_der(P, G, Th, Th * np.linalg.norm(P[1]) * np.linalg.norm(G[1]))),
    "pos  /|r||r*| (geo-mean)": (cf_pos_gm, lambda P, G, Th: sampled_pos(P, G, Th, (gram(Th)[0] + gram(Th)[1]) * np.linalg.norm(P[1]) * np.linalg.norm(G[1]))),
    "unit tangent  /Th": (cf_unit_tangent, sampled_unit_tangent),
    "Frenet frame  /Th": (cf_frenet, sampled_frenet),
    "SO(3) Frobenius": (lambda P, G, Th: cf_so3(P[0], G[0], Th), lambda P, G, Th: sampled_so3(P[0], G[0], Th)),
}
print(f"{'loss':28s}" + "".join(f"{s:>12s}" for s in sweeps))
for name, (cf, sm) in checks.items():
    row = []
    for Th in sweeps.values():
        err = max(abs(cf(P, G, Th) - sm(P, G, Th)) for P, G in cases)
        row.append(err)
    print(f"{name:28s}" + "".join(f"{e:12.2e}" for e in row))

# phase-optimised full-circle unit tangent -> 1 - k
print("\nphase-optimised 2pi unit-tangent loss vs (1 - n.n*):")
errs = []
for P, G in cases[:30]:
    n, r, t = P
    best = np.inf
    for b in np.linspace(0, 2 * np.pi, 720, endpoint=False):
        rb = np.cos(b) * r + np.sin(b) * np.cross(n, r)
        best = min(best, cf_unit_tangent((n, rb, np.cross(n, rb)), G, 2 * np.pi))
    errs.append(abs(best - (1 - n @ G[0])))
print(f"   max |min_phase L_ut(2pi) - (1-k)| = {max(errs):.2e}  (grid 0.5 deg -> O(1e-5) expected)")

print("\nposition-term phase-leak amplitude rho_Theta = |(A-C, B2)|/(A+C)  vs  H1's sin(Theta)/Theta:")
for s_, Th in sweeps.items():
    A, C, B2, _, _ = gram(Th)
    print(f"   Theta={s_:5s}: rho_L2={np.hypot(A-C, B2)/(A+C):.3f}   rho_H1={abs(np.sin(Th))/Th:.3f}")

# sweep-independence of Frenet frame loss
print("\nFrenet-frame loss sweep independence: ",
      max(abs(sampled_frenet(P, G, 0.3) - sampled_frenet(P, G, 5.0)) for P, G in cases[:20]))

# SO(3) monotonicity in axis angle
print("\nSO(3) Frobenius loss g(x)=8 I1 x - 2 A x^2, x = 1-cos: g'(2) (sign says monotone to antipode):")
for s, Th in sweeps.items():
    A, _, _, _, I1 = gram(Th)
    print(f"   Theta={s:5s}: g'(0)={8*I1:7.3f}  g'(2)={8*I1-8*A:7.3f}  peak at k={1-2*I1/A if 8*I1-8*A<0 else -1:.3f}")

# ----------------------------------------------------------------------------
# 2. flipped-axis floors: min over (o, p0) of each loss at n = -n*
# ----------------------------------------------------------------------------
def nelder_mead(f, x0, iters=4000, step=0.3):
    d = len(x0)
    simplex = [np.array(x0, float)]
    for i in range(d):
        x = np.array(x0, float); x[i] += step; simplex.append(x)
    vals = [f(x) for x in simplex]
    for _ in range(iters):
        order = np.argsort(vals); simplex = [simplex[i] for i in order]; vals = [vals[i] for i in order]
        c = np.mean(simplex[:-1], 0)
        xr = c + (c - simplex[-1]); fr = f(xr)
        if fr < vals[0]:
            xe = c + 2 * (c - simplex[-1]); fe = f(xe)
            if fe < fr: simplex[-1], vals[-1] = xe, fe
            else: simplex[-1], vals[-1] = xr, fr
        elif fr < vals[-2]:
            simplex[-1], vals[-1] = xr, fr
        else:
            xc = c + 0.5 * (simplex[-1] - c); fc = f(xc)
            if fc < vals[-1]: simplex[-1], vals[-1] = xc, fc
            else:
                for i in range(1, d + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0]); vals[i] = f(simplex[i])
    i = int(np.argmin(vals)); return simplex[i], vals[i]


ng = np.array([0., 0., 1.]); og = np.zeros(3); pg = np.array([1., 0., 0.0])
G = lever(ng, pg, og); G = (ng, G[0], G[1])


def make_loss(name, Th):
    def L(n, o, p):
        r, t = lever(n, p, o)
        if np.linalg.norm(r) < 1e-9: r = r + 1e-9 * np.array([1, 0, 0]); t = np.cross(n, r)
        P = (n, r, t)
        if name == "pos_gt": return cf_pos_gt(P, G, Th)
        if name == "der_gt": return cf_der_gt(P, G, Th)
        if name == "der_gm": return cf_der_gm(P, G, Th)
        if name == "pos_gm": return cf_pos_gm(P, G, Th)
        if name == "ut": return cf_unit_tangent(P, G, Th)
        if name == "frenet": return cf_frenet(P, G, Th)
        if name == "so3": return cf_so3(n, ng, Th) / cf_so3(-ng, ng, Th)   # normalised to 1 at antipode
        if name == "anchor": return 1 - n @ ng
        if name == "der_gm+anchor": return cf_der_gm(P, G, Th) + (1 - n @ ng)
    return L


table = [("pos_gt", "pi/2"), ("pos_gt", "2pi"), ("der_gt", "pi/2"), ("der_gt", "pi"),
         ("pos_gm", "pi/2"), ("pos_gm", "2pi"), ("der_gm", "pi/2"), ("der_gm", "pi"),
         ("ut", "pi/2"), ("ut", "pi"), ("frenet", "pi/2"), ("so3", "pi/2"), ("anchor", "pi/2")]
print("\n" + "=" * 78)
print("2. flipped axis n=-n*: value at exact (o,p0), and floor = min over (o,p0)")
print("=" * 78)
print(f"{'loss':10s}{'sweep':7s}{'L(exact o,p0)':>15s}{'floor':>9s}{'lam*':>7s}{'phase*':>8s}")
floors = {}
for name, s in table:
    Th = sweeps[s]; L = make_loss(name, Th)
    f = lambda x: L(-ng, x[:3], x[3:])
    best = (None, np.inf)
    for trial in range(12):
        x0 = np.concatenate([og, pg]) + (0.6 * rng.normal(size=6) if trial else 0)
        x, v = nelder_mead(f, x0, iters=1500)
        if v < best[1]: best = (x, v)
    x, v = best
    r, t = lever(-ng, x[3:], x[:3]); lam = np.linalg.norm(r) / np.linalg.norm(G[1])
    ph = np.degrees(np.arctan2(unit(r) @ unit(G[2]), unit(r) @ unit(G[1])))
    floors[(name, s)] = v
    print(f"{name:10s}{s:7s}{L(-ng, og, pg):15.3f}{v:9.3f}{lam:7.2f}{ph:8.1f}")

# ----------------------------------------------------------------------------
# 3. axis gradient / curvature (finite differences on the sphere)
# ----------------------------------------------------------------------------
def riem_grad(L, n, h=1e-5):
    e1 = unit(np.cross(n, [0.3, 0.7, 0.1] if abs(n @ [0.3, 0.7, 0.1]) < 0.9 else [1, 0, 0])); e2 = np.cross(n, e1)
    g = np.array([(L(unit(n + h * e)) - L(unit(n - h * e))) / (2 * h) for e in (e1, e2)])
    H = np.zeros((2, 2))
    for i, a in enumerate((e1, e2)):
        for j, b in enumerate((e1, e2)):
            H[i, j] = (L(unit(n + h * a + h * b)) - L(unit(n + h * a - h * b)) - L(unit(n - h * a + h * b)) + L(unit(n - h * a - h * b))) / (4 * h * h)
    return np.linalg.norm(g), np.linalg.eigvalsh(H)


print("\n" + "=" * 78)
print("3. |grad_n L| at 90 deg axis error, at the antipode, and Hessian eigs at antipode")
print("   (o,p0 exact; then o,p0 at the compensated flipped state)")
print("=" * 78)
print(f"{'loss':10s}{'sweep':7s}{'|g| 90deg':>10s}{'|g| 170':>9s}{'|g| anti':>9s}{'Hess eig anti (exact)':>24s}{'|g| anti comp':>14s}{'Hess comp':>22s}")
n90 = np.array([0., 1., 0.]); n170 = unit(np.array([np.sin(np.radians(170)), 0, np.cos(np.radians(170))]))
for name, s in table:
    Th = sweeps[s]; L = make_loss(name, Th)
    Le = lambda n: L(n, og, pg)
    g90, _ = riem_grad(Le, n90); g170, _ = riem_grad(Le, n170); ga, Ha = riem_grad(Le, -ng, 1e-4)
    f = lambda x: L(-ng, x[:3], x[3:])
    x, _ = nelder_mead(f, np.concatenate([og, pg]), iters=2500)
    Lc = lambda n: L(n, x[:3], x[3:])
    gc, Hc = riem_grad(Lc, -ng, 1e-4)
    print(f"{name:10s}{s:7s}{g90:10.3f}{g170:9.3f}{ga:9.1e}   {Ha[0]:8.3f} {Ha[1]:8.3f}    {gc:10.1e}   {Hc[0]:8.3f} {Hc[1]:8.3f}")


def full_hessian_eigs(L, n, o, p, h=1e-4):
    e1 = unit(np.cross(n, [0.3, 0.7, 0.1])); e2 = np.cross(n, e1)
    def F(z):
        nn = unit(n + z[0] * e1 + z[1] * e2)
        return L(nn, o + z[2:5], p + z[5:8])
    H = np.zeros((8, 8)); z0 = np.zeros(8)
    for i in range(8):
        for j in range(8):
            a = np.zeros(8); a[i] = h; b = np.zeros(8); b[j] = h
            H[i, j] = (F(z0 + a + b) - F(z0 + a - b) - F(z0 - a + b) + F(z0 - a - b)) / (4 * h * h)
    return np.linalg.eigvalsh(H)

print("\nfull 8-D Hessian eigenvalues (n on sphere, o, p0) at the compensated flipped state:")
for name, s_ in [("pos_gt", "pi/2"), ("der_gt", "pi/2"), ("der_gt", "pi"), ("pos_gt", "2pi"), ("der_gm", "pi")]:
    Th = sweeps[s_]; L = make_loss(name, Th)
    f = lambda x: L(-ng, x[:3], x[3:])
    x, v = nelder_mead(f, np.concatenate([og, pg]), iters=3000)
    ev = full_hessian_eigs(L, -ng, x[:3], x[3:])
    print(f"   {name:7s} {s_:5s} value={v:.3f}  min eig={ev[0]:+.3f}  eigs={np.round(ev, 3)}")

# radius scaling of the axis gradient (predicted lever lam * GT lever), 90 deg error
print("\naxis gradient at 90 deg vs predicted lever scale lam (o exact, p0 scaled radially):")
print(f"{'lam':>6s}" + "".join(f"{n+'/'+s:>14s}" for n, s in [("pos_gt", "pi/2"), ("der_gt", "pi/2"), ("der_gm", "pi"), ("ut", "pi"), ("anchor", "pi/2")]))
for lam in [0.01, 0.1, 1.0, 10.0]:
    p = og + lam * (pg - og)
    row = []
    for name, s in [("pos_gt", "pi/2"), ("der_gt", "pi/2"), ("der_gm", "pi"), ("ut", "pi"), ("anchor", "pi/2")]:
        L = make_loss(name, sweeps[s]); row.append(riem_grad(lambda n: L(n, og, p), n90)[0])
    print(f"{lam:6.2f}" + "".join(f"{v:14.3f}" for v in row))

# along-axis offset: axis gradient of the current loss at the antipode with exact o,p0
print("\ncurrent losses, exact (o,p0), n=-n*: |grad_n| vs along-axis offset a = n*.(p0-o) (R*=1):")
for a in [0.0, 0.5, 2.0]:
    p = np.array([1., 0., a])
    Gp = lever(ng, p, og); Gp = (ng, Gp[0], Gp[1])
    out = []
    for cf, Th in [(cf_pos_gt, np.pi / 2), (cf_der_gt, np.pi / 2), (cf_der_gm, np.pi)]:
        def Ln(n):
            r, t = lever(n, p, og); return cf((n, r, t), Gp, Th)
        out.append(riem_grad(Ln, -ng)[0])
    print(f"   a={a:4.1f}: pos_gt(pi/2)={out[0]:.3f}  der_gt(pi/2)={out[1]:.3f}  der_gm(pi)={out[2]:.3f}")

# ----------------------------------------------------------------------------
# 4. the exact split of the geo-mean H1 at Theta = pi
# ----------------------------------------------------------------------------
print("\n" + "=" * 78)
print("4. der_gm at Theta=pi == (lam+1/lam-2) + (1-k) + (1+k)(1-cos psi): max err over cases")
print("=" * 78)
errs = []
for P, G2 in cases:
    lam, k, S, D, X = phase_terms(P, G2, np.pi)
    M11, M12, M21, M22 = frame_M(P[0], P[1], G2[0], G2[1])
    psi = np.arctan2(M21 - M12, M11 + M22)
    split = (lam + 1 / lam - 2) + (1 - k) + (1 + k) * (1 - np.cos(psi))
    errs.append(abs(split - cf_der_gm(P, G2, np.pi)))
print("   max err:", max(errs))
