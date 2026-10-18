"""Numerical checks for "Ghosts in Poisson brane gases".

Run with  python3 verify_ghosts.py .  Every check prints PASS or FAIL and the
script exits with status 1 if any check fails.  The checks re-derive, by
independent means, the identities and estimates that the paper proves by
hand; no statement of the paper depends on them.
"""

import math
import sys

import mpmath as mp
import numpy as np

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print(("PASS " if ok else "FAIL ") + name + ("  " + detail if detail else ""))


# ---------------------------------------------------------------------------
# A. The averaged propagator of a truncated scale-free gas (Theorem 1).
#    For intensity theta dt/t on [t0, T] the exact average of 1/F is
#    ((1 + z/t0)/(1 + z/T))^(-theta).
# ---------------------------------------------------------------------------
def check_average():
    rng = np.random.default_rng(20261018)
    theta, t0, T = 0.7, 1.0, 1.0e4
    M = 200000
    L = math.log(T / t0)
    counts = rng.poisson(theta * L, size=M)
    pts = t0 * np.exp(rng.uniform(0.0, L, size=counts.sum()))
    owner = np.repeat(np.arange(M), counts)
    for z in [2.0, 10.0 + 5.0j, 3.0j, -0.5 + 1.0j]:
        logs = np.zeros(M, dtype=complex)
        np.add.at(logs, owner, np.log(1.0 + z / pts))
        vals = np.exp(-logs)
        mean = vals.mean()
        se = vals.std() / math.sqrt(M)
        exact = ((1 + z / t0) / (1 + z / T)) ** (-theta)
        err = abs(mean - exact)
        check("average of 1/F at z=%s" % z, err < 5 * se,
              "MC %.5f%+.5fi exact %.5f%+.5fi err/se %.2f"
              % (mean.real, mean.imag, exact.real, exact.imag, err / se))


# ---------------------------------------------------------------------------
# B. Residues of a frozen configuration (Theorem 5): signs, sum rule,
#    monotone truncation.
# ---------------------------------------------------------------------------
def residues(ts):
    out = []
    for k, tk in enumerate(ts):
        p = mp.mpf(1)
        for j, tj in enumerate(ts):
            if j != k:
                p *= 1 - tk / tj
        out.append(-1 / p)
    return out


def check_residues():
    mp.mp.dps = 80
    rng = np.random.default_rng(7)
    theta, t0 = 1.0, 1.0
    u = np.cumsum(rng.exponential(1.0 / theta, size=60))
    ts = [mp.mpf(t0) * mp.e ** mp.mpf(float(x)) for x in u]
    r40 = residues(ts[:40])
    r60 = residues(ts[:60])
    signs = all((r < 0) == (k % 2 == 0) for k, r in enumerate(r60))
    check("residues alternate, lightest is a ghost", signs)
    s = mp.fsum(r60)
    check("sum rule for 60 masses", abs(s + 1) < mp.mpf(10) ** -60,
          "sum + 1 = %s" % mp.nstr(s + 1, 3))
    mono = all(abs(r60[k]) >= abs(r40[k]) and (r60[k] < 0) == (r40[k] < 0)
               for k in range(40))
    check("|r_k^(N)| increases with N, sign fixed", mono)


# ---------------------------------------------------------------------------
# C. Gaussian decay of residues for the scale-free gas (Theorem 6):
#    log|r_k| = -S_k + R_k with S_k ~ k^2/(2 theta) and 0 <= R_k = O(log^2 k).
# ---------------------------------------------------------------------------
def log_residue_parts(u, k):
    uk = u[k]
    S = float(np.sum(uk - u[:k]))
    d = np.abs(np.delete(u, k) - uk)
    R = float(-np.sum(np.log1p(-np.exp(-d))))
    return S, R


def check_gaussian_decay():
    rng = np.random.default_rng(11)
    theta = 1.3
    K, extra = 400, 200
    worst = 0.0
    ratios = []
    for _ in range(40):
        u = np.cumsum(rng.exponential(1.0 / theta, size=K + extra))
        for k in (50, 100, 200, 399):
            S, R = log_residue_parts(u, k)
            worst = max(worst, R / math.log(k + 2) ** 2)
            if R < 0:
                worst = float("inf")
        S, _ = log_residue_parts(u, 399)
        ratios.append(S / (399 ** 2 / (2 * theta)))
    check("0 <= R_k <= C log^2 k", worst < 3.0, "max R_k/log^2 k = %.3f" % worst)
    m = float(np.mean(ratios))
    check("S_k / (k^2/2theta) near 1 at k=399", abs(m - 1) < 0.02, "mean %.4f" % m)


# ---------------------------------------------------------------------------
# D. Boundary values for lambda(t) = A exp(-log^2 t) (Theorem 3).
# ---------------------------------------------------------------------------
A = mp.mpf("1.5")


def lam(t):
    return A * mp.e ** (-mp.log(t) ** 2)


def phi(z, lamfun=None, u0=None):
    lamfun = lamfun or (lambda u: A * mp.e ** (-u ** 2))
    f = lambda u: z / (z + mp.e ** u) * lamfun(u)
    pts = [-12, -4, -1, 0, 1, 4, 12]
    if u0 is not None:
        pts = sorted(set(pts + [u0 + d for d in
                                (-1e-2, -1e-4, -1e-6, 0, 1e-6, 1e-4, 1e-2)]))
    return mp.quad(f, pts)


def check_boundary():
    mp.mp.dps = 30
    ok = True
    worst = 0
    for s in [0.3, 0.8, 1.0, 1.5, 3.0]:
        s = mp.mpf(s)
        eps = mp.mpf("1e-9")
        p = phi(-s + 1j * eps * s, u0=mp.log(s))
        rho = -mp.im(mp.e ** (-p) / (-s + 1j * eps * s)) / mp.pi
        formula = -mp.e ** (-mp.re(p)) * mp.sin(mp.pi * lam(s)) / (mp.pi * s)
        worst = max(worst, abs(mp.im(p) / mp.pi - lam(s)), abs(rho - formula))
    ok = worst < 1e-6
    check("Im phi(-s+i0) = pi lambda(s) and the density formula", ok,
          "max deviation %.2e" % float(worst))


# ---------------------------------------------------------------------------
# E, F. Scale-free averaged propagator (Corollary 4).
# ---------------------------------------------------------------------------
def check_scalefree():
    mp.mp.dps = 30
    t0 = mp.mpf(1)
    for theta in [mp.mpf("0.25"), mp.mpf("0.5"), mp.mpf("0.75")]:
        c = mp.sin(mp.pi * theta) / mp.pi
        # s = t0 (1 + e^w): sigma(ds) = c e^{(1-theta) w} / (1 + e^w) dw.
        w_dens = lambda w: c * mp.e ** ((1 - theta) * w) / (1 + mp.e ** w)
        total = mp.quad(w_dens, [-mp.inf, -5, 0, 5, mp.inf])
        x = mp.mpf(2)
        lhs = (1 + x / t0) ** (-theta) / x
        rhs = 1 / x - mp.quad(lambda w: w_dens(w) / (x + t0 * (1 + mp.e ** w)),
                              [-mp.inf, -5, 0, 5, mp.inf])
        check("theta=%s: ghost weight 1 and Stieltjes form" % theta,
              abs(total - 1) < 1e-20 and abs(lhs - rhs) < 1e-20,
              "total - 1 = %.1e, residual %.1e" % (float(total - 1), float(abs(lhs - rhs))))
    z = mp.mpc(3, 2)
    one = (1 + z / t0) ** (-1) / z - (1 / z - 1 / (z + t0))
    check("theta=1: one ghost of residue -1 at t0", abs(one) < 1e-25)


# ---------------------------------------------------------------------------
# G. lambda <= 1: pure ghost representation (Theorem 3(i)).
# ---------------------------------------------------------------------------
def check_pure_ghost():
    mp.mp.dps = 25
    B = mp.mpf("0.9")
    lamB = lambda u: B * mp.e ** (-u ** 2)
    nu = B * mp.sqrt(mp.pi)
    minim = mp.inf
    for x in [-3, -1, -0.2, 0.5, 2]:
        for y in [0.01, 0.3, 2]:
            z = mp.mpc(x, y)
            psi = 1 - mp.e ** (-phi(z, lamB))
            minim = min(minim, mp.im(psi))
    check("lambda <= 1: Im psi >= 0 on a grid", minim >= 0, "min %.3e" % float(minim))

    def sigma(s):
        s = mp.mpf(s)
        p = phi(-s + 1j * mp.mpf("1e-9") * s, lamB, u0=mp.log(s))
        return mp.e ** (-mp.re(p)) * mp.sin(mp.pi * lamB(mp.log(s))) / (mp.pi * s)

    tot = mp.quad(lambda u: sigma(mp.e ** u) * mp.e ** u, [-6, -2, 0, 2, 6])
    check("lambda <= 1: ghost weight 1 - e^{-nu}", abs(tot - (1 - mp.e ** (-nu))) < 1e-5,
          "%.7f vs %.7f" % (float(tot), float(1 - mp.e ** (-nu))))
    x = mp.mpf(1)
    lhs = mp.e ** (-phi(x, lamB)) / x
    rhs = 1 / x - mp.quad(lambda u: sigma(mp.e ** u) * mp.e ** u / (x + mp.e ** u),
                          [-6, -2, 0, 2, 6])
    check("lambda <= 1: Gbar = 1/z - int sigma/(z+s) at z=1", abs(lhs - rhs) < 1e-5,
          "residual %.1e" % float(abs(lhs - rhs)))


# ---------------------------------------------------------------------------
# H, I. The trade-off bound (Theorem 7) and Jensen's inequality.
# ---------------------------------------------------------------------------
def check_tradeoff():
    rng = np.random.default_rng(3)
    worst = -np.inf
    for _ in range(200):
        edges = np.sort(rng.uniform(-4, 6, size=8))
        vals = rng.uniform(0, 3, size=7)
        Lam = vals.max()
        a = math.exp(rng.uniform(-3, 5))
        x = math.exp(rng.uniform(-3, 8))
        # phi(x) = sum over pieces of int lambda x/((x+t)t) dt, exact.
        ph = 0.0
        nu_a = 0.0
        for (l, r), v in zip(zip(edges[:-1], edges[1:]), vals):
            tl, tr = math.exp(l), math.exp(r)
            ph += v * (math.log(tr / tl) - math.log((x + tr) / (x + tl)))
            lo, hi = tl, min(tr, a)
            if hi > lo:
                nu_a += v * math.log(hi / lo)
        bound = nu_a + Lam * math.log(1 + x / a)
        worst = max(worst, ph - bound)
    check("phi(x) <= nu(0,a] + sup(lambda) log(1+x/a)", worst <= 1e-12,
          "max excess %.2e" % worst)
    theta, t0 = 0.7, 1.0
    ok = all((1 + x / t0) ** (-theta) >= math.exp(-theta * x / t0)
             for x in np.exp(np.linspace(-5, 5, 41)))
    check("Jensen: E[1/F] >= 1/E[F] on the positive axis", ok)


def main():
    check_average()
    check_residues()
    check_gaussian_decay()
    check_boundary()
    check_scalefree()
    check_pure_ghost()
    check_tradeoff()
    n = len(RESULTS)
    f = n - sum(RESULTS)
    print("%d checks passed, %d failed" % (n - f, f))
    sys.exit(1 if f else 0)


if __name__ == "__main__":
    main()
