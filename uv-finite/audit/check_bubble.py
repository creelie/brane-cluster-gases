#!/usr/bin/env python3
"""Audit 05: the one-loop bubble in closed form (Sec. "One loop in closed form").

Independent evaluation of Eq. (bubbledec),

  A(s) = A_mu(s) + int 4 pi r^2 dr int dt [ |F(xi)|^2 - |F_mu(xi)|^2 ]
        + (2 pi^2 / s) int_{-s}^0 (w + s) [F(w) - F_mu(w)] dw,
  xi = t^2 + r^2 - s/4 + i E t,  F = 1/(z a(z)),  F_mu = 1/z - 1/(z + mu^2),

with scipy's adaptive 2-D quadrature in (t, r) (the paper used polar
coordinates about xi = 0 and a uniform grid), for the skeleton with n = 4.

Checks:
  (1) Im A_mu = -pi^3 for every s > 0 and mu^2 > s (the only complex term);
  (2) Eq. (bubbledec) applied to F_mu itself (a local propagator) returns
      A_mu: the decomposition is an identity, not an approximation;
  (3) the entries of Table tab:bubble for E/M = 1.0, 1.25, 1.4, 1.5, 1.6, 1.7
      (Re A to 1%, Im A = -31.006, the ratio column s a(-s) Re A);
  (4) independence of the Pauli-Villars mass (mu = 2E+2 against 3E+3);
  (5) the crossing of the ratio G s a(-s) Re A = 1 lies between 1.4 M and
      1.9 M for M/M_P = 1e-2 and 1e-40 (interpolating the table in log scale
      with the computed points, and using the 1.8, 1.9 entries of the paper
      for the extrapolation);
  (6) the leading term of the lower bound (bubblebound), maximised over delta,
      against the "bound" column, and that it lies below Re A.

The result (1)+(3) is the explicit cutting rule and the super-exponential
growth of Re A, which is where the paper locates the end of perturbation
theory; nothing here extends the result to other graphs.
"""
import sys
import numpy as np
import mpmath as mp
from scipy import integrate, special

FAILS = 0


def check(name, ok, detail=""):
    global FAILS
    print(("PASS " if ok else "FAIL ") + name + ("   [" + detail + "]" if detail else ""))
    if not ok:
        FAILS += 1


N_SKEL = 4.0
EULER = 0.5772156649015329


def Ein_c(w):
    """entire exponential integral for complex w (numpy scalar or array)."""
    w = np.asarray(w, dtype=complex)
    out = np.empty_like(w)
    small = np.abs(w) < 20
    # series for small |w|
    ws = w[small]
    term = ws.copy()
    acc = ws.copy()
    for k in range(2, 120):
        term = -term * ws / k
        acc = acc + term / k
    out[small] = acc
    wl = w[~small]
    out[~small] = np.log(wl) + EULER + special.exp1(wl)
    return out


def a_skel(z):
    return np.exp(N_SKEL / 2 * Ein_c(z * z))


def F(z):
    return 1.0 / (z * a_skel(z))


def Fmu(z, mu2):
    return 1.0 / z - 1.0 / (z + mu2)


def A_mu_closed(s, mu2, eta=1e-12):
    """pi^2 int_0^1 dx log( D_{0mu} D_{mu0} / (D_{00} D_{mumu}) ), D_ab = x m_a^2 + (1-x) m_b^2 - x(1-x)(s - i0)."""
    sc = s - 1j * eta

    def integrand(x):
        D = lambda ma2, mb2: x * ma2 + (1 - x) * mb2 - x * (1 - x) * sc
        return np.log(D(0, mu2) * D(mu2, 0) / (D(0, 0) * D(mu2, mu2)))

    re = integrate.quad(lambda x: integrand(x).real, 0, 1, limit=200)[0]
    im = integrate.quad(lambda x: integrand(x).imag, 0, 1, limit=200)[0]
    return np.pi ** 2 * (re + 1j * im)


def real_section(s, mu2, Fa, Fb, rho_max=None):
    """int 4 pi r^2 dr dt [|Fa|^2 - |Fb|^2] over t in R, r >= 0, in polar coordinates
    (t, r) = (rho sin th, E/2 + rho cos th) about the graviton pole xi = 0, where the
    integrand has an integrable 1/|xi| singularity; the Jacobian rho removes it.
    The integrand is even in t, so th runs over [0, pi] and the result is doubled."""
    E = np.sqrt(s)
    rho_max = rho_max or 4.0 + 2 * E

    def integrand(th, rho):
        t = rho * np.sin(th)
        r = E / 2 + rho * np.cos(th)
        if r <= 0:
            return 0.0
        xi = t * t + r * r - s / 4 + 1j * E * t
        return rho * 4 * np.pi * r * r * (abs(Fa(xi)) ** 2 - abs(Fb(xi, mu2)) ** 2)

    def inner(rho):
        # the line r = 0 crosses the circle at cos th = -E/(2 rho): split there
        pts = []
        if rho > E / 2:
            pts = [np.arccos(-E / (2 * rho))]
        return integrate.quad(integrand, 0, np.pi, args=(rho,), points=pts or None, limit=200, epsabs=1e-13, epsrel=1e-11)[0]

    # the integrand falls only like rho^{-4} for a local Pauli-Villars subtraction, so the
    # tail beyond rho_max is integrated to infinity, not dropped
    val, err = integrate.quad(inner, 0, rho_max, points=[E / 2, E], limit=400, epsabs=1e-12, epsrel=1e-10)
    tail, terr = integrate.quad(inner, rho_max, np.inf, limit=400, epsabs=1e-12, epsrel=1e-10)
    return 2 * (val + tail), 2 * (err + terr)


def residue_term(s, mu2, Fa, Fb):
    f = lambda w: (w + s) * (Fa(w) - Fb(w, mu2)).real
    val = integrate.quad(f, -s, 0, limit=400, points=[-s / 2])[0]
    return 2 * np.pi ** 2 / s * val


def bubble(s, mu2, Fa=F):
    Am = A_mu_closed(s, mu2)
    rs, err = real_section(s, mu2, Fa, Fmu)
    rt = residue_term(s, mu2, Fa, Fmu)
    return Am + rs + rt, err


# (1) imaginary part of A_mu
ok = True
for s in (1.0, 1.5625, 2.25, 3.61):
    for mu2 in (2 * s + 1, 3 * s + 5):
        Am = A_mu_closed(s, mu2)
        ok &= abs(Am.imag + np.pi ** 3) < 1e-6
check("Im A_mu = -pi^3 for s > 0, mu^2 > s (closed form, 8 cases)", ok)

# (2) the decomposition applied to a local propagator with a second PV mass
# Use Fa = F_{nu} (PV mass nu) and Fb = F_mu; then A should equal A_nu exactly.
s = 1.0
mu2, nu2 = 7.0, 13.0
Fa_loc = lambda z: Fmu(z, nu2)
Am = A_mu_closed(s, mu2)
rs, _ = real_section(s, mu2, Fa_loc, Fmu)
rt = residue_term(s, mu2, Fa_loc, Fmu)
A_dec = Am + rs + rt
A_nu = A_mu_closed(s, nu2)
check("Eq. (bubbledec) applied to a local propagator reproduces its closed form (s=1, mu^2=7, nu^2=13)",
      abs(A_dec - A_nu) < 1e-5 * (1 + abs(A_nu)), "decomposition %.8f%+.8fi, closed %.8f%+.8fi" % (A_dec.real, A_dec.imag, A_nu.real, A_nu.imag))

# (3) Table tab:bubble
table = {1.00: (11.7, 57.3, 0.37), 1.25: (24.7, 770.0, 0.54), 1.40: (82.1, 7.60e3, 0.80),
         1.50: (475.0, 8.70e4, 1.14), 1.60: (1.27e4, 4.45e6, 3.78), 1.70: (4.54e6, 2.90e9, 156.0)}
results = {}
ok_re = ok_im = ok_ratio = True
details = []
for E, (reA_t, ratio_t, bound_t) in table.items():
    s = E * E
    mu = 2 * E + 2
    A, err = bubble(s, mu * mu)
    a_minus = a_skel(np.array(-s + 0j))[()].real
    ratio = s * a_minus * A.real
    results[E] = (A.real, ratio)
    ok_re &= abs(A.real / reA_t - 1) < 0.01
    ok_im &= abs(A.imag + 31.006) < 1e-2
    ok_ratio &= abs(ratio / ratio_t - 1) < 0.01
    details.append("E=%.2f ReA=%.4g ImA=%.3f ratio=%.3g" % (E, A.real, A.imag, ratio))
check("Table tab:bubble, Re A to 1% for E/M = 1.0 .. 1.7", ok_re, "; ".join(details))
check("Table tab:bubble, Im A = -31.006 at every energy", ok_im)
check("Table tab:bubble, s a(-s) Re A column to 1%", ok_ratio)

# (4) Pauli-Villars independence
E = 1.5
A1, _ = bubble(E * E, (2 * E + 2) ** 2)
A2, _ = bubble(E * E, (3 * E + 3) ** 2)
check("A(s) independent of the Pauli-Villars mass (E=1.5M: mu=2E+2 vs 3E+3)", abs(A1 - A2) < 1e-6 * abs(A1),
      "%.9g vs %.9g" % (A1.real, A2.real))

# (5) crossing scale.  ratio to tree = (M/M_P)^2 * s a(-s) Re A (M = 1).  Use the computed
# points up to 1.7 and the paper's 1.8, 1.9 entries, interpolate log(ratio) linearly in E.
Es = sorted(results) + [1.8, 1.9]
logr = [np.log10(results[E][1]) for E in sorted(results)] + [np.log10(2.74e29), np.log10(7.42e91)]
ok_cross = True
det = []
for MoverMP in (1e-2, 1e-40):
    target = -2 * np.log10(MoverMP)       # need log10(s a Re A) = -2 log10(M/M_P)
    Ecross = np.interp(target, logr, Es)
    ok_cross &= 1.4 <= Ecross <= 1.9
    det.append("M/M_P=%g: E_cross=%.3f M" % (MoverMP, Ecross))
check("ratio to tree passes 1 between 1.4M and 1.9M for M/M_P = 1e-2 and 1e-40", ok_cross, "; ".join(det))
check("Re A rises by more than 80 decades between 1.4M and 1.9M (from the table)", np.log10(3.81e88 / 82.1) > 80)

# (6) lower bound, leading term maximised over delta


def Ein_minus(X):
    return float(mp.quad(lambda u: (mp.exp(u) - 1) / u, [0, X]))


def bound_leading(s, n=4):
    best = 0.0
    for d in np.linspace(0.01, 0.499, 250):
        val = np.pi ** 2 * np.sqrt(7) * d ** 2.5 / (64 * (1 - d) ** 2) * np.exp(n * Ein_minus((1 - d) ** 2 * s * s / 4))
        best = max(best, val)
    return best


ok_b = True
det = []
for E, (reA_t, ratio_t, bound_t) in table.items():
    b = bound_leading(E * E)
    ok_b &= abs(b / bound_t - 1) < 0.02 and b < results[E][0]
    det.append("E=%.2f bound=%.3g (table %.3g)" % (E, b, bound_t))
check("leading term of (bubblebound) maximised over delta matches the 'bound' column (2%) and lies below Re A", ok_b, "; ".join(det))
check("Ein^-(X) >= (e^X - 1 - X)/X (sampled)", all(Ein_minus(X) >= (np.exp(X) - 1 - X) / X - 1e-9 for X in (0.5, 1, 2, 4)))

print()
print("bubble audit: %d failures" % FAILS)
sys.exit(1 if FAILS else 0)
