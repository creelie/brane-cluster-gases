#!/usr/bin/env python3
"""Audit 01: the one-loop pole matrix of the three quartic operators.

Re-derives, in exact rational arithmetic and independently of verify.py,
every step of Sec. "Three quartic operators" and Appendix "Angular averages"
of uv-finite-brane-gas.tex:

  (a) the moments <k k> = delta/4 and <k k k k> = (3 pairings)/24 on S^3,
      computed by direct integration in hyperspherical coordinates;
  (b) the spin projectors P2, P0 and the tensors theta, T, with their algebra
      (P2 P0 = 0, P2 theta = 0, theta.P0.theta = 3, tr P2 = 5);
  (c) the linearized Ricci and Riemann tensors of a plane-wave perturbation,
      from the Christoffel symbols, and the identities
      delta R_{mu nu} = (k^2/2)[(P2 h) + (theta.h) T]   (kappa = 1),
      Rbar . delta Riem = 2 k^2 W . h  with  W^{nu sigma} = Rbar^{mu nu rho sigma} k_mu k_rho / k^2;
  (d) the two-point functions of the linearized curvatures in the free local
      theory (covariance (P2 - P0/2)/(c k^{2n+2}) on the gauge-invariant part);
  (e) the angular averages of Eqs. (angular) and (angularRiem), for a
      generic symmetric tensor and for random algebraic curvature tensors
      built from Kulkarni-Nomizu products (exact rationals);
  (f) the pole matrix (1/c)[[-12,-1,-8/3],[0,1,8],[0,0,2]], its determinant
      -24, and the tuning that cancels all three poles.

Every check prints PASS or FAIL; the script exits nonzero on any FAIL.
"""
import itertools
import random
import sys

import sympy as sp
from sympy import Rational as Q

FAILS = 0


def check(name, ok):
    global FAILS
    print(("PASS " if ok else "FAIL ") + name)
    if not ok:
        FAILS += 1


D = 4
idx = range(D)

# ---------------------------------------------------------------- (a) moments
# Hyperspherical coordinates on S^3: k = (cos a, sin a cos b, sin a sin b cos g,
# sin a sin b sin g), measure sin^2 a sin b da db dg, total area 2 pi^2.
a, b, g = sp.symbols("a b g", real=True)
kvec = [sp.cos(a), sp.sin(a) * sp.cos(b), sp.sin(a) * sp.sin(b) * sp.cos(g),
        sp.sin(a) * sp.sin(b) * sp.sin(g)]
meas = sp.sin(a) ** 2 * sp.sin(b)
area = sp.integrate(meas, (a, 0, sp.pi), (b, 0, sp.pi), (g, 0, 2 * sp.pi))
check("area of S^3 is 2 pi^2", sp.simplify(area - 2 * sp.pi ** 2) == 0)


def sphere_avg(expr):
    return sp.simplify(sp.integrate(sp.expand(expr * meas), (a, 0, sp.pi),
                                    (b, 0, sp.pi), (g, 0, 2 * sp.pi)) / area)


# second moment: all 10 independent components
ok2 = True
for i in idx:
    for j in idx:
        if i <= j:
            val = sphere_avg(kvec[i] * kvec[j])
            ok2 &= val == (Q(1, 4) if i == j else 0)
check("<k_i k_j> = delta_ij / 4 by direct integration", ok2)

# fourth moment: compare with the three-pairing formula on every component
def pair4(i, j, k, l):
    d = lambda x, y: 1 if x == y else 0
    return Q(d(i, j) * d(k, l) + d(i, k) * d(j, l) + d(i, l) * d(j, k), 24)


ok4 = True
for comb in itertools.combinations_with_replacement(idx, 4):
    val = sphere_avg(kvec[comb[0]] * kvec[comb[1]] * kvec[comb[2]] * kvec[comb[3]])
    ok4 &= (val == pair4(*comb))
check("<k_i k_j k_k k_l> = (3 pairings)/24 by direct integration (all 35 components)", ok4)

# ------------------------------------------------------------- (b) projectors
def projectors(k):
    """theta, T, P2, P0 for a 4-vector k (symbolic or exact), as nested lists."""
    k2 = sum(x * x for x in k)
    th = [[(1 if i == j else 0) - k[i] * k[j] / k2 for j in idx] for i in idx]
    om = [[k[i] * k[j] / k2 for j in idx] for i in idx]
    T = [[th[i][j] / 3 + om[i][j] for j in idx] for i in idx]
    P2 = {}
    P0 = {}
    for i, j, m, n in itertools.product(idx, repeat=4):
        P2[i, j, m, n] = Q(1, 2) * (th[i][m] * th[j][n] + th[i][n] * th[j][m]) - Q(1, 3) * th[i][j] * th[m][n]
        P0[i, j, m, n] = Q(1, 3) * th[i][j] * th[m][n]
    return th, T, P2, P0


def mul4(A, B):
    return {(i, j, m, n): sp.expand(sum(A[i, j, p, q] * B[p, q, m, n] for p in idx for q in idx))
            for i, j, m, n in itertools.product(idx, repeat=4)}


def apply4(A, h):
    return [[sp.expand(sum(A[i, j, m, n] * h[m][n] for m in idx for n in idx)) for j in idx] for i in idx]


kk = [Q(3), Q(-1), Q(2), Q(5)]          # an exact non-axis direction
th, T, P2, P0 = projectors(kk)
P2P2 = mul4(P2, P2)
check("P2 P2 = P2", all(P2P2[key] == P2[key] for key in P2))
check("P0 P0 = P0", all(mul4(P0, P0)[key] == P0[key] for key in P0))
check("P2 P0 = 0", all(v == 0 for v in mul4(P2, P0).values()))
check("tr P2 = 5", sum(P2[i, j, i, j] for i in idx for j in idx) == 5)
check("tr P0 = 1", sum(P0[i, j, i, j] for i in idx for j in idx) == 1)
check("P2 theta = 0", all(v == 0 for row in apply4(P2, th) for v in row))
thP0th = sum(th[i][j] * P0[i, j, m, n] * th[m][n] for i, j, m, n in itertools.product(idx, repeat=4))
check("theta . P0 . theta = D - 1 = 3", thP0th == 3)
delta = [[1 if i == j else 0 for j in idx] for i in idx]
check("delta . P2 . delta = 0", sum(delta[i][j] * P2[i, j, m, n] * delta[m][n]
                                   for i, j, m, n in itertools.product(idx, repeat=4)) == 0)
check("T . delta = 2", sum(T[i][i] for i in idx) == 2)

# ------------------------------------------------- (c) linearized curvatures
# Metric g = delta + eps h e^{i k x}; curvature to first order in eps, from the
# Christoffel symbols. Conventions of App. "Conventions": R^r_{s m n} =
# d_m Gamma^r_{n s} - d_n Gamma^r_{m s} + ..., R_{mn} = R^r_{m r n}.
x = sp.symbols("x0:4", real=True)
eps = sp.symbols("epsilon")
ksym = sp.symbols("k0:4", real=True)
hsym = sp.Matrix(4, 4, lambda i, j: sp.Symbol("h%d%d" % (min(i, j), max(i, j))))
phase = sp.exp(sp.I * sum(ksym[i] * x[i] for i in idx))
gmat = sp.eye(4) + eps * hsym * phase
ginv = sp.eye(4) - eps * hsym * phase   # inverse to first order


def christoffel(gm, gi):
    Gam = [[[0] * D for _ in idx] for _ in idx]
    for r, m, n in itertools.product(idx, repeat=3):
        Gam[r][m][n] = sum(gi[r, s] * (sp.diff(gm[s, m], x[n]) + sp.diff(gm[s, n], x[m]) - sp.diff(gm[m, n], x[s]))
                           for s in idx) / 2
    return Gam


Gam = christoffel(gmat, ginv)


def riemann_lin(Gam):
    R = {}
    for r, s, m, n in itertools.product(idx, repeat=4):
        val = sp.diff(Gam[r][n][s], x[m]) - sp.diff(Gam[r][m][s], x[n])
        R[r, s, m, n] = sp.expand(val).coeff(eps, 1)   # first order; Gamma Gamma is O(eps^2)
    return R


Rlin = riemann_lin(Gam)
# strip the plane wave: everything is proportional to phase
Rlin = {key: sp.simplify(v / phase) for key, v in Rlin.items()}
# lower the first index (delta at zeroth order): R_{rsmn}
Ric_lin = [[sp.expand(sum(Rlin[r, m, r, n] for r in idx)) for n in idx] for m in idx]
Rs_lin = sp.expand(sum(Ric_lin[m][m] for m in idx))

# Compare with the closed forms used in the paper, at an exact numeric k, with
# the symbolic h: delta R_{mn} = (k^2/2)[(P2 h) + (theta.h) T].
subsk = {ksym[i]: kk[i] for i in idx}
hl = [[hsym[i, j] for j in idx] for i in idx]
k2 = sum(v * v for v in kk)
P2h = apply4(P2, hl)
thh = sum(th[i][j] * hl[i][j] for i in idx for j in idx)
ok_ric = True
for m in idx:
    for n in idx:
        lhs = sp.expand(Ric_lin[m][n].subs(subsk))
        rhs = sp.expand(k2 / 2 * (P2h[m][n] + thh * T[m][n]))
        ok_ric &= sp.simplify(lhs - rhs) == 0
check("linearized Ricci = (k^2/2)[(P2 h) + (theta.h) T]  (Eq. Ric1)", ok_ric)
lhsR = sp.expand(Rs_lin.subs(subsk))
check("linearized scalar curvature = k^2 (theta.h)", sp.simplify(lhsR - k2 * thh) == 0)
# gauge invariance: h -> h + k xi + xi k leaves the linearized curvatures unchanged
xi = sp.symbols("xi0:4")
gauge = {hsym[min(i, j), max(i, j)]: hsym[min(i, j), max(i, j)] + kk[i] * xi[j] + kk[j] * xi[i]
         for i in idx for j in idx if i <= j}
ok_gauge = all(sp.simplify(sp.expand(Rlin[key].subs(subsk)).subs(gauge, simultaneous=True)
                           - sp.expand(Rlin[key].subs(subsk))) == 0 for key in Rlin)
check("linearized Riemann is gauge invariant", ok_gauge)

# Riemann contraction with an algebraic curvature tensor: Rbar . dRiem = 2 k^2 W . h
# Build algebraic curvature tensors as sums of Kulkarni-Nomizu products of
# symmetric matrices; they have all the Riemann symmetries and the cyclic identity.


def kn(A, B):
    R = {}
    for i, j, m, n in itertools.product(idx, repeat=4):
        R[i, j, m, n] = A[i][m] * B[j][n] + A[j][n] * B[i][m] - A[i][n] * B[j][m] - A[j][m] * B[i][n]
    return R


def rand_sym(rng):
    M = [[Q(rng.randint(-5, 5)) for _ in idx] for _ in idx]
    return [[M[i][j] + M[j][i] for j in idx] for i in idx]


def rand_curv(rng, terms=3):
    R = {key: Q(0) for key in itertools.product(idx, repeat=4)}
    for _ in range(terms):
        A, B = rand_sym(rng), rand_sym(rng)
        K = kn(A, B)
        for key in R:
            R[key] += K[key]
    return R


def curv_symmetries(R):
    ok = True
    for i, j, m, n in itertools.product(idx, repeat=4):
        ok &= R[i, j, m, n] == -R[j, i, m, n] == -R[i, j, n, m] == R[m, n, i, j]
        ok &= R[i, j, m, n] + R[i, m, n, j] + R[i, n, j, m] == 0
    return ok


rng = random.Random(20261021)
Rb = rand_curv(rng)
check("random Kulkarni-Nomizu sum has the Riemann symmetries and the cyclic identity", curv_symmetries(Rb))
Rlin_num = {key: sp.expand(v.subs(subsk)) for key, v in Rlin.items()}
contr = sp.expand(sum(Rb[key] * Rlin_num[key] for key in Rb))
W = [[sum(Rb[m, nu, r, s] * kk[m] * kk[r] for m in idx for r in idx) / k2 for s in idx] for nu in idx]
Wh = sp.expand(2 * k2 * sum(W[nu][s] * hl[nu][s] for nu in idx for s in idx))
check("Rbar . delta Riem = 2 k^2 W . h  (Eq. RiemW)", sp.simplify(contr - Wh) == 0)
check("W symmetric and transverse", all(W[i][j] == W[j][i] for i in idx for j in idx)
      and all(sum(kk[i] * W[i][j] for i in idx) == 0 for j in idx))

# ------------------------------------------------ (d) curvature correlators
# Free local theory: <h h> = (P2 - P0/2) * s(k) on the gauge-invariant part,
# s(k) = M^{2n}/(c k^{2n+2}); the gauge-dependent part drops out because the
# linearized curvatures are gauge invariant (checked above).  Define
# C(k) = P2 - P0/2 and compute the curvature correlators as bilinear forms.
Cov = {key: P2[key] - P0[key] / 2 for key in P2}


def corr_ricci(Rbar_sym):
    """<(Rbar . dRic)(Rbar . dRic)> / s(k), with Rbar symmetric, kappa=1."""
    # dRic = (k^2/2) [P2 h + (theta.h) T]  ->  vertex V_{ab} = (k^2/2)[ (P2 Rbar)_{ab} + (T.Rbar) theta_{ab} ]
    P2R = apply4(P2, Rbar_sym)
    TR = sum(T[i][j] * Rbar_sym[i][j] for i in idx for j in idx)
    V = [[k2 / 2 * (P2R[i][j] + TR * th[i][j]) for j in idx] for i in idx]
    return sp.expand(sum(V[i][j] * Cov[i, j, m, n] * V[m][n] for i, j, m, n in itertools.product(idx, repeat=4)))


Rsym = [[sp.Symbol("r%d%d" % (min(i, j), max(i, j))) for j in idx] for i in idx]
corrR = corr_ricci(Rsym)
# the paper: (k^4/4) [ Rbar.P2.Rbar - (3/2) (T.Rbar)^2 ]
RP2R = sum(Rsym[i][j] * P2[i, j, m, n] * Rsym[m][n] for i, j, m, n in itertools.product(idx, repeat=4))
TR = sum(T[i][j] * Rsym[i][j] for i in idx for j in idx)
check("<(Rbar.dRic)^2> = (k^4/4)[Rbar.P2.Rbar - (3/2)(T.Rbar)^2]  (Eq. RicRic2pt)",
      sp.simplify(corrR - k2 ** 2 / 4 * (RP2R - Q(3, 2) * TR ** 2)) == 0)
# scalar: Rbar = delta gives <dR dR> = -(3/2) k^4 ... with Rbar=lambda delta: (k^4/4)(0 - 3/2 * 4 lambda^2) = -(3/2)k^4 lambda^2
corr_scalar = corr_ricci(delta)
check("<dR dR> = -(3/2) k^4  (Eq. RR2pt)", sp.simplify(corr_scalar + Q(3, 2) * k2 ** 2) == 0)
# Riemann: vertex 2 k^2 W, so <(Rbar.dRiem)^2> = 4 k^4 W.C.W = 4 k^4 [tr W^2 - (1/2)(tr W)^2]
Wnum = W
WCW = sum(Wnum[i][j] * Cov[i, j, m, n] * Wnum[m][n] for i, j, m, n in itertools.product(idx, repeat=4))
trW = sum(Wnum[i][i] for i in idx)
trW2 = sum(Wnum[i][j] * Wnum[j][i] for i in idx for j in idx)
check("W.(P2 - P0/2).W = tr W^2 - (1/2)(tr W)^2  (Eq. RiemRiem2pt)", sp.simplify(WCW - (trW2 - trW ** 2 / 2)) == 0)

# ----------------------------------------------- (e) exact angular averages
# Use the exact moments (a) to average over the direction of k.  Build the
# averaged bilinear forms by expanding theta, T in k-hat and integrating.
kh = sp.symbols("n0:4", real=True)


def avg_poly(expr):
    """Average a polynomial in the components of k-hat over S^3 using (a)."""
    expr = sp.expand(expr)
    poly = sp.Poly(expr, *kh)
    total = 0
    for monom, coeff in poly.terms():
        deg = sum(monom)
        if deg == 0:
            total += coeff
        elif deg == 2:
            inds = [i for i in idx for _ in range(monom[i])]
            total += coeff * (Q(1, 4) if inds[0] == inds[1] else 0)
        elif deg == 4:
            inds = [i for i in idx for _ in range(monom[i])]
            total += coeff * pair4(*inds)
        else:
            raise ValueError("odd degree %d" % deg)
    return sp.expand(total)


thS, TS, P2S, P0S = projectors(list(kh))
# replace k2 by 1 (unit vector): projectors() divides by sum kh^2; substitute by hand
unit = {sum(v * v for v in kh): 1}


def unitize(e):
    return sp.expand(e.subs(sum(v * v for v in kh), 1))


# generic symmetric Rbar
RP2R_S = unitize(sum(Rsym[i][j] * P2S[i, j, m, n] * Rsym[m][n] for i, j, m, n in itertools.product(idx, repeat=4)))
TR_S = unitize(sum(TS[i][j] * Rsym[i][j] for i in idx for j in idx))
avg_ric = avg_poly(RP2R_S - Q(3, 2) * TR_S ** 2)
Ric2 = sum(Rsym[i][j] * Rsym[i][j] for i in idx for j in idx)
Rtr = sum(Rsym[i][i] for i in idx)
check("<Rbar.P2.Rbar - (3/2)(T.Rbar)^2> = (1/2)(Ric^2 - R^2)  (Eq. angular, generic symmetric Rbar)",
      sp.simplify(avg_ric - (Ric2 - Rtr ** 2) / 2) == 0)
# the individual averages of App. angular
check("<(k Rbar k)^2> = (R^2 + 2 Ric^2)/24",
      sp.simplify(avg_poly(unitize(sum(kh[i] * Rsym[i][j] * kh[j] for i in idx for j in idx) ** 2))
                  - (Rtr ** 2 + 2 * Ric2) / 24) == 0)
check("<Rbar.P2.Rbar> = (5/9) Ric^2 - (5/36) R^2",
      sp.simplify(avg_poly(RP2R_S) - (Q(5, 9) * Ric2 - Q(5, 36) * Rtr ** 2)) == 0)
check("<(T.Rbar)^2> = (13/54) R^2 + (1/27) Ric^2",
      sp.simplify(avg_poly(TR_S ** 2) - (Q(13, 54) * Rtr ** 2 + Q(1, 27) * Ric2)) == 0)

# Riemann average, with random exact algebraic curvature tensors
ok_riem = True
for trial in range(4):
    Rb = rand_curv(rng, terms=3)
    Wk = [[sum(Rb[m, nu, r, s] * kh[m] * kh[r] for m in idx for r in idx) for s in idx] for nu in idx]
    trW = sum(Wk[i][i] for i in idx)
    trW2 = sum(Wk[i][j] * Wk[j][i] for i in idx for j in idx)
    avg = avg_poly(trW2 - trW ** 2 / 2)
    Riem2 = sum(v * v for v in Rb.values())
    Ricb = [[sum(Rb[r, i, r, j] for r in idx) for j in idx] for i in idx]
    Ric2b = sum(Ricb[i][j] ** 2 for i in idx for j in idx)
    Rb_s = sum(Ricb[i][i] for i in idx)
    ok_riem &= (avg == (3 * Riem2 - Rb_s ** 2) / 48)
    # the intermediate identities of the appendix
    ok_riem &= (avg_poly(trW2) == (Ric2b + Q(3, 2) * Riem2) / 24)
    X = sum(Rb[i, j, m, n] * Rb[i, m, j, n] for i, j, m, n in itertools.product(idx, repeat=4))
    ok_riem &= (2 * X == Riem2)
check("<tr W^2 - (1/2)(tr W)^2> = (3 Riem^2 - R^2)/48 on 4 random algebraic curvature tensors (Eq. angularRiem)", ok_riem)

# ---------------------------------------------------------- (f) pole matrix
c, s1, s2, s3 = sp.symbols("c s1 s2 s3")
# From Eq. (Sq2): coefficients 8 s_i /(kappa^2 M^{2n}); correlators carry kappa^2 M^{2n}/(c k^{2n-2});
# (-Box)^{n-3} -> k^{2n-6}; integrand k^{-4}, pole 1/(16 pi^2 eps).  In units of 1/(16 pi^2 eps):
pole_s1 = 8 * s1 * (-Q(3, 2) / c)                     # times R^2
R2, Ric2s, E = sp.symbols("R2 Ric2 E")
pole_s2 = 8 * s2 * Q(1, 4) / c * Q(1, 2) * (Ric2s - R2)
Riem2s = E + 4 * Ric2s - R2
pole_s3 = 8 * s3 * 4 / c * (3 * Riem2s - R2) / 48
total = sp.expand(pole_s1 * R2 + pole_s2 + pole_s3)
Mmat = sp.Matrix([[total.coeff(R2).coeff(sv) for sv in (s1, s2, s3)],
                  [total.coeff(Ric2s).coeff(sv) for sv in (s1, s2, s3)],
                  [total.coeff(E).coeff(sv) for sv in (s1, s2, s3)]])
expected = sp.Matrix([[-12, -1, Q(-8, 3)], [0, 1, 8], [0, 0, 2]]) / c
check("pole matrix = (1/c)[[-12,-1,-8/3],[0,1,8],[0,0,2]]  (Eq. killcontrib)", sp.simplify(Mmat - expected) == sp.zeros(3))
check("determinant = -24/c^3", sp.simplify(Mmat.det() + 24 / c ** 3) == 0)
b1, b2, bE = sp.symbols("beta1 beta2 betaE")
tune = {s3: -c / 2 * bE, s2: -c * (b2 - 4 * bE), s1: c / 12 * (b1 + b2 - Q(8, 3) * bE)}
res = sp.simplify(sp.Matrix([b1, b2, bE]) + (Mmat * sp.Matrix([s1, s2, s3])).subs(tune))
check("tuning (Eq. tuning) cancels beta1, beta2, betaE", res == sp.zeros(3, 1))
# the inverse-matrix solution agrees with the closed form
sol = Mmat.solve(-sp.Matrix([b1, b2, bE]))
check("closed-form tuning equals the unique solution", all(sp.simplify(sol[i] - tune[v]) == 0 for i, v in enumerate((s1, s2, s3))))

print()
print("one-loop matrix audit: %d failures" % FAILS)
sys.exit(1 if FAILS else 0)
