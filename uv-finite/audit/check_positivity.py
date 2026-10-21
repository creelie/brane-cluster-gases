#!/usr/bin/env python3
"""Audit 04: the free Euclidean graviton of the gas is not reflection positive.

Sec. "In what sense the theory is ultraviolet complete" argues: a Gaussian
Euclidean field is reflection positive only if its covariance is a Stieltjes
function int d rho(m^2)/(z + m^2) with rho >= 0; a Stieltjes function analytic
across the negative real axis is c/z; 1/(z a(z)) is entire away from z = 0 and
is not c/z, so the free field fails reflection positivity.  This script makes
the failure quantitative, independently of that argument:

  For fixed spatial momentum q the time-slice covariance
      F_q(t) = int dp0/(2 pi) e^{i p0 t} / ((p0^2 + q^2) a(p0^2 + q^2))
  of a reflection-positive Gaussian field is a Laplace transform of a positive
  measure in the energy, so every Hankel matrix [F_q(t_i + t_j)] is positive
  semidefinite and F_q is completely monotone (positive, decreasing, convex,...).
  A negative Hankel eigenvalue, or a sign change of F_q or of (-d/dt)^k F_q,
  certifies the failure.

Checks:
  (1) the control a = 1 (general relativity): F_q(t) = e^{-q t}/(2 q); Hankel
      matrices PSD, completely monotone (so the numerics do not produce false
      negatives);
  (2) the skeleton a = exp[(n/2) Ein(z^2)] (M = 1) for n = 4, 6, 8 and several
      q: a negative Hankel eigenvalue far above the quadrature error, and a sign
      change of F_q(t) itself at finite t;
  (3) the sign of the deviation is robust: it survives doubling the precision
      and changing the grid;
  (4) the general obstruction, in the Stieltjes form: the spectral function
      that would have to be positive, rho(m^2) = (1/pi) Im[1/(z a(z))] at
      z = -m^2 + i0, vanishes identically for m^2 > 0 (a is real-entire), so
      the only candidate measure is the delta at m^2 = 0 with weight 1, and
      then the covariance would be 1/z, which differs from 1/(z a) as soon as
      a is not 1; the difference is quantified.

This is a property of every entire non-constant form factor, as the paper
says; the script shows it is not an artifact of the argument.
"""
import sys
import numpy as np
import mpmath as mp

mp.mp.dps = 20
FAILS = 0


def check(name, ok, detail=""):
    global FAILS
    print(("PASS " if ok else "FAIL ") + name + ("   [" + detail + "]" if detail else ""))
    if not ok:
        FAILS += 1


def Ein(w):
    if w < 1:
        return mp.nsum(lambda k: (-1) ** (k + 1) * w ** k / (k * mp.factorial(k)), [1, mp.inf])
    return mp.log(w) + mp.euler + mp.e1(w)


def a_skel(z, n):
    return mp.exp(n / mp.mpf(2) * Ein(z * z))


def F_time(t, q, n):
    """time-slice covariance F_q(t) = (1/pi) int_0^inf cos(p t) / ((p^2+q^2) a(p^2+q^2)) dp.
    The tail is oscillatory with a slowly decaying envelope when a = 1, so t > 0 uses
    mpmath's oscillatory quadrature (period 2 pi / t); n = None means a = 1."""
    if n is None:
        f = lambda p: mp.cos(p * t) / (p * p + q * q)
    else:
        f = lambda p: mp.cos(p * t) / ((p * p + q * q) * a_skel(p * p + q * q, n))
    if t == 0:
        return mp.quad(f, [0, 1, 2, 4, mp.inf]) / mp.pi
    return mp.quadosc(f, [0, mp.inf], omega=t) / mp.pi


def hankel_min_eig(ts, q, n):
    vals = {}
    N = len(ts)
    H = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            s = ts[i] + ts[j]
            if s not in vals:
                vals[s] = float(F_time(mp.mpf(s), q, n))
            H[i, j] = vals[s]
    ev = np.linalg.eigvalsh(H)
    return ev.min(), ev.max(), H


ts = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

# (1) control: general relativity
ok = True
det = []
for q in (mp.mpf('0.5'), mp.mpf(1), mp.mpf(2)):
    exact = [mp.exp(-q * t) / (2 * q) for t in ts]
    num = [F_time(mp.mpf(t), q, None) for t in ts]
    ok &= all(abs(x - y) < mp.mpf(10) ** -15 for x, y in zip(exact, num))
    mn, mx, H = hankel_min_eig(ts, q, None)
    ok &= mn > -1e-13
    det.append("q=%s minEig=%.2e" % (q, mn))
check("control a = 1: F_q(t) = e^{-qt}/(2q), Hankel matrices PSD (reflection positive)", ok, "; ".join(det))

# (2) skeleton: Hankel test and sign changes
summary = []
ok_neg = True
ok_sign = True
for n in (4, 6, 8):
    for q in (mp.mpf('0.3'), mp.mpf(1)):
        mn, mx, H = hankel_min_eig(ts, q, n)
        ok_neg &= mn < -1e-6 * mx
        # sign change of F itself on a fine grid
        grid = [mp.mpf(i) for i in range(0, 21)]
        vals = [F_time(t, q, n) for t in grid]
        signs = [1 if v > 0 else -1 for v in vals]
        changes = sum(1 for i in range(1, len(signs)) if signs[i] != signs[i - 1])
        tneg = next((float(t) for t, v in zip(grid, vals) if v < 0), None)
        if q == 1:
            ok_sign &= changes >= 1
        summary.append("n=%d q=%s: minEig/maxEig=%.3e, F<0 first at t=%s" % (n, q, mn / mx, tneg))
check("skeleton n=4,6,8: a Hankel matrix of F_q has a negative eigenvalue (not reflection positive)", ok_neg, "; ".join(summary))
check("skeleton, q = M: F_q(t) itself changes sign at finite t (a completely monotone function cannot)", ok_sign)

# (3) robustness: higher precision, shifted grid
mp.mp.dps = 30
mn1, mx1, _ = hankel_min_eig(ts, mp.mpf(1), 4)
mn2, mx2, _ = hankel_min_eig([0.25, 0.75, 1.25, 1.75, 2.25, 2.75], mp.mpf(1), 4)
check("negative eigenvalue persists at 30 digits and on a shifted grid (n=4, q=1)", mn1 < -1e-6 * mx1 and mn2 < -1e-6 * mx2,
      "minEig/maxEig = %.3e, %.3e" % (mn1 / mx1, mn2 / mx2))
mp.mp.dps = 20

# (4) the Stieltjes obstruction, quantified.  For a Stieltjes covariance
# C(z) = int d rho(m^2)/(z + m^2) the measure is recovered from the jump across
# the negative axis.  1/(z a(z)) has no jump (a entire, real on the real axis), so
# rho is supported at m^2 = 0 with weight = residue at z = 0 = 1/a(0) = 1.
# Then C would be 1/z, and the L^2 distance between 1/(z a) and 1/z on the
# Euclidean axis is positive for every n: the form factor itself is the
# obstruction.
for n in (4, 6, 8):
    res = mp.limit(lambda z: z / (z * a_skel(z, n)), 0)
    ok_res = abs(res - 1) < mp.mpf(10) ** -20
    jump_ok = all(abs(mp.im(1 / (z * a_skel(z, n)))) < mp.mpf(10) ** -20 for z in (mp.mpf(-0.5), mp.mpf(-1), mp.mpf(-2)))
    dist = mp.sqrt(mp.quad(lambda x: (1 / (x * a_skel(x, n)) - 1 / x) ** 2, [0, 1, 2, 5, mp.inf]))
    check("n=%d: residue of 1/(z a) at 0 is 1, no jump on the negative axis, ||1/(za) - 1/z||_2 = %s > 0" % (n, mp.nstr(dist, 6)),
          ok_res and jump_ok and dist > mp.mpf('0.1'))
# a is real and positive on the whole real axis (so the Euclidean Gaussian measure exists) and
# yet the Stieltjes test fails: the two statements are compatible and both true.
check("a(z) > 0 on the real axis for n = 4 (Gaussian measure well defined)", all(a_skel(z, 4) > 0 for z in (mp.mpf(-3), mp.mpf(-1), mp.mpf(0), mp.mpf(1), mp.mpf(3))))

print()
print("reflection positivity audit: %d failures" % FAILS)
sys.exit(1 if FAILS else 0)
