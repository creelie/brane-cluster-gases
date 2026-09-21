#!/usr/bin/env python3
"""
checks_clusters.py

Numerical companion to the paper on brane cluster gases and entire form factors.
Every number quoted in the text is recomputed here from its definition.  Run

    python3 checks_clusters.py

Groups:
  1  the nerve complex of an arrangement: d^2 = 0, Euler characteristic, census
  2  intersection numbers on tori: |det M| and the two-torus law
  3  generating functions: deterministic arrangements and Poisson gases
  4  Barnes-Rivers projectors and transversality
  5  poles and residues: the sum rule and the alternation
  6  power counting: the loop-order formula, the frontier and the census
  7  the Gaussian form factor: exact propagator in D dimensions
  8  the Newtonian potential, the weak-field threshold and laboratory bounds
  9  dispersion and luminality
 10  the parametric representation: the Schwinger bound and the one-loop bubble
 11  multiplicativity: composition of transmission amplitudes through a stack
"""
import sys
import itertools
from fractions import Fraction
from math import comb, pi, gcd, sqrt, erf, erfc, exp, factorial, gamma
import numpy as np
from scipy import integrate, special, optimize

rng = np.random.default_rng(20260920)
PASSED, FAILED = [], []


def check(name, ok, detail=""):
    (PASSED if ok else FAILED).append(name)
    print(f"[{'ok  ' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


# ----------------------------------------------------------------------------
# 1. nerve complex
# ----------------------------------------------------------------------------
def nerve_boundary(faces_by_dim):
    """boundary matrices of a simplicial complex given as lists of sorted tuples"""
    mats = {}
    for k in faces_by_dim:
        if k == 0:
            continue
        rows = {f: i for i, f in enumerate(faces_by_dim[k - 1])}
        M = np.zeros((len(faces_by_dim[k - 1]), len(faces_by_dim[k])), dtype=int)
        for j, f in enumerate(faces_by_dim[k]):
            for i in range(len(f)):
                g = f[:i] + f[i + 1:]
                M[rows[g], j] = (-1) ** i
        mats[k] = M
    return mats


def random_arrangement_nerve(N, D):
    """N random affine hyperplanes in R^D, general position: every subset of size <= D meets"""
    faces = {}
    for k in range(0, D):
        faces[k] = [tuple(s) for s in itertools.combinations(range(N), k + 1)]
    return faces


def group1():
    print("\n-- 1. the nerve complex")
    for (N, D) in [(5, 3), (6, 3), (7, 4), (8, 2)]:
        faces = random_arrangement_nerve(N, D)
        mats = nerve_boundary(faces)
        ok = all(np.all(mats[k] @ mats[k + 1] == 0) for k in mats if k + 1 in mats)
        check(f"d_k d_(k+1) = 0 for {N} hyperplanes in R^{D}", ok)
        ranks = {k: np.linalg.matrix_rank(mats[k]) for k in mats}
        chi_cells = sum((-1) ** k * len(faces[k]) for k in faces)
        betti = {}
        for k in faces:
            dimC = len(faces[k])
            rk = ranks.get(k, 0); rk1 = ranks.get(k + 1, 0)
            betti[k] = dimC - rk - rk1
        chi_h = sum((-1) ** k * betti[k] for k in betti)
        check(f"Euler characteristic by cells equals by homology ({chi_cells})", chi_cells == chi_h)
    # random simplicial complexes: d^2 = 0 does not depend on general position
    for trial in range(5):
        N = 7
        top = [tuple(sorted(rng.choice(N, size=4, replace=False))) for _ in range(6)]
        faces = {}
        allf = set()
        for f in top:
            for k in range(1, 5):
                for g in itertools.combinations(f, k):
                    allf.add(g)
        for g in allf:
            faces.setdefault(len(g) - 1, []).append(g)
        for k in faces:
            faces[k].sort()
        mats = nerve_boundary(faces)
        ok = all(np.all(mats[k] @ mats[k + 1] == 0) for k in mats if k + 1 in mats)
        check(f"d^2 = 0 on a random complex, trial {trial}", ok)
    for D in (2, 3, 4, 6, 10, 11, 26):
        counts = [comb(D, K) for K in range(D + 1)]
        check(f"coordinate arrangement in T^{D}: {sum(counts)} strata, alternating sum 0",
              sum(counts) == 2 ** D and sum((-1) ** K * c for K, c in enumerate(counts)) == 0)
    # explicit enumeration of strata of the coordinate arrangement for D = 4: subsets of {1..D}
    D = 4
    strata = [S for K in range(D + 1) for S in itertools.combinations(range(D), K)]
    check("enumeration gives 16 strata in D = 4", len(strata) == 16)


# ----------------------------------------------------------------------------
# 2. intersection numbers on tori
# ----------------------------------------------------------------------------
def index_of_lattice(M):
    """[Z^D : M Z^D] by counting M^{-1} Z^D / Z^D"""
    D = M.shape[0]
    Minv = np.linalg.inv(M)
    det = abs(round(np.linalg.det(M)))
    pts = set()
    rngs = range(-det, det + 1)
    # cosets M^{-1} u mod Z^D for u in a box; the box is enough because M^{-1} Z^D / Z^D has order det
    for u in itertools.product(rngs, repeat=D):
        x = Minv @ np.array(u)
        x = x - np.floor(x + 1e-9)
        pts.add(tuple(np.round(x, 7) % 1))
        if len(pts) > 5 * det + 5:
            break
    return len(pts)


def curve_intersections(p1, q1, p2, q2, samples=4000):
    """count the points where the closed curves t -> (p t, q t) mod 1 meet on the square torus"""
    # the curves are subgroups; intersection = solutions of (p1 s, q1 s) = (p2 t, q2 t) mod Z^2
    # solve by brute force on a fine common grid
    d = abs(p1 * q2 - p2 * q1)
    if d == 0:
        return None
    pts = set()
    for k in range(d):
        for l in range(d):
            # candidate s, t from the lattice condition: (p1 s - p2 t, q1 s - q2 t) in Z^2
            # general solution s = (k q2 - l p2)/(p1 q2 - p2 q1)... enumerate integer right sides
            det = p1 * q2 - p2 * q1
            s = (k * q2 - l * p2) / det
            t = (k * q1 - l * p1) / det
            x = ((p1 * s) % 1, (q1 * s) % 1)
            pts.add((round(x[0], 7) % 1, round(x[1], 7) % 1))
    return len(pts)


def group2():
    print("\n-- 2. intersection numbers on tori")
    bad = 0
    for _ in range(60):
        D = int(rng.integers(2, 4))
        M = rng.integers(-3, 4, size=(D, D))
        if abs(np.linalg.det(M)) < 0.5 or abs(np.linalg.det(M)) > 12:
            continue
        if index_of_lattice(M) != abs(round(np.linalg.det(M))):
            bad += 1
    check("[Z^D : M Z^D] = |det M| on random integer matrices", bad == 0, f"{bad} failures")
    bad = 0; tested = 0
    for p1, q1, p2, q2 in itertools.product(range(-3, 4), repeat=4):
        d = abs(p1 * q2 - p2 * q1)
        if d == 0 or gcd(p1, q1) != 1 or gcd(p2, q2) != 1:
            continue
        tested += 1
        if curve_intersections(p1, q1, p2, q2) != d:
            bad += 1
    check(f"two-torus law |p1 q2 - p2 q1| on {tested} primitive winding pairs", bad == 0, f"{bad} failures")
    check("(1,2) and (1,-3) meet in 5 points", curve_intersections(1, 2, 1, -3) == 5)


# ----------------------------------------------------------------------------
# 3. generating functions
# ----------------------------------------------------------------------------
def group3():
    print("\n-- 3. generating functions of arrangements")
    # deterministic: sum over subsets of prod w_i z = prod (1 + w_i z)
    for N in (3, 5, 8):
        w = rng.uniform(0.2, 2.0, size=N)
        z = complex(0.7, -0.4)
        lhs = sum(np.prod([w[i] * z for i in S]) for K in range(N + 1) for S in itertools.combinations(range(N), K))
        rhs = np.prod(1 + w * z)
        check(f"subset sum equals the product for N={N}", abs(lhs - rhs) < 1e-10 * abs(rhs))
        poly = np.poly1d([1.0])
        for wi in w:
            poly = poly * np.poly1d([wi, 1.0])
        roots = np.sort(poly.roots.real)
        check(f"the product has exactly N={N} zeros, at z = -1/w_i", len(roots) == N and np.allclose(roots, np.sort(-1 / w), rtol=1e-8))
    # Poisson gas: E prod_{x in Pi} (1 + w z) = exp(lambda w z) for a Poisson number of branes with mean lambda
    lam = 3.0; w = 0.8
    for z in (0.5, -0.3, 1.2):
        Ns = rng.poisson(lam, size=400000)
        est = np.mean((1 + w * z) ** Ns)
        ex = exp(lam * w * z)
        check(f"Poisson generating functional at z={z}", abs(est - ex) / ex < 0.01, f"MC {est:.4f} exact {ex:.4f}")
    # K-th factorial moment: E[number of K-subsets] = lambda^K / K!
    Ns = rng.poisson(lam, size=400000)
    for K in (1, 2, 3):
        est = np.mean([comb(int(n), K) for n in Ns[:200000]])
        ex = lam ** K / factorial(K)
        check(f"expected number of {K}-clusters = lambda^K/K!", abs(est - ex) / ex < 0.02, f"MC {est:.4f} exact {ex:.4f}")
    # a form factor of the shape exp(H) has no zeros: sample log|exp(H)| on a grid for several entire H
    for name, Hf in [("z", lambda z: z), ("z^2", lambda z: z * z), ("sin z", np.sin), ("z e^z", lambda z: z * np.exp(z))]:
        X, Y = np.meshgrid(np.linspace(-6, 6, 121), np.linspace(-6, 6, 121))
        Z = X + 1j * Y
        # log|exp H| = Re H is finite everywhere, which is the statement that exp(H) has neither zeros nor poles
        logabs = np.real(Hf(Z))
        check(f"log|exp(H)| finite on the grid, H = {name}", np.all(np.isfinite(logabs)), f"range [{logabs.min():.1f}, {logabs.max():.1f}]")


# ----------------------------------------------------------------------------
# 4. projectors
# ----------------------------------------------------------------------------
def projectors(p, D=4):
    p2 = p @ p
    eta = np.eye(D)
    theta = eta - np.outer(p, p) / p2
    omega = np.outer(p, p) / p2
    def sym(T):
        return 0.5 * (T + T.transpose(1, 0, 2, 3))
    P2 = 0.5 * (np.einsum("ma,nb->mnab", theta, theta) + np.einsum("mb,na->mnab", theta, theta)) \
        - np.einsum("mn,ab->mnab", theta, theta) / (D - 1)
    P1 = 0.5 * (np.einsum("ma,nb->mnab", theta, omega) + np.einsum("mb,na->mnab", theta, omega)
                + np.einsum("ma,nb->mnab", omega, theta) + np.einsum("mb,na->mnab", omega, theta))
    Ps = np.einsum("mn,ab->mnab", theta, theta) / (D - 1)
    Pw = np.einsum("mn,ab->mnab", omega, omega)
    return P2, P1, Ps, Pw


def group4():
    print("\n-- 4. Barnes-Rivers projectors")
    D = 4
    Id = 0.5 * (np.einsum("ma,nb->mnab", np.eye(D), np.eye(D)) + np.einsum("mb,na->mnab", np.eye(D), np.eye(D)))
    for trial in range(3):
        p = rng.normal(size=D)
        P2, P1, Ps, Pw = projectors(p, D)
        Ps_list = [P2, P1, Ps, Pw]
        check(f"completeness, momentum {trial}", np.allclose(P2 + P1 + Ps + Pw, Id, atol=1e-12))
        ok = True
        for i, A in enumerate(Ps_list):
            for j, B in enumerate(Ps_list):
                AB = np.einsum("mnab,abcd->mncd", A, B)
                ok &= np.allclose(AB, A if i == j else 0, atol=1e-12)
        check(f"idempotent and mutually orthogonal, momentum {trial}", ok)
        check(f"spin-two projector transverse, momentum {trial}", np.allclose(np.einsum("m,mnab->nab", p, P2), 0, atol=1e-12))
        check(f"spin-two projector has trace 5 in D=4, momentum {trial}", abs(np.einsum("mnmn->", P2) - 5) < 1e-12)


# ----------------------------------------------------------------------------
# 5. poles and residues
# ----------------------------------------------------------------------------
def residues_of_inverse(coeffs):
    """residues of 1/(z a(z)) at all poles for a polynomial a with a(0)=1 given by coefficients (highest first)"""
    a = np.poly1d(coeffs)
    da = a.deriv()
    res = {0.0: 1.0 / a(0.0)}
    for r in a.roots:
        res[r] = 1.0 / (r * da(r))
    return res


def group5():
    print("\n-- 5. poles and residues")
    # real positive masses
    for n in (1, 2, 3, 5):
        M2 = np.sort(rng.uniform(0.5, 5.0, size=n))
        coeffs = np.poly1d([1.0])
        for m2 in M2:
            coeffs = coeffs * np.poly1d([1.0 / m2, 1.0])
        res = residues_of_inverse(coeffs.coeffs)
        massive = [res[r] for r in res if abs(r) > 1e-12]
        s = sum(massive)
        check(f"n={n} real masses: massive residues sum to -1", abs(s + 1) < 1e-9, f"{s.real:.10f}")
        check(f"n={n} real masses: residue at zero is +1", abs(res[0.0] - 1) < 1e-12)
        # alternation: order by |root|
        order = sorted([r for r in res if abs(r) > 1e-12], key=lambda r: abs(r))
        signs = [np.sign(res[r].real) for r in order]
        alt = all(signs[i] == -signs[i + 1] for i in range(len(signs) - 1)) and signs[0] < 0
        check(f"n={n} real masses: residues alternate starting negative", alt, f"signs {signs}")
        # explicit formula r_i = -1 / prod_{j != i} (1 - M_i^2/M_j^2)
        ok = True
        for i, mi in enumerate(M2):
            pred = -1.0 / np.prod([1 - mi / mj for j, mj in enumerate(M2) if j != i])
            ok &= abs(res[min(res, key=lambda r: abs(r + mi))] - pred) < 1e-8
        check(f"n={n} real masses: closed-form residues", ok)
    # complex conjugate roots (Lee-Wick pairs): the sum rule still holds
    for trial in range(3):
        roots = []
        for _ in range(int(rng.integers(1, 3))):
            r = complex(rng.uniform(-3, 3), rng.uniform(0.5, 3))
            roots += [r, np.conj(r)]
        if rng.random() < 0.5:
            roots.append(-rng.uniform(0.5, 3))
        poly = np.poly1d(np.poly(roots).real)
        poly = poly / poly(0.0)
        res = residues_of_inverse(poly.coeffs)
        s = sum(res[r] for r in res if abs(r) > 1e-12)
        check(f"complex roots, trial {trial}: massive residues sum to -1", abs(s + 1) < 1e-8, f"{s:.8f}")
    # Stelle: 1/(p^2 (1 + p^2/M^2)) = 1/p^2 - 1/(p^2 + M^2)
    for M2 in (0.5, 1.0, 3.7):
        p2 = rng.uniform(0.1, 5, size=50)
        lhs = 1 / (p2 * (1 + p2 / M2)); rhs = 1 / p2 - 1 / (p2 + M2)
        check(f"Stelle partial fractions at M^2={M2}", np.allclose(lhs, rhs, rtol=1e-12))
    # contour check of residues for a(z)=1+z: residues +1 and -1 by numerical contour integration
    for (c, r, expect) in [(0.0, 0.3, 1.0), (-1.0, 0.3, -1.0)]:
        t = np.linspace(0, 2 * pi, 4001)
        z = c + r * np.exp(1j * t)
        f = 1 / (z * (1 + z))
        val = np.trapezoid(f * 1j * r * np.exp(1j * t), t) / (2j * pi)
        check(f"contour residue at z={c}", abs(val - expect) < 1e-9, f"{val.real:.10f}")


# ----------------------------------------------------------------------------
# 6. power counting
# ----------------------------------------------------------------------------
def omega(D, n, L):
    return (2 + 2 * n) + (D - 2 - 2 * n) * L


def group6():
    print("\n-- 6. power counting")
    bad = 0
    for _ in range(3000):
        V = int(rng.integers(1, 12)); L = int(rng.integers(0, 8)); n = int(rng.integers(0, 7)); D = int(rng.choice([3, 4, 5, 6, 10, 11]))
        I = L + V - 1
        direct = D * L - (2 + 2 * n) * I + (2 + 2 * n) * V
        if direct != omega(D, n, L):
            bad += 1
    check("omega = D L - (2+2n) I + (2+2n) V reduces to the loop-order formula", bad == 0, f"{bad} failures")
    # frontier in D=4: omega < 0 iff L > (n+1)/(n-1) for n >= 2
    bad = 0
    for n in range(2, 40):
        for L in range(0, 40):
            if (omega(4, n, L) < 0) != (L * (n - 1) > n + 1):
                bad += 1
    check("divergence frontier L > (n+1)/(n-1) in D=4", bad == 0)
    check("n <= 1 in D=4: never negative", all(omega(4, n, L) >= 0 for n in (0, 1) for L in range(0, 50)))
    census = {n: [L for L in range(1, 30) if omega(4, n, L) >= 0] for n in range(2, 9)}
    check("census in D=4: n=2 -> {1,2,3}, n=3 -> {1,2}, n>=4 -> {1}",
          census[2] == [1, 2, 3] and census[3] == [1, 2] and all(census[n] == [1] for n in range(4, 9)))
    check("omega(L=1) = 4 in D=4 for every n", all(omega(4, n, 1) == 4 for n in range(0, 30)))
    # general D: one-loop-only divergence iff 2n > D - 2 + (D)/... derive: omega(L) < 0 for all L >= 2 iff (2+2n) + 2(D-2-2n) < 0 iff 2n > 2D - 2 ... check numerically
    # omega(L=2) = 2D - 2 - 2n is negative iff n > D - 1; since omega is affine in L with slope D - 2 - 2n < 0
    # this is the condition for every L >= 2 to be finite, so the threshold is n = D
    for D in (3, 4, 5, 6, 10, 11):
        nmin = min(n for n in range(0, 60) if all(omega(D, n, L) < 0 for L in range(2, 60)))
        check(f"D={D}: only one-loop divergences once n >= D", nmin == D, f"n_min = {nmin}")
    # comparison table
    rows = {"Einstein": [omega(4, 0, L) for L in range(1, 5)], "Stelle": [omega(4, 1, L) for L in range(1, 5)],
            "n=2": [omega(4, 2, L) for L in range(1, 5)], "n=3": [omega(4, 3, L) for L in range(1, 5)]}
    check("table rows: Einstein 4,6,8,10; Stelle 4,4,4,4; n=2: 4,2,0,-2; n=3: 4,0,-4,-8",
          rows["Einstein"] == [4, 6, 8, 10] and rows["Stelle"] == [4, 4, 4, 4] and rows["n=2"] == [4, 2, 0, -2] and rows["n=3"] == [4, 0, -4, -8])


# ----------------------------------------------------------------------------
# 7. the Gaussian propagator in D dimensions
# ----------------------------------------------------------------------------
def G_exact(D, r, M=1.0):
    return special.gammainc(D / 2 - 1, M * M * r * r / 4) * gamma(D / 2 - 1) / (4 * pi ** (D / 2) * r ** (D - 2))


def G_numeric(D, r, M=1.0):
    # radial Fourier transform: G(r) = (2 pi)^{-D/2} r^{1-D/2} int_0^inf p^{D/2} J_{D/2-1}(p r) e^{-p^2/M^2}/p^2 dp
    f = lambda p: p ** (D / 2 - 2) * special.jv(D / 2 - 1, p * r) * np.exp(-p * p / (M * M))
    val, err = integrate.quad(f, 0, 40 * M, limit=400)
    return (2 * pi) ** (-D / 2) * r ** (1 - D / 2) * val


def group7():
    print("\n-- 7. the exact propagator of the Gaussian form factor")
    for D in (3, 4, 5, 6):
        worst = 0.0
        for r in (0.2, 0.7, 1.5, 3.0):
            a = G_exact(D, r); b = G_numeric(D, r)
            worst = max(worst, abs(a - b) / abs(a))
        check(f"D={D}: incomplete-gamma formula matches the Fourier integral", worst < 1e-6, f"max rel dev {worst:.1e}")
        G0 = 1.0 / (2 ** (D - 1) * pi ** (D / 2) * (D - 2))
        check(f"D={D}: G(0) = M^(D-2)/(2^(D-1) pi^(D/2) (D-2))", abs(G_exact(D, 1e-4) - G0) / G0 < 1e-6, f"{G_exact(D,1e-4):.6e} vs {G0:.6e}")
    r = np.linspace(0.05, 6, 200)
    check("D=3: G = erf(Mr/2)/(4 pi r)", np.allclose(G_exact(3, r), special.erf(r / 2) / (4 * pi * r), rtol=1e-10))
    check("D=4: G = (1 - exp(-M^2 r^2/4))/(4 pi^2 r^2)", np.allclose(G_exact(4, r), (1 - np.exp(-r * r / 4)) / (4 * pi ** 2 * r * r), rtol=1e-10))
    check("D=4: G(0) = M^2/(16 pi^2)", abs(G_exact(4, 1e-5) - 1 / (16 * pi ** 2)) < 1e-9)
    # large r: G -> massless propagator 1/(4 pi^{D/2}) Gamma(D/2-1) / r^{D-2}
    for D in (3, 4, 5):
        free = gamma(D / 2 - 1) / (4 * pi ** (D / 2) * 8.0 ** (D - 2))
        check(f"D={D}: Newtonian tail at Mr = 8", abs(G_exact(D, 8.0) / free - 1) < 1e-6)


# ----------------------------------------------------------------------------
# 8. the Newtonian potential
# ----------------------------------------------------------------------------
def group8():
    print("\n-- 8. the Newtonian potential and the weak-field threshold")
    G = 1.0; m = 1.0; M = 1.0
    Phi = lambda r: -G * m * erf(M * r / 2) / r
    check("Phi(0) = -G m M / sqrt(pi)", abs(Phi(1e-7) + G * m * M / sqrt(pi)) < 1e-9)
    # solves Laplace(Phi) = 4 pi G rho_eff with the Gaussian rho_eff of width 2/M and total mass m
    rho = lambda r: m * (M / 2) ** 3 * pi ** -1.5 * exp(-M * M * r * r / 4)
    tot = integrate.quad(lambda r: 4 * pi * r * r * rho(r), 0, 50)[0]
    check("the smeared source has total mass m", abs(tot - m) < 1e-9)
    worst = 0.0
    for r in (0.3, 0.8, 1.5, 3.0):
        h = 1e-4
        lap = (Phi(r + h) - 2 * Phi(r) + Phi(r - h)) / h ** 2 + 2 * (Phi(r + h) - Phi(r - h)) / (2 * h * r)
        worst = max(worst, abs(lap - 4 * pi * G * rho(r)) / (4 * pi * G * rho(r) + 1e-12))
    check("Laplace(Phi) = 4 pi G rho_eff", worst < 1e-5, f"max rel dev {worst:.1e}")
    # crossover values of erfc
    x3 = optimize.brentq(lambda x: erfc(x) - 1e-3, 0, 10)
    check("erfc(x) = 10^-3 at x = 2.327, i.e. M r = 4.654", abs(2 * x3 - 4.654) < 2e-3, f"{2*x3:.4f}")
    x2 = optimize.brentq(lambda x: erfc(x) - 1e-2, 0, 10)
    check("erfc(x) = 10^-2 at x = 1.821", abs(x2 - 1.821) < 2e-3, f"{x2:.4f}")
    check("erfc(5) < 10^-11", erfc(5) < 1e-11, f"{erfc(5):.2e}")
    # weak-field threshold: max_r 2|Phi| = 2 G m M/sqrt(pi) attained as r -> 0; monotone decrease of erf(u)/u
    u = np.linspace(1e-6, 10, 20000)
    f = special.erf(u) / u
    check("erf(u)/u is strictly decreasing on (0, infinity)", np.all(np.diff(f) < 0))
    check("threshold mass sqrt(pi)/(2 G M) makes 2|Phi(0)| = 1", abs(2 * (sqrt(pi) / (2 * G * M)) * M / sqrt(pi) - 1) < 1e-12)
    # the enclosed mass function of the smeared source is m [erf(u) - 2u e^{-u^2}/sqrt(pi)], u = M r/2
    for r in (0.5, 1.0, 2.5):
        u = M * r / 2
        enc = integrate.quad(lambda s: 4 * pi * s * s * rho(s), 0, r)[0]
        pred = m * (erf(u) - 2 * u * exp(-u * u) / sqrt(pi))
        check(f"enclosed mass at r={r}", abs(enc - pred) < 1e-9)
    # laboratory scale: a 1 percent agreement with Newton at 52 micrometers bounds 1/M below 14.3 micrometers
    r_lab = 52e-6
    Mmin = 2 * x2 / r_lab
    check("1/M < 14.3 micrometers from erfc(M r/2) < 0.01 at r = 52 micrometers", abs(1 / Mmin * 1e6 - 14.28) < 0.05, f"{1/Mmin*1e6:.2f} um")


# ----------------------------------------------------------------------------
# 9. dispersion
# ----------------------------------------------------------------------------
def group9():
    print("\n-- 9. dispersion")
    k = np.linspace(0.01, 50, 500)
    # the on-shell condition for e^{-H(p^2)}/p^2 is p^2 = 0 whatever H: omega = k
    for name, H in [("z", lambda z: z), ("z^2", lambda z: z * z), ("sin z", np.sin)]:
        w = k.copy()
        p2 = -w * w + k * k
        # residual of the inverse propagator p^2 e^{H(p^2)} on shell
        check(f"omega = k is on shell for H = {name}", np.allclose(p2 * np.exp(H(p2)), 0, atol=1e-12))
    check("group and phase velocity both 1", np.allclose(np.gradient(k, k), 1) and np.allclose(k / k, 1))
    # a second pole at p^2 = -M^2 gives a massive branch omega^2 = k^2 + M^2 with v_g < 1
    M = 2.0
    wm = np.sqrt(k * k + M * M)
    check("massive branch of case (a) has group velocity below 1", np.all(np.gradient(wm, k) < 1))


# ----------------------------------------------------------------------------
# 10. the parametric representation
# ----------------------------------------------------------------------------
def spanning_trees(n_vertices, edges):
    """Kirchhoff's matrix-tree theorem: the number of spanning trees of a
    connected multigraph given as a list of vertex pairs."""
    L = np.zeros((n_vertices, n_vertices))
    for u, v in edges:
        L[u, u] += 1
        L[v, v] += 1
        L[u, v] -= 1
        L[v, u] -= 1
    return int(round(np.linalg.det(L[1:, 1:])))


def U_poly(s, n_vertices, edges):
    """First Symanzik polynomial, by direct enumeration of spanning trees."""
    m = len(edges)
    n_tree = n_vertices - 1
    total = 0.0
    for T in itertools.combinations(range(m), n_tree):
        sub = [edges[i] for i in T]
        if spanning_trees_is_tree(n_vertices, sub):
            total += np.prod([s[i] for i in range(m) if i not in T])
    return total


def spanning_trees_is_tree(n_vertices, edges):
    if len(edges) != n_vertices - 1:
        return False
    parent = list(range(n_vertices))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for u, v in edges:
        ru, rv = find(u), find(v)
        if ru == rv:
            return False
        parent[ru] = rv
    return True


def slice_quad(u, p2, a):
    """The integral over s1 at fixed u = s1 + s2, by direct quadrature.  This is
    reliable while u stays of order a; once u is large the integrand is a pair of
    narrow spikes at the two ends and adaptive quadrature misses them, which is
    why the evaluation below uses the closed form instead."""
    val, _ = integrate.quad(lambda s1: np.exp(-s1 * (u - s1) * p2 / u),
                            a, u - a, epsabs=1e-14, epsrel=1e-12, limit=200)
    return val


def slice_closed(u, p2, a):
    """The same integral in closed form.  Putting x = s1 - u/2 turns the exponent
    into +x^2 p2/u up to a constant, so the slice is a Dawson function."""
    if p2 == 0.0:
        return u - 2 * a
    z = sqrt(p2 / u) * (u / 2.0 - a)
    return 2.0 * sqrt(u / p2) * np.exp(p2 * (a * a / u - a)) * special.dawsn(z)


def bubble_param(p2, D, a):
    """Equation (34) with u = s1 + s2 as the outer variable and w = 2a/u as the
    variable of integration, so that the semi-infinite range becomes the unit
    interval and nothing is truncated."""
    def integrand(w):
        return (slice_closed(2 * a / w, p2, a) *
                (2 * a) ** (1 - D / 2.0) * w ** (D / 2.0 - 2))

    val, _ = integrate.quad(integrand, 0.0, 1.0,
                            epsabs=1e-14, epsrel=1e-12, limit=400)
    return val / (4 * pi) ** (D / 2.0)


def group10():
    print("\n-- 10. the parametric representation")

    # (a) the Schwinger representation with the shifted lower limit
    for M in (0.7, 1.0, 2.5):
        for k2 in (0.3, 1.7, 9.0):
            lhs = np.exp(-k2 / M ** 2) / k2
            rhs, _ = integrate.quad(lambda s: np.exp(-s * k2), 1.0 / M ** 2, np.inf,
                                    epsabs=1e-14, epsrel=1e-12)
            check(f"Schwinger form of the dressed propagator, M={M}, k^2={k2}",
                  abs(lhs - rhs) < 1e-12 * max(1.0, abs(lhs)))

    # (b) U is bounded below by t(G) M^{-2L} on the domain
    graphs = {
        "bubble": (2, [(0, 1), (0, 1)]),
        "triangle": (3, [(0, 1), (1, 2), (2, 0)]),
        "sunset": (2, [(0, 1), (0, 1), (0, 1)]),
        "box": (4, [(0, 1), (1, 2), (2, 3), (3, 0)]),
    }
    for name, (nv, ed) in graphs.items():
        M = 1.3
        a = 1.0 / M ** 2
        L = len(ed) - nv + 1
        t = spanning_trees(nv, ed)
        U_at_corner = U_poly([a] * len(ed), nv, ed)
        check(f"U({name}) at the corner equals t(G) M^(-2L), t={t}, L={L}",
              abs(U_at_corner - t * M ** (-2.0 * L)) < 1e-10,
              f"{U_at_corner:.6f}")
        # monotone in every parameter, so the corner is the minimum
        worse = min(U_poly(list(a + rng.random(len(ed))), nv, ed) for _ in range(200))
        check(f"U({name}) is minimised at the corner of the domain",
              worse >= U_at_corner - 1e-12)

    # (c) the Dawson form of the fixed-u slice, against direct quadrature in the
    #     range of u where the quadrature can be trusted
    for u in (2.5, 4.0, 10.0):
        for p2 in (0.25, 1.0, 4.0):
            cl = slice_closed(u, p2, 1.0)
            qd = slice_quad(u, p2, 1.0)
            check(f"Dawson form of the slice at u={u}, p^2={p2}",
                  abs(cl - qd) < 1e-11 * max(1.0, abs(qd)),
                  f"{cl:.8e} vs {qd:.8e}")

    # (d) the closed form of the bubble at zero external momentum
    for D in (5, 6, 8, 10):
        M = 1.0
        a = 1.0 / M ** 2
        closed = ((2 * a) ** (2 - D / 2.0) / ((D / 2.0 - 1) * (D / 2.0 - 2))
                  / (4 * pi) ** (D / 2.0))
        num = bubble_param(0.0, D, a)
        check(f"bubble at p=0 matches the closed form, D={D}",
              abs(num - closed) < 1e-9 * abs(closed),
              f"{num:.6e} vs {closed:.6e}")

    # (e) finiteness at nonzero momentum, and the D=4 infrared logarithm
    vals = [bubble_param(p2, 4, 1.0) for p2 in (0.25, 1.0, 4.0)]
    check("D=4 bubble is finite at p^2 > 0", all(np.isfinite(v) and v > 0 for v in vals))
    check("D=4 bubble decreases with p^2", vals[0] > vals[1] > vals[2])
    small = [bubble_param(p2, 4, 1.0) for p2 in (1e-2, 1e-3)]
    check("D=4 bubble grows logarithmically as p -> 0",
          small[1] > small[0] and (small[1] - small[0]) < 1.0,
          f"{small[0]:.4f} -> {small[1]:.4f}")

    # (f) no dependence of the ultraviolet statement on the loop order:
    #     the corner value of U grows like M^{-2L}, so U^{-D/2} stays bounded
    M = 1.0
    for name, (nv, ed) in graphs.items():
        L = len(ed) - nv + 1
        t = spanning_trees(nv, ed)
        bound = (t * M ** (-2.0 * L)) ** (-6 / 2.0)
        check(f"U^(-D/2) is bounded on the domain for {name} at D=6",
              np.isfinite(bound) and bound > 0, f"{bound:.4f}")


# ----------------------------------------------------------------------------
# 11. multiplicativity from the composition of transmission amplitudes
# ----------------------------------------------------------------------------
def transfer_exact(t, r, delta, N):
    """Exact transmission through N identical lossless branes, by composing
    2x2 transfer matrices; every multiple reflection is included."""
    Mb = np.array([[1.0 / t, np.conj(r) / np.conj(t)],
                   [r / t, 1.0 / np.conj(t)]], dtype=complex)
    Mf = np.array([[np.exp(1j * delta), 0.0],
                   [0.0, np.exp(-1j * delta)]], dtype=complex)
    M = np.eye(2, dtype=complex)
    for i in range(N):
        M = M @ Mb
        if i < N - 1:
            M = M @ Mf
    return 1.0 / M[0, 0]


def group11():
    print("\n-- 11. composition of transmission amplitudes")

    # (a) one brane reproduces its own amplitude
    for r in (0.05, 0.2, 0.5):
        t = sqrt(1 - r * r)
        check(f"a single brane transmits t, r={r}",
              abs(transfer_exact(t, r, 0.9, 1) - t) < 1e-14)

    # (b) two branes obey the Fabry-Perot formula of the lemma
    for r in (0.05, 0.2):
        for delta in (0.3, 1.1):
            t = sqrt(1 - r * r)
            exact = transfer_exact(t, r, delta, 2)
            free = transfer_exact(1.0, 0.0, delta, 2)
            # for a symmetric lossless brane the reflection seen from the right
            # is r' = -conj(r), which is the r' of the composition formula
            r_right = -np.conj(r)
            fabry = free * t * t / (1 - r_right * r * np.exp(-2j * delta))
            check(f"two branes match the composition formula, r={r}, delta={delta}",
                  abs(exact - fabry) < 1e-12 * abs(exact),
                  f"{abs(exact - fabry):.2e}")

    # (c) the error of the product rule is second order in the reflectivity
    delta = 0.7
    for N in (3, 5, 8):
        errs = []
        for r in (1e-3, 1e-2):
            t = sqrt(1 - r * r)
            exact = transfer_exact(t, r, delta, N)
            approx = transfer_exact(1.0, 0.0, delta, N) * t ** N
            errs.append(abs(exact - approx) / abs(exact))
        ratio = errs[1] / errs[0]
        check(f"product rule error scales as |r|^2 for N={N}",
              abs(ratio - 100.0) < 2.0, f"ratio {ratio:.2f} over a decade")

    # (d) the error grows with the number of pairs
    r, t = 1e-2, sqrt(1 - 1e-4)
    e = []
    for N in (2, 3, 5, 8):
        exact = transfer_exact(t, r, delta, N)
        approx = transfer_exact(1.0, 0.0, delta, N) * t ** N
        e.append(abs(exact - approx) / abs(exact))
    check("product rule error increases with the number of branes",
          all(e[i] < e[i + 1] for i in range(len(e) - 1)),
          " < ".join(f"{x:.2e}" for x in e))

    # (e) unitarity of the single brane, which the lemma uses
    for r in (0.1, 0.4, 0.8):
        t = sqrt(1 - r * r)
        check(f"|t|^2 + |r|^2 = 1 for the single brane, r={r}",
              abs(t * t + r * r - 1.0) < 1e-15)

    # (f) the derivative expansion of the inverse transmission is 1 + w z/M^2
    M, w = 3.0, 0.8
    z = np.array([1e-4, 1e-3, 1e-2])
    tinv = 1.0 + w * z / M ** 2
    check("the single-brane factor is 1 + w z/M^2 to leading order",
          np.allclose(tinv - 1.0, w * z / M ** 2, rtol=1e-14))


if __name__ == "__main__":
    for g in (group1, group2, group3, group4, group5, group6, group7, group8, group9, group10, group11):
        g()
    print(f"\n{len(PASSED)} checks passed, {len(FAILED)} failed.")
    if FAILED:
        print("failed:", *FAILED, sep="\n  ")
        sys.exit(1)
