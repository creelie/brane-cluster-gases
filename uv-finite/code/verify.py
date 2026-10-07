"""Numerical cross-checks for the manuscript "Finite quantum gravity from a
scale-invariant gas of branes" (D. Bhattacharjee).

Every statement checked here is proved analytically in the paper.  This script
only re-evaluates the identities, inequalities and distributions at sample
points, and samples realizations of the gas by Monte Carlo, so that a reader can
confirm the algebra independently.  Needs numpy, scipy, mpmath.

Run:  python3 verify.py
"""
import cmath
import itertools
import math

import mpmath as mp
import numpy as np
from scipy import integrate, optimize, special

rng = np.random.default_rng(20261007)
GAMMA = float(mp.euler)
results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


def Ein(y):
    """entire exponential integral for real y >= 0, via Ein(y) = gamma + log y + E1(y)"""
    return float(mp.euler + mp.log(y) + mp.e1(y)) if y > 0 else 0.0


def Ein_series(y, terms=200):
    return float(mp.nsum(lambda k: (-1) ** (k + 1) * mp.mpf(y) ** k / (k * mp.factorial(k)), [1, mp.inf]))


# 1. Euler's constant as the difference of the two integrals (Lemma 2).
g = mp.quad(lambda t: (1 - mp.e ** (-t)) / t, [0, 1]) - mp.quad(lambda t: mp.e ** (-t) / t, [1, mp.inf])
check("gamma = int_0^1 (1-e^-t)/t - int_1^inf e^-t/t", abs(g - mp.euler) < 1e-25)

# 2. 0 <= Ein(y) - log(1+y) <= gamma, increasing (Lemma 2).
ys = np.concatenate([np.linspace(0, 5, 51)[1:], np.logspace(0.7, 3, 40)])
diffs = [float(mp.quad(lambda t: (1 - mp.e ** (-t)) / t, [0, y])) - math.log1p(y) for y in ys]
check("0 <= Ein(y)-log(1+y) <= gamma", all(0 <= d <= GAMMA + 1e-12 for d in diffs))
check("Ein(y)-log(1+y) increasing", all(b >= a - 1e-13 for a, b in zip(diffs, diffs[1:])))
y = 40.0
lhs = float(mp.quad(lambda t: (1 - mp.e ** (-t)) / t, [0, y]))
check("Ein(y) = log y + gamma + E1(y)", abs(lhs - (math.log(y) + GAMMA + special.exp1(y))) < 1e-12)

# 3. Phi(x) = int_0^1 phi(x tau) dtau/tau = Ein(x^2)/2 for phi(w) = 1 - exp(-w^2) (Prop. 2).
for x in [0.3, 1.0, 2.5, -1.7, 6.0]:
    val = integrate.quad(lambda tau: (1 - math.exp(-(x * tau) ** 2)) / tau, 0, 1, limit=200)[0]
    ref = 0.5 * float(mp.quad(lambda t: (1 - mp.e ** (-t)) / t, [0, x * x]))
    check(f"Phi({x}) = Ein(x^2)/2", abs(val - ref) < 1e-10)

# complex z: H(z) via the power series n*sum c_k (z/M^2)^k / k (Theorem 1)
n = 6
for zc in [0.7 + 0.4j, -1.2 + 0.9j, 2.0 - 1.5j]:
    direct = n * mp.quad(lambda tau: (1 - mp.e ** (-(zc * tau) ** 2)) / tau, [0, 1])
    series = n * 0.5 * mp.nsum(lambda k: (-1) ** (k + 1) * (zc * zc) ** k / (k * mp.factorial(k)), [1, mp.inf])
    check(f"H({zc}) integral = series", abs(direct - series) < 1e-12)

# 4. Probability generating functional, Monte Carlo: E prod(1 + f) = exp int f dnu (Eq. pgfl).
#    Gas on scales s in (eps, S], intensity n ds/s, f(s) = 1 - exp(-(s z)^2).
S, eps, z = 1.0, 1e-4, 1.3
lam = n * math.log(S / eps)
N_real = 200000
counts = rng.poisson(lam, N_real)
tot = counts.sum()
logs = np.log(eps) + rng.random(tot) * math.log(S / eps)
svals = np.exp(logs)
fac = 1 + (1 - np.exp(-(svals * z) ** 2))
idx = np.repeat(np.arange(N_real), counts)
prod = np.ones(N_real)
np.multiply.at(prod, idx, fac)
mc = prod.mean()
exact = math.exp(n * 0.5 * (float(mp.quad(lambda t: (1 - mp.e ** (-t)) / t, [0, (S * z) ** 2]))
                             - float(mp.quad(lambda t: (1 - mp.e ** (-t)) / t, [0, (eps * z) ** 2]))))
se = prod.std() / math.sqrt(N_real)
check("PGFL: E prod(1+f) = exp int f dnu (MC)", abs(mc - exact) < 5 * se, f"mc={mc:.5f} exact={exact:.5f} se={se:.1e}")

# 5. Two-sided bound (1+x^2)^{n/2} <= a <= e^{n gamma/2}(1+x^2)^{n/2} on the real axis (Prop. 2).
ok = True
for x in np.concatenate([-np.logspace(-2, 2, 40), np.logspace(-2, 2, 40)]):
    a = math.exp(n * 0.5 * float(mp.quad(lambda t: (1 - mp.e ** (-t)) / t, [0, x * x])))
    lo, hi = (1 + x * x) ** (n / 2), math.exp(n * GAMMA / 2) * (1 + x * x) ** (n / 2)
    ok &= lo * (1 - 1e-12) <= a <= hi * (1 + 1e-12)
check("two-sided bound on a(z), all real z", ok)

# 6. Spin projectors and linearized curvature (Lemma 3), D = 4 and D = 5.
def projectors(k, D):
    eta = np.eye(D)
    k2 = k @ k
    om = np.outer(k, k) / k2
    th = eta - om
    P2 = 0.5 * (np.einsum('ma,nb->mnab', th, th) + np.einsum('mb,na->mnab', th, th)) \
        - np.einsum('mn,ab->mnab', th, th) / (D - 1)
    P0s = np.einsum('mn,ab->mnab', th, th) / (D - 1)
    return th, om, P2, P0s


def lin_ricci(h, k):
    # Fourier transform of (1/2)(d_r d_m h^r_n + d_r d_n h^r_m - box h_mn - d_m d_n h), d -> i k
    tr = np.trace(h)
    k2 = k @ k
    kh = k @ h
    return 0.5 * (-np.outer(k, kh) - np.outer(kh, k) + k2 * h + np.outer(k, k) * tr)


for D in (4, 5):
    k = rng.normal(size=D)
    k2 = k @ k
    th, om, P2, P0s = projectors(k, D)
    A = rng.normal(size=(D, D)); h = A + A.T
    Ric = lin_ricci(h, k)
    R = np.trace(Ric)
    hP2h = np.einsum('mn,mnab,ab->', h, P2, h)
    hP0h = np.einsum('mn,mnab,ab->', h, P0s, h)
    check(f"D={D}: Ric.Ric = k^4/4 (hP2h + D hP0h)", abs(np.sum(Ric * Ric) - k2 ** 2 / 4 * (hP2h + D * hP0h)) < 1e-9 * k2 ** 2 * np.sum(h * h))
    check(f"D={D}: R^2 = k^4 (D-1) hP0h", abs(R * R - k2 ** 2 * (D - 1) * hP0h) < 1e-9 * k2 ** 2 * np.sum(h * h))
    Gein = Ric - 0.5 * np.eye(D) * R
    check(f"D={D}: h.G1 = k^2/2 (hP2h - (D-2) hP0h)", abs(np.sum(h * Gein) - k2 / 2 * (hP2h - (D - 2) * hP0h)) < 1e-9 * k2 * np.sum(h * h))
    xi = rng.normal(size=D)
    hg = np.outer(k, xi) + np.outer(xi, k)
    check(f"D={D}: linearized Ricci gauge invariant", np.max(np.abs(lin_ricci(hg, k))) < 1e-12 * k2 * np.max(np.abs(hg)))
    T = th / (D - 1) + om
    P2h = np.einsum('mnab,ab->mn', P2, h)
    pred = 0.5 * k2 * (P2h + np.trace(th @ h) * T)
    check(f"D={D}: Ric = k^2/2 [P2 h + (theta.h) T]", np.max(np.abs(Ric - pred)) < 1e-10 * k2 * np.max(np.abs(h)))

# 7. Angular averages and the killer matrix (Prop. 9), D = 4.
D = 4
eta = np.eye(D)


def avg_kk(f):
    """exact average over unit k in D dims of a function polynomial of degree <= 4 in khat,
    using <k_i k_j> = d_ij/D and <k_i k_j k_l k_m> = (dd+dd+dd)/(D(D+2))."""
    return f


Rb = rng.normal(size=(D, D)); Rb = Rb + Rb.T
Ric2 = np.sum(Rb * Rb); Rs = np.trace(Rb)
# Monte Carlo over the sphere with many points, plus exact formula
pts = rng.normal(size=(400000, D)); pts /= np.linalg.norm(pts, axis=1, keepdims=True)
kRk = np.einsum('pi,ij,pj->p', pts, Rb, pts)
kRRk = np.einsum('pi,ij,jl,pl->p', pts, Rb, Rb, pts)
trTRTR = Ric2 - 2 * kRRk + kRk ** 2
trTR = Rs - kRk
X_mc = np.mean(trTRTR - trTR ** 2 / (D - 1))
Y_mc = np.mean((trTR / (D - 1) + kRk) ** 2)
X_ex = 5 / 9 * Ric2 - 5 / 36 * Rs ** 2
Y_ex = 13 / 54 * Rs ** 2 + Ric2 / 27
scale = Ric2 + Rs ** 2
check("<P2 RR> = 5/9 Ric2 - 5/36 R^2", abs(X_mc - X_ex) < 1e-2 * scale, f"mc={X_mc:.4f} ex={X_ex:.4f}")
check("<(T.R)^2> = 13/54 R^2 + Ric2/27", abs(Y_mc - Y_ex) < 1e-2 * scale, f"mc={Y_mc:.4f} ex={Y_ex:.4f}")
# exact averaging via isotropic tensors
I2 = eta / D
I4 = (np.einsum('ij,kl->ijkl', eta, eta) + np.einsum('ik,jl->ijkl', eta, eta) + np.einsum('il,jk->ijkl', eta, eta)) / (D * (D + 2))
avg_kRk = np.einsum('ij,ij->', I2, Rb)
avg_kRRk = np.einsum('ij,ij->', I2, Rb @ Rb)
avg_kRk2 = np.einsum('ijkl,ij,kl->', I4, Rb, Rb)
X_iso = Ric2 - 2 * avg_kRRk + avg_kRk2 - (Rs ** 2 - 2 * Rs * avg_kRk + avg_kRk2) / (D - 1)
Y_iso = Rs ** 2 / (D - 1) ** 2 + 2 * Rs * avg_kRk * (1 / (D - 1)) * (1 - 1 / (D - 1)) + avg_kRk2 * (1 - 1 / (D - 1)) ** 2
check("exact isotropic average X", abs(X_iso - X_ex) < 1e-10 * scale)
check("exact isotropic average Y", abs(Y_iso - Y_ex) < 1e-10 * scale)
check("X - (3/2) Y = (Ric2 - R^2)/2", abs(X_ex - 1.5 * Y_ex - 0.5 * (Ric2 - Rs ** 2)) < 1e-12 * scale)
# <dR dR> from <dRic dRic>: eta.eta contraction
th, om, P2, P0s = projectors(np.array([0.3, -1.1, 0.7, 0.2]), D)
T = th / 3 + om
check("eta P2 eta = 0, (T.eta)^2 = 4", abs(np.einsum('mn,mnab,ab->', eta, P2, eta)) < 1e-12 and abs(np.trace(T) ** 2 - 4) < 1e-12)
# killer contributions in units of 1/c: dbeta1 = -(12 s1 + s2), dbeta2 = s2
A = np.array([[-12.0, -1.0], [0.0, 1.0]])
check("killer matrix d(beta1,beta2)/d(s1,s2) has det -12", abs(np.linalg.det(A) + 12) < 1e-12)
# assemble the coefficients from the correlators: 8*(-3/2) for s1 R^2, 8*(1/4)*(1/2) for s2 (Ric2 - R^2)
check("s1 coefficient 8*(-3/2) = -12", 8 * (-1.5) == -12)
check("s2 coefficient 8*(1/4)*(1/2) = 1", 8 * 0.25 * 0.5 == 1)
b1, b2 = 0.731, -2.4  # arbitrary stand-ins for beta_1^(0), beta_2^(0)
c = 1.0
s2 = -c * b2; s1 = c * (b1 + b2) / 12
check("tuning cancels both poles", abs(b1 - (12 * s1 + s2) / c) < 1e-14 and abs(b2 + s2 / c) < 1e-14)

# 8. Divided differences: Hermite-Genocchi and the symbol bound (Lemmas 5, 6).
def divdiff(f, zs):
    zs = list(zs)
    if len(zs) == 1:
        return f(zs[0])
    return (divdiff(f, zs[1:]) - divdiff(f, zs[:-1])) / (zs[-1] - zs[0])


def G_of(z, n=6):
    # G(z) = (a(z)-1)/z - c z^{n-1}, M = 1, c = e^{n gamma/2}
    a = mp.e ** (n * mp.mpf(0.5) * mp.quad(lambda t: (1 - mp.e ** (-t)) / t, [0, z * z]))
    return (a - 1) / z - mp.e ** (n * mp.euler / 2) * z ** (n - 1)


mp.mp.dps = 60
# The constant K_m of the lemma may be large; what the lemma asserts is that the
# ratio |G[z]| / prod (1+z_i)^-1 stays bounded as the arguments are scaled up,
# including mixed configurations in which some arguments stay small.
patterns = [[0.1, 1.0], [0.1, 1.0, 1.3], [0.2, 1.0, 0.5, 1.7], [1.0, 1.2, 1.5], [0.05, 0.9, 0.07, 1.4]]
for p in patterns:
    ratios = []
    for lam in (1, 10, 100, 1000):
        zs = [mp.mpf(v) * (lam if v > 0.5 else 1) for v in p]
        dd = abs(divdiff(lambda t: G_of(t), zs))
        ratios.append(float(dd) / np.prod([1 / (1 + float(v)) for v in zs]))
    check(f"divided-difference bound stays bounded under scaling {p}", max(ratios[1:]) <= ratios[0] * 1.05 + 10,
          str([f"{r:.3g}" for r in ratios]))
mp.mp.dps = 15
f = lambda t: 1 / (1 + t)
zs = [0.3, 2.0, 7.5, 40.0]
check("f=1/(1+z): divided difference equals (-1)^m prod 1/(1+z_i)",
      abs(divdiff(f, zs) - (-1) ** 3 * np.prod([1 / (1 + v) for v in zs])) < 1e-14)
# Hermite-Genocchi for exp on 3 points
zs = [0.2, 0.9, 1.7]
hg = integrate.dblquad(lambda t2, t1: math.exp(zs[0] + t1 * (zs[1] - zs[0]) + t2 * (zs[2] - zs[1])), 0, 1, 0, lambda t1: t1)[0]
check("Hermite-Genocchi, m = 2", abs(hg - divdiff(math.exp, zs)) < 1e-10)
# complete homogeneous polynomial: (z^k)[z0..zm] = h_{k-m}
def hcomp(deg, zs):
    return sum(np.prod([zs[i] ** e for i, e in enumerate(c)]) for c in itertools.product(range(deg + 1), repeat=len(zs)) if sum(c) == deg)
zs = [0.5, 1.3, 2.2]
check("(z^5)[z0,z1,z2] = h_3(z0,z1,z2)", abs(divdiff(lambda t: t ** 5, zs) - hcomp(3, zs)) < 1e-10)

# 9. Power counting from the vertex bound (Sec. IV D):
#    omega_bar(L) = 4L - (2n+2)(L-1) + 2(L-n-1)_+ < 0 for every L >= 2 iff n >= 4.
def omega_bar(L, nn):
    return 4 * L - (2 * nn + 2) * (L - 1) + 2 * max(L - nn - 1, 0)
for nn in range(2, 12):
    neg = all(omega_bar(L, nn) < 0 for L in range(2, 400))
    check(f"n={nn}: vertex bound gives convergence for all L>=2: {neg}", neg == (nn >= 4))
check("n=3 is marginal: omega_bar(2) = 0", omega_bar(2, 3) == 0)
table = {3: [4, 0, -4, -8, -10], 4: [4, -2, -8, -14, -20], 5: [4, -4, -12, -20, -28],
         6: [4, -6, -16, -26, -36], 7: [4, -8, -20, -32, -44], 8: [4, -10, -24, -38, -52]}
check("Table III entries of omega_bar(L), n = 3..8, L = 1..5",
      all(omega_bar(L, nn) == table[nn][L - 1] for nn in table for L in range(1, 6)))
# merging inequality: sum_v (h_v - k)_+ <= (sum h_v - 2(V-1) - k)_+ for h_v >= 2, k >= 2 (random)
bad = 0
for _ in range(100000):
    V = int(rng.integers(1, 9)); h = rng.integers(2, 20, size=V); k = int(rng.integers(2, 25))
    bad += sum(max(x - k, 0) for x in h) > max(int(h.sum()) - 2 * (V - 1) - k, 0)
check("merging inequality, 1e5 random vertex sets", bad == 0)

# 10. Potential at the origin: Beta-function bounds (Prop. 10).
for nn in (4, 5, 6, 7, 8):
    J = 0.25 * special.gamma(0.25) * special.gamma(nn / 2 - 0.25) / special.gamma(nn / 2)
    Jn = integrate.quad(lambda k: (1 + k ** 4) ** (-nn / 2), 0, np.inf)[0]
    I = integrate.quad(lambda k: math.exp(-nn * 0.5 * float(mp.quad(lambda t: (1 - mp.e ** (-t)) / t, [0, k ** 4]))), 0, 30, limit=200)[0]
    check(f"n={nn}: J_n = Beta formula", abs(J - Jn) < 1e-9)
    check(f"n={nn}: e^(-n gamma/2) J_n <= int dk/a <= J_n", math.exp(-nn * GAMMA / 2) * J <= I <= J, f"[{math.exp(-nn*GAMMA/2)*J:.4f}, {I:.4f}, {J:.4f}]")

# 10b. Sharper bracket: Ein(w) <= w gives a <= exp((n/2) x^2), hence the Gaussian lower
#      bound K_n = (1/4) Gamma(1/4) (2/n)^(1/4); Wendel's inequality gives J_n <= 2n/(2n-1) K_n.
ws = np.linspace(0, 40, 801)
check("Ein(w) <= w for w >= 0", all(Ein(float(w)) <= w + 1e-14 for w in ws))
for nn in (4, 5, 6, 7, 8, 10, 14):
    J = 0.25 * special.gamma(0.25) * special.gamma(nn / 2 - 0.25) / special.gamma(nn / 2)
    K = 0.25 * special.gamma(0.25) * (2 / nn) ** 0.25
    Kq = integrate.quad(lambda k: math.exp(-nn / 2 * k ** 4), 0, np.inf)[0]
    ein = lambda w: (w - w * w / 4 + w ** 3 / 18 - w ** 4 / 96) if w < 1e-3 else GAMMA + math.log(w) + special.exp1(w)
    I = sum(integrate.quad(lambda k: math.exp(-nn / 2 * ein(k ** 4)), lo, hi, epsabs=1e-13, epsrel=1e-12)[0]
            for lo, hi in ((0, 1), (1, 2), (2, 4), (4, np.inf)))
    check(f"n={nn}: K_n = Gaussian integral", abs(K - Kq) < 1e-10)
    check(f"n={nn}: K_n <= int dk/a <= J_n <= 2n/(2n-1) K_n", K <= I <= J <= 2 * nn / (2 * nn - 1) * K,
          f"[{K:.5f}, {I:.5f}, {J:.5f}, {2*nn/(2*nn-1)*K:.5f}]")
    if nn % 2 == 0:
        m = nn // 2
        prod = math.pi * math.sqrt(2) / 4 * np.prod([1 - 1 / (4 * j) for j in range(1, m)])
        check(f"n={nn}: J_(2m) = (pi sqrt2/4) prod_(j<m) (1 - 1/(4j))", abs(prod - J) < 1e-12)
    # effective source density at the origin: (1/4)Gamma(3/4)(2/n)^(3/4) <= int k^2/a <= (1/4)B(3/4, n/2-3/4)
    I2 = sum(integrate.quad(lambda k: k * k * math.exp(-nn / 2 * ein(k ** 4)), lo, hi, epsabs=1e-13, epsrel=1e-12)[0]
             for lo, hi in ((0, 1), (1, 2), (2, 4), (4, np.inf)))
    lo2 = 0.25 * special.gamma(0.75) * (2 / nn) ** 0.75
    hi2 = 0.25 * special.beta(0.75, nn / 2 - 0.75)
    check(f"n={nn}: Gaussian and Beta bounds on int k^2 dk/a", lo2 <= I2 <= hi2, f"[{lo2:.5f}, {I2:.5f}, {hi2:.5f}]")
# Wendel's inequality on a grid
okw = all((x / (x + s)) ** (1 - s) <= special.gamma(x + s) / (x ** s * special.gamma(x)) <= 1 + 1e-14
          for x in np.linspace(0.3, 30, 60) for s in (0.1, 0.25, 0.5, 0.75, 0.9))
check("Wendel: (x/(x+s))^(1-s) <= Gamma(x+s)/(x^s Gamma(x)) <= 1", okw)

# 11. Newtonian tail: deviation falls faster than any power (Prop. 10).
nn = 6
def inv_a(k):
    return math.exp(-nn * 0.5 * float(mp.quad(lambda t: (1 - mp.e ** (-t)) / t, [0, k ** 4])))
def dev(r):
    # delta(r) = (2/pi) int_0^inf sin(k r) (1 - 1/a(k^2))/k dk ; use 1 - 1/a -> 1 tail analytically
    f = lambda k: (1 - inv_a(k)) / k - (1 / k if k > 8 else 0.0)
    head = integrate.quad(lambda k: math.sin(k * r) * ((1 - inv_a(k)) / k), 0, 8, limit=400)[0]
    tail = 0.5 * math.pi - float(mp.si(8 * r))  # int_8^inf sin(kr)/k dk
    corr = integrate.quad(lambda k: -math.sin(k * r) * inv_a(k) / k, 8, 60, limit=400)[0]
    return 2 / math.pi * (head + tail + corr)
d = [abs(dev(r)) for r in (2.0, 4.0, 8.0, 12.0)]
check("deviation delta(r) from Newton decays rapidly (r M = 2,4,8,12)", d[3] < d[2] < d[0] and d[3] < 1e-3, str([f"{v:.2e}" for v in d]))

# 12. Resonances of the factor 2 - exp(-s^2 z^2) are never real, Eq. (qzeros).
for kk in range(-3, 4):
    w2 = -math.log(2) - 2j * math.pi * kk
    for root in (np.sqrt(w2 + 0j), -np.sqrt(w2 + 0j)):
        check(f"resonance sz={root:.3f} not real, 2-exp(-s^2z^2)=0", abs(root.imag) > 1e-6 and abs(2 - np.exp(-root ** 2)) < 1e-12)

# 13. Passivity forces phi'(0)=0: phi = 1-exp(-w^2) has phi'(0)=0, phi''(0)=2; IR expansion a = 1 + (n/2)x^2 + ...
x = 1e-3
a = math.exp(nn * 0.5 * float(mp.quad(lambda t: (1 - mp.e ** (-t)) / t, [0, x * x])))
check("a(x) = 1 + (n/2) x^2 + O(x^4)", abs((a - 1) / x ** 2 - nn / 2) < 1e-5)

# 14. Exact averages over the unit sphere S^3 with the 600-cell (a spherical 11-design),
#     and the one-loop poles of the three quartic operators (Sec. V C, Appendix B).
def cell600():
    phi = (1 + 5 ** 0.5) / 2
    V = []
    for i in range(4):
        for sgn in (1, -1):
            v = [0.0] * 4; v[i] = sgn; V.append(v)
    for sg in itertools.product((0.5, -0.5), repeat=4):
        V.append(list(sg))
    base = [phi / 2, 0.5, 1 / (2 * phi), 0.0]
    def even(p):
        p = list(p); inv = sum(1 for i in range(4) for j in range(i + 1, 4) if p[i] > p[j]); return inv % 2 == 0
    for perm in itertools.permutations(range(4)):
        if not even(perm):
            continue
        for sg in itertools.product((1, -1), repeat=3):
            v = [0.0] * 4
            vals = [base[0] * sg[0], base[1] * sg[1], base[2] * sg[2], 0.0]
            for i in range(4):
                v[perm[i]] = vals[i]
            V.append(v)
    V = np.unique(np.round(np.array(V), 12), axis=0)
    return V


K600 = cell600()
check("600-cell has 120 unit vertices", K600.shape == (120, 4) and np.allclose(np.linalg.norm(K600, axis=1), 1))
# design property up to degree 8: compare with Gaussian moments, <x1^8> = 105/1920, <x1^4 x2^4> = 9/1920, <x1^2 x2^2 x3^2 x4^2> = 1/1920
m8 = np.mean(K600[:, 0] ** 8); m44 = np.mean(K600[:, 0] ** 4 * K600[:, 1] ** 4); m2222 = np.mean(np.prod(K600 ** 2, axis=1))
check("600-cell reproduces degree-8 sphere moments", abs(m8 - 105 / 1920) < 1e-11 and abs(m44 - 9 / 1920) < 1e-11 and abs(m2222 - 1 / 1920) < 1e-11)


def kn(A, B):
    """Kulkarni-Nomizu product: an algebraic curvature tensor."""
    return (np.einsum('ac,bd->abcd', A, B) + np.einsum('bd,ac->abcd', A, B)
            - np.einsum('ad,bc->abcd', A, B) - np.einsum('bc,ad->abcd', A, B))


Riem = np.zeros((4, 4, 4, 4))
for _ in range(4):
    A = rng.normal(size=(4, 4)); A = A + A.T
    B = rng.normal(size=(4, 4)); B = B + B.T
    Riem += kn(A, B)
# add a Weyl-type piece so that the tensor is generic
Wt = rng.normal(size=(4, 4, 4, 4))
Wt = Wt - Wt.transpose(1, 0, 2, 3); Wt = Wt - Wt.transpose(0, 1, 3, 2); Wt = Wt + Wt.transpose(2, 3, 0, 1)
Wt = Wt - (Wt + Wt.transpose(0, 2, 3, 1) + Wt.transpose(0, 3, 1, 2)) / 3   # remove cyclic part
Riem += 0.3 * Wt
bianchi = Riem + Riem.transpose(0, 2, 3, 1) + Riem.transpose(0, 3, 1, 2)
check("random algebraic curvature tensor obeys all symmetries",
      np.allclose(Riem, -Riem.transpose(1, 0, 2, 3)) and np.allclose(Riem, Riem.transpose(2, 3, 0, 1)) and np.allclose(bianchi, 0, atol=1e-10))
RicB = np.einsum('abad->bd', Riem); RsB = np.trace(RicB)
Riem2 = np.sum(Riem * Riem); RicB2 = np.sum(RicB * RicB)
Ebar = Riem2 - 4 * RicB2 + RsB ** 2


def lin_riem(h, k):
    # (1/2)(d_n d_r h_ms + d_m d_s h_nr - d_m d_r h_ns - d_n d_s h_mr), d -> i k
    return -0.5 * (np.einsum('n,r,ms->mnrs', k, k, h) + np.einsum('m,s,nr->mnrs', k, k, h)
                   - np.einsum('m,r,ns->mnrs', k, k, h) - np.einsum('n,s,mr->mnrs', k, k, h))


# consistency of the linearized Riemann tensor with the linearized Ricci tensor used above
k = rng.normal(size=4); A = rng.normal(size=(4, 4)); h = A + A.T
check("contraction of linearized Riemann gives linearized Ricci", np.allclose(np.einsum('abad->bd', lin_riem(h, k)), lin_ricci(h, k)))
basis = []
for i in range(4):
    for j in range(i, 4):
        E = np.zeros((4, 4)); E[i, j] = E[j, i] = 1.0; basis.append(E)
acc = {"R": 0.0, "Ric": 0.0, "Riem": 0.0}
for kh in K600:
    th, om, P2, P0s = projectors(kh, 4)
    Pi = P2 - 0.5 * P0s                      # gauge-invariant propagator times k^2 a, |k| = 1
    # linear functionals h -> scalar, represented as symmetric tensors v with v.h
    vR = np.zeros((4, 4)); vRic = np.zeros((4, 4)); vRiem = np.zeros((4, 4))
    for E in basis:
        ric = lin_ricci(E, kh); rie = lin_riem(E, kh)
        cR = RsB * np.trace(ric); cRic = np.sum(RicB * ric); cRiem = np.sum(Riem * rie)
        w = 1.0 if np.count_nonzero(E) == 1 else 0.5
        vR += cR * E * w; vRic += cRic * E * w; vRiem += cRiem * E * w
    for key, v in (("R", vR), ("Ric", vRic), ("Riem", vRiem)):
        acc[key] += np.einsum('mn,mnab,ab->', v, Pi, v) / len(K600)
scale = Riem2 + RicB2 + RsB ** 2
check("<(Rbar dR)^2> = -(3/2) Rbar^2", abs(acc["R"] + 1.5 * RsB ** 2) < 1e-10 * scale, f"{acc['R']:.6f} vs {-1.5*RsB**2:.6f}")
check("<(Ricbar.dRic)^2> = (Ric^2 - R^2)/8", abs(acc["Ric"] - (RicB2 - RsB ** 2) / 8) < 1e-10 * scale, f"{acc['Ric']:.6f}")
check("<(Riembar.dRiem)^2> = (3 Riem^2 - R^2)/12", abs(acc["Riem"] - (3 * Riem2 - RsB ** 2) / 12) < 1e-10 * scale, f"{acc['Riem']:.6f}")
# pole coefficients (units 1/(16 pi^2 eps c)): 8 s_i times the averages
col1 = 8 * acc["R"]; col2 = 8 * acc["Ric"]; col3 = 8 * acc["Riem"]
A3 = np.array([[-12.0, -1.0, -8 / 3], [0.0, 1.0, 8.0], [0.0, 0.0, 2.0]])
pred = lambda col: col[0] * RsB ** 2 + col[1] * RicB2 + col[2] * Ebar
check("s1 operator: pole = -12 R^2", abs(col1 - pred(A3[:, 0])) < 1e-9 * scale)
check("s2 operator: pole = Ric^2 - R^2", abs(col2 - pred(A3[:, 1])) < 1e-9 * scale)
check("s3 operator: pole = 2E + 8 Ric^2 - (8/3) R^2", abs(col3 - pred(A3[:, 2])) < 1e-9 * scale)
check("3x3 killer matrix determinant = -24", abs(np.linalg.det(A3) + 24) < 1e-12)
b0 = np.array([0.731, -2.4, 1.37])          # stand-ins for beta_1^(0), beta_2^(0), beta_E^(0)
s3 = -b0[2] / 2; s2 = -b0[1] + 4 * b0[2]; s1 = (b0[0] + b0[1] - 8 / 3 * b0[2]) / 12
check("closed-form tuning cancels all three one-loop poles", np.allclose(b0 + A3 @ np.array([s1, s2, s3]), 0, atol=1e-14))

# 15. Polynomial cone: a(z) = c (z/M^2)^n [1 + O(exp(-Re z^2/M^4))] for |arg z| < pi/4 (Sec. II E)
def Ein_c(w):
    return w * mp.hyp2f2(1, 1, 2, 2, -w)


okc = True
for r in (1.5, 2.5, 4.0):
    for thf in (0.0, 0.3, 0.6, 0.9):
        th = thf * (math.pi / 4)
        zc = mp.mpc(r * math.cos(th), r * math.sin(th))
        a_z = mp.e ** (3 * Ein_c(zc ** 2))          # n = 6
        lead = mp.e ** (3 * mp.euler) * zc ** 6
        w = zc ** 2
        bound = mp.e ** (3 * mp.e ** (-mp.re(w)) / abs(w)) - 1
        okc &= abs(a_z / lead - 1) <= bound * (1 + 1e-10) + 1e-30
check("cone asymptotics of a(z) with explicit error bound (n = 6)", okc)
ai = [float(abs(mp.e ** (3 * Ein_c((mp.mpc(0, y)) ** 2)))) for y in (1.0, 1.5, 2.0)]
check("a(z) decays along the imaginary axis", ai[0] > ai[1] > ai[2] and ai[2] < 1e-10, str([f"{v:.2e}" for v in ai]))

# 16. Gas of resonant branes: zeros in the sectors pi/4 < |arg z| < 3pi/4; expected count n R^2/(pi M^4)
oks = True
for _ in range(2000):
    sv = math.exp(-8 * rng.random()); kk = int(rng.integers(-500, 500))
    zz = cmath_sqrt = np.sqrt(complex(-math.log(2), -2 * math.pi * kk)) / sv
    for zr in (zz, -zz):
        t = abs(math.atan2(zr.imag, zr.real))
        oks &= math.pi / 4 < t < 3 * math.pi / 4
check("resonant-gas zeros avoid the cones |arg z| <= pi/4 and |arg z - pi| <= pi/4", oks)


def expected_zeros(R, n=6.0):
    # E N(R) = n int_0^1 N_s(R) ds/s, N_s = 2 #{k : (log 2)^2 + 4 pi^2 k^2 <= s^4 R^4}
    f = lambda s: 2 * (2 * math.floor(math.sqrt(max(s ** 4 * R ** 4 - math.log(2) ** 2, 0)) / (2 * math.pi)) + 1) \
        if s ** 2 * R ** 2 >= math.log(2) else 0.0
    smin = math.sqrt(math.log(2)) / R
    # the integrand is a step function of u = log s; integrate exactly between its jumps
    jumps = sorted({0.0, math.log(smin)} | {0.25 * math.log((math.log(2) ** 2 + (2 * math.pi * j) ** 2) / R ** 4)
                                             for j in range(0, int(R * R / (2 * math.pi)) + 2)
                                             if (math.log(2) ** 2 + (2 * math.pi * j) ** 2) <= R ** 4})
    tot = 0.0
    for u0, u1 in zip(jumps, jumps[1:]):
        if u1 > u0:
            tot += f(math.exp(0.5 * (u0 + u1))) * (u1 - u0)
    return n * tot


ratio = expected_zeros(60.0) / (6.0 * 60.0 ** 2 / math.pi)
check("expected number of zeros in |z| < R grows like n R^2/(pi M^4)", abs(ratio - 1) < 0.05, f"ratio={ratio:.4f}")

# 17. Low-momentum expansion to fourth order: a = 1 + (n/2)x^2 + n(n-1)/8 x^4 + O(x^6)
for nn_ in (3, 6, 9):
    with mp.workdps(50):
        x = mp.mpf("1e-4")
        a_ = mp.e ** (nn_ * Ein_c(x ** 2) / 2)
        c4 = float((a_ - 1 - nn_ * x ** 2 / 2) / x ** 4)
    check(f"n={nn_}: a = 1 + (n/2)x^2 + n(n-1)/8 x^4 + ...", abs(c4 - nn_ * (nn_ - 1) / 8) < 1e-6)

# 18. Growth exponent = strength per e-fold: m branes per e-fold of strength theta give a ~ |x|^{m theta}
for th_ in (0.5, 1.0, 1.7):
    x1, x2 = 30.0, 60.0
    la = lambda x: 6 * th_ * 0.5 * float(Ein_c(mp.mpf(x) ** 2))
    slope = (la(x2) - la(x1)) / math.log(x2 / x1)
    check(f"strength {th_}, 6 branes per e-fold: growth exponent {6*th_}", abs(slope - 6 * th_) < 1e-9)

# 19. Zero-point energy of the graviton (Sec. VI C).  Per momentum the one-loop
#     log-determinant is (1/2)*10 log(p^2 a) - 4 log(p^2 a) + (1/2)*4 log a
#     = log p^2 + 3 log a, and int d^4k Ein(k^4/M^4) is continued from the
#     Mellin strip -1 < Re s < 0, where int_0^inf w^(s-1) Ein(w) dw = Gamma(s)/s.
def mEin(w):
    w = mp.mpf(w)
    return Ein_c(w) if w < 1 else mp.log(w) + mp.euler + mp.e1(w)


p2_, a_ = 2.7, 13.1
K10 = np.eye(10)
# graviton operator p^2 a (1 - 1/2 delta x delta) on symmetric tensors, orthonormal basis
basis = []
for i in range(4):
    for j in range(i, 4):
        e = np.zeros((4, 4)); e[i, j] = e[j, i] = 1.0
        basis.append(e / np.linalg.norm(e))
dvec = np.array([np.sum(b * np.eye(4)) for b in basis])
Kop = p2_ * a_ * (K10 - 0.5 * np.outer(dvec, dvec))
logdet = 0.5 * np.log(abs(np.linalg.det(Kop))) - 4 * np.log(p2_ * a_) + 0.5 * 4 * np.log(a_)
check("one-loop count per momentum: log p^2 + 3 log a", abs(logdet - (np.log(p2_) + 3 * np.log(a_))) < 1e-12)
okm = True
for sv in (-0.3, -0.5, -0.8):
    val = mp.quad(lambda t: mp.e ** (sv * t) * mEin(mp.e ** t), [-400, -100, -20, 0, 2, 5, 10, 20, 40, 80, 160, 400])
    okm &= abs(val - mp.gamma(sv) / sv) < 1e-10
check("Mellin transform of Ein equals Gamma(s)/s on -1 < s < 0", okm)
okw = True
for W in (2.0, 9.0, 35.0):
    lhs = mp.quad(lambda w: mEin(w), [0, W])
    okw &= abs(lhs - (W * mEin(W) - W + 1 - mp.e ** (-W))) < 1e-12 * W * W
check("hard cutoff: int_0^W Ein = W Ein(W) - W + 1 - e^-W", okw)
okl = True
for W in (0.5, 4.0, 60.0):
    lhs = mp.quad(lambda w: mEin(w) * mp.e ** (-w / W), [0, W, 10 * W, mp.inf])
    okl &= abs(lhs - W * mp.log(1 + W)) < 1e-10 * W * W
check("smooth cutoff: int_0^inf Ein(w) e^(-w/W) dw = W log(1+W)", okl)
W = mp.mpf(80)
const_hard = W * mEin(W) - W + 1 - mp.e ** (-W) - W * mp.log(W) - (mp.euler - 1) * W
const_soft = W * mp.log(1 + W) - W * mp.log(W) + mp.mpf(1) / (2 * W)
check("constant term is 1 for both cutoffs (= Gamma(1)/1)", abs(const_hard - 1) < 1e-12 and abs(const_soft - 1) < 1e-4,
      f"[{float(const_hard):.12f}, {float(const_soft):.6f}]")
nn = 6
rho = 1.5 * nn * (2 * math.pi ** 2) * math.gamma(1.0) / 4 / (2 * math.pi) ** 4
check("rho_1 = (3n/2)(pi^2 M^4/2)/(2 pi)^4 = 3n M^4/(64 pi^2)", abs(rho - 3 * nn / (64 * math.pi ** 2)) < 1e-15)

# 15. Round 4: insertion lemma, chain bookkeeping, admissible class, identical branes,
#     cutoff universality, Newton numbers for n = 4, 5.
worst = 0.0
for _ in range(60000):
    r = int(rng.integers(1, 6))
    sc = 10 ** rng.uniform(-2, 5, size=r + 1)
    q = rng.normal(size=4) * sc[0]
    ls = [rng.normal(size=4) * sc[a + 1] for a in range(r)]
    u = rng.random()
    if u < 0.3:
        ls[-1] = -q - sum(ls[:-1]) + rng.normal(size=4)
    elif u < 0.45:
        q = rng.normal(size=4) * 1e-3
    qp = q + sum(ls); sl = sum(np.linalg.norm(l) for l in ls)
    lhs = (np.linalg.norm(q) + sl) ** 2
    rhs = (1 + 2 * r + 2 * r * r) * (1 + np.linalg.norm(q)) * (1 + np.linalg.norm(qp)) * np.prod([1 + np.linalg.norm(l) for l in ls])
    worst = max(worst, lhs / rhs)
check("insertion lemma (|q|+s)^2 <= C_r (1+|q|)(1+|q'|) prod(1+|l_a|)", worst <= 1, f"(worst ratio {worst:.3f})")

# chain with two insertions, n = 4 (Fig. 5b): the full term over the legs stays bounded
mp.mp.dps = 50
def a_minus(z, nn=4):
    z = mp.mpf(z); c = mp.e ** (nn * mp.euler / 2)
    if z * z < 30:
        return mp.e ** (nn * (mp.e1(z * z) + mp.log(z * z) + mp.euler) / 2) - 1 - c * z ** nn
    return c * z ** nn * mp.expm1(nn * mp.e1(z * z) / 2) - 1
ratios = []
for l1, l2 in ((0, 0.0125), (2, 2.0125), (4, 4.0125), (4, 0.0125), (0, 4.0125), (2, 4.0125)):
    lam1, lam2 = 10.0 ** l1, 10.0 ** l2
    q0 = np.array([0, 0, 1.0]); e1v = np.array([lam1, 0, 0]); e2v = np.array([-lam1, lam2, 0])
    q1 = q0 + e1v; q2 = q1 + e2v
    n0, n1, n2 = (np.linalg.norm(v) for v in (q0, q1, q2))
    size = (1 + n0) * (n0 + lam1) ** 2 * (n1 + np.linalg.norm(e2v)) ** 2 * (1 + n2)
    g = abs(divdiff(lambda t: a_minus(t) / t, [mp.mpf(n0 ** 2), mp.mpf(n1 ** 2), mp.mpf(n2 ** 2)]))
    ratios.append(float(g) * size / ((1 + lam1) * (1 + np.linalg.norm(e2v))))
mp.mp.dps = 15
check("chain bookkeeping: term / prod(1+|l_a|) bounded over 4 decades", max(ratios) < 30, str([f"{v:.3g}" for v in ratios]))

# admissible class: C_+, moments, two-sided bounds, and the Mellin continuation behind Eq. (moment)
resp = {"1-e^-w^2": (lambda t: -np.expm1(-t * t), lambda t: np.exp(-t * t), GAMMA / 2, 0.5, lambda s: 0.5 * special.gamma(s / 2)),
        "1-(1+w^2)e^-w^2": (lambda t: 1 - (1 + t * t) * np.exp(-t * t), lambda t: (1 + t * t) * np.exp(-t * t), (GAMMA - 1) / 2, 1.0,
                            lambda s: 0.5 * special.gamma(s / 2) * (1 + s / 2)),
        "1-e^-w^4": (lambda t: -np.expm1(-t ** 4), lambda t: np.exp(-t ** 4), GAMMA / 4, math.sqrt(math.pi) / 4, lambda s: 0.25 * special.gamma(s / 4))}
for name, (phi, om, Cp, mom, mel1m) in resp.items():
    A = integrate.quad(lambda t: phi(t) / t, 0, 1, limit=200)[0]
    B = integrate.quad(lambda t: om(t) / t, 1, np.inf, limit=200)[0]
    m_ = integrate.quad(lambda t: t * om(t), 0, np.inf)[0]
    check(f"{name}: C_+ and int t(1-phi) dt", abs(A - B - Cp) < 1e-9 and abs(m_ - mom) < 1e-9, f"[{A-B:.6f}, {m_:.6f}]")
    xs = np.logspace(-2, 3, 400)
    Phi = np.array([integrate.quad(lambda t: phi(x * t) / t, 0, 1, limit=200)[0] for x in xs])
    g_ = Phi - 0.5 * np.log1p(xs ** 2)
    check(f"{name}: Phi - log(1+x^2)/2 bounded, tends to C_+", np.all(np.abs(g_) < 1) and abs(g_[-1] - Cp) < 1e-4, f"[{g_.min():.4f}, {g_.max():.4f}]")
    sv = -0.5
    lhs = integrate.quad(lambda t: t ** (sv - 1) * phi(t), 0, 1, limit=200)[0] + integrate.quad(lambda t: t ** (sv - 1) * phi(t), 1, np.inf, limit=400)[0]
    check(f"{name}: Mellin of phi in the strip equals the continuation -M[1-phi]", abs(lhs + mel1m(sv)) < 1e-7, f"[{lhs:.8f}, {-mel1m(sv):.8f}]")
    check(f"{name}: continuation at s = 2 gives the moment", abs(mel1m(2.0) - mom) < 1e-12)

# identical branes: Psi_* in closed form, Eq. (Psistar)
for x in (0.3, 1.0, 2.5):
    num = integrate.quad(lambda t: np.expm1(-x * x * t * t) ** 2 / t, 0, 1, limit=200)[0]
    check(f"Psi_*({x}) = Ein(x^2) - Ein(2x^2)/2", abs(num - (Ein(x * x) - 0.5 * Ein(2 * x * x))) < 1e-10)

# cutoff universality: remainder -> 1 for r(u) = exp(-u^p); sharp-cutoff closed form
okc = True
for p in (1.0, 2.0, 3.0):
    r1 = special.gamma(1 + 1 / p); r1p = special.gamma(1 / p) * special.digamma(1 / p) / p ** 2
    W = 800.0
    I_ = integrate.quad(lambda w: (math.log(w) + GAMMA + special.exp1(w) if w > 1e-8 else w) * math.exp(-(w / W) ** p), 0, 40 * W,
                        points=[1, W, 3 * W], limit=500)[0]
    okc &= abs(I_ - r1 * W * math.log(W) - (r1p + GAMMA * r1) * W - 1) < 2e-3
check("remainder -> 1 for the cutoffs exp(-u^p), p = 1, 2, 3", okc)
W = 7.0
direct = float(mp.quad(lambda w: mp.ein(w) if hasattr(mp, "ein") else mp.quad(lambda t: (1 - mp.e ** (-t)) / t, [0, w]), [0, W]))
check("sharp cutoff remainder = 1 - e^-W + W E1(W)", abs(direct - W * math.log(W) - (GAMMA - 1) * W - (1 - math.exp(-W) + W * special.exp1(W))) < 1e-9)

# Newton numbers for n = 4, 5 (Table VIII) and c = e^{n gamma/2}
for nn_, I_ref in ((4, 0.8011), (5, 0.7485)):
    I_ = integrate.quad(lambda k: math.exp(-nn_ / 2 * (Ein(k ** 4) if k > 0 else 0.0)), 0, 30, limit=200)[0]
    check(f"n={nn_}: int dk/a = {I_ref} and |Phi(0)|/GmM = 2I/pi", abs(I_ - I_ref) < 1e-4, f"[{I_:.5f}, {2*I_/math.pi:.4f}]")
check("J_4 = 3 pi sqrt2 / 16", abs(0.25 * special.gamma(0.25) * special.gamma(1.75) / special.gamma(2.0) - 3 * math.pi * math.sqrt(2) / 16) < 1e-12)
check("c = e^{n gamma/2} = 3.172, 4.234, 5.650, 7.540, 10.063 for n = 4..8",
      all(abs(math.exp(k * GAMMA / 2) - v) < 6e-4 for k, v in zip(range(4, 9), (3.172, 4.234, 5.650, 7.540, 10.063))))

# 20. Round 5: one frozen realization of the fine-grained gas (Sec. II F).
#     Power law: strength th0 tau^zeta, intensity nu = (n/th0) tau^(-zeta) dtau/tau on (0, 1],
#     so that th * nu = n dtau/tau.  Response psi_* = 1 - exp(-w^2), units M = 1.
TH0 = math.log(2)


def lgam(b, y):
    """lower incomplete gamma function gamma(b, y)"""
    return special.gammainc(b, y) * special.gamma(b)


def var_R(x, nn, zeta, th0=TH0):      # Eq. (varR)
    return nn * th0 * (1 / zeta - lgam(zeta / 2, x * x) / x ** zeta + 0.5 * lgam(zeta / 2, 2 * x * x) / (2 * x * x) ** (zeta / 2))


def var_f(x, nn, zeta, th0=TH0):      # Eq. (varf)
    return 0.5 * nn * th0 * lgam(zeta / 2, 2 * x * x) / (2 * x * x) ** (zeta / 2)


def EeR(nn, zeta, th0=TH0):           # Eq. (EeR), divided by c-hat
    return math.exp(nn / zeta * integrate.quad(lambda t: (math.expm1(t) - t) / t ** 2 if t > 1e-6 else 0.5 + t / 6, 0, th0)[0])


def sample_gas(rng, m, nn=4, zeta=9.0, tau_c=0.5, th0=TH0):
    """m independent realizations of the branes with tau >= tau_c (exact Poisson sampling)."""
    lam = (nn / th0) * (tau_c ** -zeta - 1) / zeta
    N = rng.poisson(lam, m); A = tau_c ** -zeta
    tau = (A - (A - 1) * rng.random(N.sum())) ** (-1 / zeta)
    return np.repeat(np.arange(m), N), tau, th0 * tau ** zeta


# closed forms against quadrature
okv = True
for zeta in (5.0, 9.0, 13.0):
    for x in (0.5, 1.7, 4.0):
        qR = integrate.quad(lambda t: 4 * TH0 * t ** (zeta - 1) * np.expm1(-(x * t) ** 2) ** 2, 0, 1, limit=200)[0]
        qf = integrate.quad(lambda t: 4 * TH0 * t ** (zeta - 1) * np.exp(-2 * (x * t) ** 2), 0, 1, limit=200)[0]
        okv &= abs(qR / var_R(x, 4, zeta) - 1) < 1e-8 and abs(qf / var_f(x, 4, zeta) - 1) < 1e-8
check("Eqs. (varR), (varf): incomplete-gamma closed forms = quadrature, zeta = 5, 9, 13", okv)
check("Var R_inf = n th0/zeta is the x -> inf limit of Eq. (varR)", abs(var_R(60.0, 4, 9.0) - 4 * TH0 / 9) < 1e-12)
mom = integrate.quad(lambda u: u * math.exp(-u * u), 0, np.inf)[0]
okm = True
for tau in (0.3, 1.0, 2.2):
    lhs = 2 * math.pi ** 2 * integrate.quad(lambda k: k ** 3 * -math.exp(-(tau * k * k) ** 2), 0, np.inf, limit=200)[0] / (2 * math.pi) ** 4
    okm &= abs(lhs + mom / (16 * math.pi ** 2 * tau ** 2)) < 1e-12
check("Eq. (moment1): int d^4k/(2pi)^4 [psi(tau k^2) - 1] = -mu/(16 pi^2 tau^2), mu = 1/2", okm and abs(mom - 0.5) < 1e-12)

# Monte Carlo over realizations of the gas with tau >= 1/2 (about 330 branes each)
nn, zeta, tc = 4, 9.0, 0.5
xs = np.array([1.0, 2.0, 3.0])
Ssum = []; Dsum = []; Esum = []; Rsum = []
for _ in range(10):
    idx, tau, th = sample_gas(rng, 4000, nn, zeta, tc)
    psi = -np.expm1(-(np.outer(tau, xs)) ** 2)
    S = np.stack([np.bincount(idx, weights=th * psi[:, j], minlength=4000) for j in range(3)], 1)
    D = np.stack([np.bincount(idx, weights=th * (psi[:, j] - 1), minlength=4000) for j in range(3)], 1)
    Ssum.append(S); Dsum.append(D)
    Rsum.append(np.bincount(idx, weights=th * tau ** -2, minlength=4000))
S = np.concatenate(Ssum); D = np.concatenate(Dsum); Rr = np.concatenate(Rsum)
mS = [nn * integrate.quad(lambda t: -np.expm1(-(x * t) ** 2) / t, tc, 1)[0] for x in xs]
vS = [nn * TH0 * integrate.quad(lambda t: t ** (zeta - 1) * np.expm1(-(x * t) ** 2) ** 2, tc, 1)[0] for x in xs]
vD = [nn * TH0 * integrate.quad(lambda t: t ** (zeta - 1) * np.exp(-2 * (x * t) ** 2), tc, 1)[0] for x in xs]
M_ = S.shape[0]
okc = all(abs(S[:, j].mean() - mS[j]) < 5 * math.sqrt(vS[j] / M_) for j in range(3))
check("Campbell: E sum th psi(x tau) = n int psi dtau/tau (MC, 4e4 gases)", okc)
okI = all(abs(S[:, j].var() / vS[j] - 1) < 0.05 and abs(D[:, j].var() / vD[j] - 1) < 0.05 for j in range(3))
check("isometry: Var of the sums = int th^2 psi^2 dnu and Eq. (fdef) variance (MC)", okI,
      str([f"{S[:, j].var()/vS[j]:.3f}/{D[:, j].var()/vD[j]:.3f}" for j in range(3)]))
x = 2.0
EeS = np.exp(S[:, 1]).mean()
ref = math.exp(nn / TH0 * integrate.quad(lambda t: t ** -zeta * np.expm1(TH0 * t ** zeta * -np.expm1(-(x * t) ** 2)) / t, tc, 1)[0])
check("E a_Pi = a-hat exp int (e^{th psi} - 1 - th psi) dnu, not a-hat (MC)", abs(EeS / ref - 1) < 0.01 and EeS / math.exp(mS[1]) > 1.0005,
      f"[{EeS/ref:.4f}, excess {EeS/math.exp(mS[1]):.5f}]")
vrho = nn * TH0 * integrate.quad(lambda t: t ** (zeta - 5), tc, 1)[0]
check("Var of sum th(s)/(sM^2)^2 = n th0 int tau^(zeta-5) dtau (fluctuation of the zero-point energy, MC)",
      abs(Rr.var() / vrho - 1) < 0.05, f"[{Rr.var()/vrho:.4f}]")

# the leading coefficient: R_inf as a compensated sum of jumps with Levy measure (n/zeta) dt/t^2 on (0, th0]
okq = True; info = []
for nn_ in (4, 6, 8):
    zeta_ = 2 * nn_ + 1; k_ = nn_ / zeta_; e_ = 1e-3
    lam_ = k_ * (1 / e_ - 1 / TH0)
    out = []
    for _ in range(10):
        N = rng.poisson(lam_, 10000)
        t = 1 / (1 / e_ - rng.random(N.sum()) * (1 / e_ - 1 / TH0))
        Sx = np.bincount(np.repeat(np.arange(10000), N), weights=t, minlength=10000) - k_ * math.log(TH0 / e_)
        out.append(Sx + rng.standard_normal(10000) * math.sqrt(k_ * e_))   # jumps below e_: Gaussian limit
    Rv = np.concatenate(out)
    bound95 = math.exp(-math.sqrt(2 * nn_ * TH0 / zeta_ * math.log(20)))
    okq &= abs(Rv.var() / (nn_ * TH0 / zeta_) - 1) < 0.03 and abs(np.exp(Rv).mean() / EeR(nn_, zeta_) - 1) < 0.01
    okq &= (np.exp(Rv) < bound95).mean() <= 0.05
    info.append(f"n={nn_}: var {Rv.var():.4f}, E e^R {np.exp(Rv).mean():.4f}, P(<{bound95:.3f}) {(np.exp(Rv) < bound95).mean():.4f}")
check("R_inf: variance n th0/zeta, Eq. (EeR), lower tail (lowertail) holds (MC, 1e5 each)", okq, "; ".join(info))

# identical branes (zeta = 0): random walk in log x, characteristic function -> 0
nn, tm = 4, 1e-4
lam0 = nn / TH0 * math.log(1 / tm)
xs0 = np.array([3.0, 30.0, 300.0])
vals = []
for _ in range(10):
    N = rng.poisson(lam0, 5000)
    tau = np.exp(rng.uniform(math.log(tm), 0, N.sum())); idx = np.repeat(np.arange(5000), N)
    vals.append(np.stack([np.bincount(idx, weights=TH0 * -np.expm1(-(tau * x) ** 2), minlength=5000) for x in xs0], 1))
V = np.concatenate(vals)
V -= np.array([nn * integrate.quad(lambda t: -np.expm1(-(x * t) ** 2) / t, tm, 1, points=[1 / x], limit=200)[0] for x in xs0])
pred = [nn * TH0 * (Ein(x * x) - 0.5 * Ein(2 * x * x)) for x in xs0]
check("identical branes: Var R_Pi(x) = n th0 Psi_*(x), x = 3, 30, 300 (MC)", all(abs(V[:, j].var() / pred[j] - 1) < 0.05 for j in range(3)),
      str([f"{V[:, j].var():.3f}/{pred[j]:.3f}" for j in range(3)]))
check("Psi_*(x) = log x + (gamma - log 2)/2 + O(e^{-x^2})",
      all(abs(Ein(x * x) - 0.5 * Ein(2 * x * x) - math.log(x) - 0.5 * (GAMMA - math.log(2))) < 2 * math.exp(-x * x) / x ** 2 for x in (2.0, 3.0, 5.0)))
u = 2.0
cf = [abs(np.exp(1j * u * V[:, j]).mean()) for j in range(3)]
cfp = [math.exp(nn / TH0 * integrate.quad(lambda t: (math.cos(u * TH0 * -math.expm1(-t * t)) - 1) / t, 0, x, limit=400)[0]) for x in xs0]
check("identical branes: |E e^{iuR}| = exp[(n/th0) int (cos(u th0 psi) - 1) dt/t] -> 0 (MC, u = 2)",
      all(abs(cf[j] - cfp[j]) < 0.02 for j in range(3)) and cfp[2] < cfp[1] < cfp[0] < 1, str([f"{a_:.3f}/{b_:.3f}" for a_, b_ in zip(cf, cfp)]))
slope_cf = (math.log(cfp[2]) - math.log(cfp[1])) / math.log(10)
check("identical branes: |E e^{iuR}| ~ x^{(n/th0)(cos(u th0) - 1)}", abs(slope_cf - nn / TH0 * (math.cos(u * TH0) - 1)) < 1e-6)

# mean-square decay in the cones, Eq. (msdecay): slope -zeta in |z| along several rays
okd = True
for zeta in (5.0, 9.0):
    for ang in (0.0, math.pi / 8, -math.pi / 6):
        vv = []
        for r in (20.0, 40.0):
            w = r * cmath.exp(1j * ang)
            vv.append(4 * TH0 * float(mp.quad(lambda t: t ** (zeta - 1) * abs(mp.e ** (-(w * t) ** 2)) ** 2, [0, 1 / r, 5 / r, 1])))
        okd &= abs(math.log(vv[1] / vv[0]) / math.log(2) + zeta) < 1e-6
check("Eq. (msdecay): E|R_Pi - R_inf|^2 falls like |z|^-zeta along rays in the cone", okd)

# one realization in the cone (small branes in their Gaussian limit): almost-sure slope -zeta/2, Eq. (asdecay)
def realization(rng_, nn=4, zeta=9.0, tau_c=0.5, tau_min=1e-7, h=0.01):
    lam = (nn / TH0) * (tau_c ** -zeta - 1) / zeta
    N = rng_.poisson(lam); A = tau_c ** -zeta
    tau = (A - (A - 1) * rng_.random(N)) ** (-1 / zeta)
    L = np.arange(math.log(tau_min), math.log(tau_c), h) + h / 2
    tg = np.exp(L)
    return dict(tau=tau, th=TH0 * tau ** zeta, tg=tg, sg=np.sqrt(nn * TH0 * tg ** zeta * h) * rng_.standard_normal(len(L)), nn=nn, tc=tau_c)


def f_real(G, w):
    """R_Pi(w) - R_inf for psi_*, w complex array"""
    w = np.asarray(w, dtype=complex).ravel()
    out = -(G['th'][None, :] * np.exp(-(G['tau'][None, :] * w[:, None]) ** 2)).sum(1)
    s = np.linspace(math.log(G['tc']), 0, 801)
    out += G['nn'] * integrate.trapezoid(np.exp(-(np.exp(s)[None, :] * w[:, None]) ** 2), s, axis=1)
    out -= (G['sg'][None, :] * np.exp(-(G['tg'][None, :] * w[:, None]) ** 2)).sum(1)
    return out


oka = True; info = []
for zeta in (9.0, 5.0):
    G = realization(np.random.default_rng(3), zeta=zeta)
    sups = []; supa = []
    for k in range(3, 10):
        r = np.linspace(2 ** k, 2 ** (k + 1), 16); a_ = np.linspace(-math.pi / 4 + 0.1, math.pi / 4 - 0.1, 17)
        W = (r[:, None] * np.exp(1j * a_[None, :])).ravel()
        sups.append(np.abs(f_real(G, W)).max())
        xr = np.linspace(2 ** k, 2 ** (k + 1), 64)
        supa.append((xr ** 4 * np.abs(np.expm1(2 * special.exp1(xr * xr) + f_real(G, xr).real))).max())
    kk = np.arange(3, 10) * math.log(2)
    sl = np.polyfit(kk, np.log(sups), 1)[0]; sla = np.polyfit(kk, np.log(supa), 1)[0]
    oka &= abs(sl + zeta / 2) < 0.4 and abs(sla - (4 - zeta / 2)) < 0.6
    info.append(f"zeta={zeta:g}: slope {sl:.2f} (-{zeta/2:g}), (a - c x^4)/c slope {sla:.2f} ({4-zeta/2:g})")
check("one realization: sup over dyadic annuli of |R_Pi - R_inf| ~ |z|^(-zeta/2); a - c x^n bounded iff zeta > 2n", oka, "; ".join(info))

# no zeros in a realization; the resonant factor 2 - e^{-s^2 z^2} has them, Eq. (qzeros)
idx, tau, th = sample_gas(np.random.default_rng(5), 1, 4, 9.0, 0.5)
Rc = 2.0; ph = np.linspace(0, 2 * math.pi, 40001)
zc = Rc * np.exp(1j * ph)
L = (th[None, :] * -np.expm1(-(tau[None, :] * zc[:, None]) ** 2)).sum(1)
wind_a = np.unwrap(np.angle(np.exp(L)))
wind_r = sum(np.unwrap(np.angle(2 - np.exp(-(t_ * zc) ** 2)))[-1] - np.angle(2 - np.exp(-(t_ * Rc) ** 2)) for t_ in tau)
cnt = sum(2 * sum(1 for kq in range(-50, 51) if abs(complex(math.log(2), 2 * math.pi * kq)) < (t_ * Rc) ** 2) for t_ in tau)
check("argument principle on |z| = 2M^2: realization winds 0 times; resonant product winds = its zero count",
      abs(wind_a[-1] - wind_a[0]) < 1e-6 and abs(wind_r / (2 * math.pi) - cnt) < 1e-6 and cnt > 0, f"[0, {wind_r/(2*math.pi):.4f} = {cnt}]")

# Table (fluct) and the zero-point fluctuation, Eq. (drhorel)
tabf = {4: (0.555, 1.190, 0.257, 4.47, 0.372), 5: (0.561, 1.195, 0.253, 3.74, 0.281), 6: (0.566, 1.198, 0.250, 3.35, 0.227),
        7: (0.569, 1.201, 0.249, 3.11, 0.190), 8: (0.571, 1.202, 0.247, 2.97, 0.163)}
okt = True
for nn_, row in tabf.items():
    zeta_ = 2 * nn_ + 1
    sd = math.sqrt(nn_ * TH0 / zeta_); b95 = math.exp(-math.sqrt(2 * nn_ * TH0 / zeta_ * math.log(20)))
    xstar = optimize.brentq(lambda x: math.sqrt(var_f(x, nn_, zeta_)) - 1e-3, 0.5, 50)
    drr = 2 * math.sqrt(TH0 / (nn_ * (zeta_ - 4)))
    got = (sd, EeR(nn_, zeta_), b95, xstar, drr)
    okt &= all(abs(g_ - v_) <= 0.0005 + 0.0011 * v_ for g_, v_ in zip(got, row))
check("Table (fluct): sd, E e^R_inf, 95% bound, x_*, delta rho/rho_1 for n = 4..8", okt)
check("Eq. (drhorel): sd(delta rho)/rho_1 = 2 [th0/(n(zeta-4))]^(1/2) from (3 mu/16 pi^2)^2 n th0/(zeta-4) and rho_1 = 3 n mu/(32 pi^2)",
      abs(3 * 0.5 / (16 * math.pi ** 2) * math.sqrt(4 * TH0 / 5) / (3 * 4 * 0.5 / (32 * math.pi ** 2)) - 0.372) < 5e-4)
check("mean of delta rho by continuation: n int_0^1 tau^-3 dtau = -n/2 gives rho_1", abs(-(3 * 0.5 / (16 * math.pi ** 2)) * 6 * (-0.5) - 3 * 6 / (64 * math.pi ** 2)) < 1e-15)

npass = sum(ok for _, ok in results)
print(f"\n{npass} checks passed, {len(results) - npass} failed")
