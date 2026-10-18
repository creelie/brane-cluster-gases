"""Write the data tables plotted in the figures of the paper.

Run with  python3 make_figure_data.py ../paper/figdata .  Each table is a
whitespace-separated text file read by pgfplots.
"""

import math
import os
import sys

import numpy as np
from scipy.integrate import quad

OUT = sys.argv[1] if len(sys.argv) > 1 else "../paper/figdata"
os.makedirs(OUT, exist_ok=True)
A = 1.5


def write(name, header, rows):
    with open(os.path.join(OUT, name), "w") as f:
        f.write(header + "\n")
        for r in rows:
            f.write(" ".join("%.6g" % v for v in r) + "\n")


# Im phi(x+iy)/pi for lambda(t) = A exp(-log^2 t), on a grid.
u = np.linspace(-9, 9, 72001)
lam_u = A * np.exp(-u ** 2)
et = np.exp(u)


def phi(z):
    return np.trapezoid(z / (z + et) * lam_u, u)


xs = np.linspace(-4.0, 1.0, 41)
ys = np.linspace(0.04, 1.2, 25)
rows = []
for y in ys:
    for x in xs:
        rows.append((x, y, phi(complex(x, y)).imag / math.pi))
    rows.append(())
with open(os.path.join(OUT, "imphi.dat"), "w") as f:
    f.write("x y v\n")
    for r in rows:
        f.write((" ".join("%.6g" % v for v in r) + "\n") if r else "\n")
write("lamprofile.dat", "x y v",
      [(x, 0.0, A * math.exp(-math.log(-x) ** 2) if x < 0 else 0.0)
       for x in np.linspace(-4.0, -0.02, 200)])

# Spectral density s rho(s) for the same lambda, from boundary values.
rows = []
for ls in np.linspace(-3, 3, 241):
    s = math.exp(ls)
    # Re phi(-s+i0) = PV int_0^inf (-s/(t-s)) lambda(t) dt/t
    f = lambda t: -s * A * math.exp(-math.log(t) ** 2) / t
    re_phi = quad(f, 1e-4, 1e4, weight="cauchy", wvar=s, limit=400)[0]
    lam = A * math.exp(-ls ** 2)
    rows.append((ls, lam, -math.exp(-re_phi) * math.sin(math.pi * lam) / math.pi))
write("density.dat", "ls lam srho", rows)

# Residues of three frozen scale-free configurations (theta = 1).
rng = np.random.default_rng(5)
theta = 1.0
for c in range(3):
    uk = np.cumsum(rng.exponential(1 / theta, size=80))
    rows = []
    for k in range(40):
        d = np.delete(uk, k) - uk[k]
        # log|r_k| = -sum_{j<k}(u_k-u_j) - sum_{j != k} log(1-e^{-|u_j-u_k|})
        S = np.sum(uk[k] - uk[:k])
        R = -np.sum(np.log1p(-np.exp(-np.abs(d))))
        rows.append((k + 1, (-S + R) / math.log(10), 1 if k % 2 == 0 else -1))
    write("residues%d.dat" % c, "k log10r sign", rows)
write("parabola.dat", "k v",
      [(k, -k * (k - 1) / (2 * theta) / math.log(10)) for k in np.linspace(1, 40, 80)])

# Histogram of the standardised S_k at k = 30 over many configurations.
k = 30
M = 40000
g = rng.exponential(1 / theta, size=(M, k - 1))
S = g @ np.arange(1, k)
mean = (k - 1) * k / (2 * theta)
sd = math.sqrt(np.sum(np.arange(1, k) ** 2.0)) / theta
zv = (S - mean) / sd
hist, edges = np.histogram(zv, bins=40, range=(-4, 4), density=True)
write("hist.dat", "x h", [(edges[i], hist[i]) for i in range(40)] + [(edges[40], hist[-1])])
write("gauss.dat", "x g",
      [(x, math.exp(-x * x / 2) / math.sqrt(2 * math.pi)) for x in np.linspace(-4, 4, 161)])

# Scale-free averaged spectra.
rows = []
for v in np.exp(np.linspace(math.log(1e-3), math.log(60), 300)):
    s = 1 + v
    rows.append([s] + [math.sin(math.pi * th) / math.pi * v ** (-th)
                       for th in (0.25, 0.5, 0.75, 1.5)])
write("scalefree.dat", "s a b c d", rows)

# Decay on the Euclidean axis: frozen, averaged, and 1/E F (theta = 1.5, t0 = 1).
th = 1.5
uk = np.cumsum(rng.exponential(1 / th, size=400))
rows = []
for lx in np.linspace(-2, 9, 221):
    x = 10.0 ** lx
    frozen = math.exp(-np.sum(np.log1p(x / np.exp(uk))))
    rows.append((x, frozen, (1 + x) ** (-th), (1 + x) ** (-0.5), math.exp(-th * x)))
write("decay.dat", "x frozen avg15 avg05 annealedF", rows)

# Gas points used in the thinning picture.
uk = np.cumsum(rng.exponential(1 / 1.2, size=12))
write("points.dat", "lt", [(v - 0.5,) for v in uk if v < 7])
print("figure data written to", OUT)

# x Gbar(x) = exp(-phi(x)) for lambda = 1.5 g and 0.9 g, g = exp(-log^2 t),
# against a positive Kallen-Lehmann propagator 1/x + 0.6/(x+1).
lam09 = 0.9 * np.exp(-u ** 2)
rows = []
for lx in np.linspace(-3, 4, 141):
    x = 10.0 ** lx
    p15 = np.trapezoid(x / (x + et) * lam_u, u)
    p09 = np.trapezoid(x / (x + et) * lam09, u)
    rows.append((x, math.exp(-p15), math.exp(-p09), 1 + 0.6 * x / (x + 1)))
write("monotone.dat", "x e15 e09 kl", rows)
print("monotone data written")
