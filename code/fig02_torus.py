"""
fig02_torus.py -- intersection numbers on a torus.

(a) Two closed curves of winding numbers (1,2) and (1,-3) on a torus in space.
    They meet in exactly |1*(-3) - 1*2| = 5 points, marked.
(b) The determinant law on the square torus: for every primitive pair with
    entries in [-3,3] the number of intersection points equals |p1 q2 - p2 q1|.
(c) The count of K-fold strata of the coordinate arrangement in T^D for several D,
    symmetric under K -> D-K, with alternating sum zero.
"""
import numpy as np
import matplotlib.pyplot as plt
from math import comb, gcd
from figstyle import *

fig = plt.figure(figsize=(7.2, 2.9))
ax = fig.add_subplot(1, 3, 1, projection="3d")
blank_3d(ax); ax.computed_zorder = False
ax.view_init(elev=38, azim=-55)
R, r = 1.0, 0.45
u = np.linspace(0, 2 * np.pi, 80); v = np.linspace(0, 2 * np.pi, 36)
U, V = np.meshgrid(u, v)
X = (R + r * np.cos(V)) * np.cos(U); Y = (R + r * np.cos(V)) * np.sin(U); Z = r * np.sin(V)
add_surface(ax, X, Y, Z, "#dcdad3", light=(0.5, -0.5, 0.6), base=0.7, spread=0.35, alpha=0.55, zorder=1)
def curve(p, q, N=600):
    t = np.linspace(0, 2 * np.pi, N)
    uu = p * t; vv = q * t
    return np.column_stack([(R + (r + 0.012) * np.cos(vv)) * np.cos(uu), (R + (r + 0.012) * np.cos(vv)) * np.sin(uu), (r + 0.012) * np.sin(vv)])
c1 = curve(1, 2); c2 = curve(1, -3)
tube(ax, c1, 0.022, BLUE, nseg=6)
tube(ax, c2, 0.022, ORANGE, nseg=6)
# intersection points: solve (s, 2s) = (t, -3t) mod 2pi: s = t + 2 pi a, 2s + 3t = 2 pi b -> 5t = 2 pi (b - 2a)
for k in range(5):
    t = 2 * np.pi * k / 5
    uu, vv = t, -3 * t
    p = np.array([(R + (r + 0.03) * np.cos(vv)) * np.cos(uu), (R + (r + 0.03) * np.cos(vv)) * np.sin(uu), (r + 0.03) * np.sin(vv)])
    ax.scatter([p[0]], [p[1]], [p[2]], color=INK, s=22, zorder=9, depthshade=False, edgecolor=SURFACE, linewidth=0.5)
ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5); ax.set_zlim(-1.2, 1.2)
ax.text2D(0.02, 0.97, "(a)", transform=ax.transAxes, fontsize=10, fontweight="bold")
ax.text2D(0.5, 0.0, r"windings $(1,2)$ and $(1,-3)$: five points", transform=ax.transAxes, ha="center", fontsize=8, color=INK2)

ax2 = fig.add_subplot(1, 3, 2)
clean_axes(ax2)
xs, ys = [], []
def count(p1, q1, p2, q2):
    d = abs(p1 * q2 - p2 * q1)
    pts = set()
    det = p1 * q2 - p2 * q1
    for k in range(d):
        for l in range(d):
            s = (k * q2 - l * p2) / det
            pts.add((round((p1 * s) % 1, 7) % 1, round((q1 * s) % 1, 7) % 1))
    return len(pts)
seen = {}
for p1 in range(-3, 4):
    for q1 in range(-3, 4):
        for p2 in range(-3, 4):
            for q2 in range(-3, 4):
                d = abs(p1 * q2 - p2 * q1)
                if d == 0 or gcd(p1, q1) != 1 or gcd(p2, q2) != 1:
                    continue
                c = count(p1, q1, p2, q2)
                seen[(d, c)] = seen.get((d, c), 0) + 1
for (d, c), n in seen.items():
    ax2.scatter([d], [c], s=8 + 1.2 * n, color=BLUE, zorder=3, alpha=0.9)
ax2.plot([0, 19], [0, 19], color=INK3, lw=0.8, ls=(0, (4, 3)))
ax2.set_xlabel(r"$|p_1q_2-p_2q_1|$")
ax2.set_ylabel("intersection points counted")
ax2.set_xlim(0, 19); ax2.set_ylim(0, 19)
ax2.set_xticks([0, 5, 10, 15]); ax2.set_yticks([0, 5, 10, 15])
ax2.text(1.0, 16.5, "960 primitive pairs,\nno exception", fontsize=7.2, color=INK2, bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.2))
panel_label(ax2, "(b)")

ax3 = fig.add_subplot(1, 3, 3)
clean_axes(ax3)
for D, col, mk in ((4, BLUE, "o"), (6, ORANGE, "s"), (10, AQUA, "^"), (11, VIOLET, "D")):
    Ks = np.arange(0, D + 1)
    ax3.plot(Ks, [comb(D, K) for K in Ks], color=col, marker=mk, ms=3, lw=1.0, label=rf"$D={D}$: $2^{{{D}}}$ strata")
ax3.set_yscale("log")
ax3.set_xlabel(r"$K$ (number of branes meeting)")
ax3.set_ylabel(r"$\binom{D}{K}$ strata of type $K$")
ax3.set_xlim(-0.5, 11.5); ax3.set_ylim(0.7, 3000)
ax3.legend(loc="upper left", fontsize=6.4, frameon=True, framealpha=1, edgecolor="none")
panel_label(ax3, "(c)")
fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.17, wspace=0.42)
save(fig, "fig02_torus")
