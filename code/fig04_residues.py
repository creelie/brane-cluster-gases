"""
fig04_residues.py -- the residue sum rule and the divergence census.

(a) Residues of 1/(z a(z)) at the poles of the propagator for three deterministic
    arrangements (polynomial form factors of degree 1, 3 and 5 with real masses,
    ordered by mass): the residue at zero is +1, the massive residues alternate
    starting negative, and in every case they sum to exactly -1.
(b) The superficial degree of divergence omega(L) in four dimensions for
    n = 0,...,5 extra derivative pairs, and the frontier L = (n+1)/(n-1).
(c) The number of divergent loop orders as a function of n: infinite for n <= 1,
    then 3, 2, and 1 for every n >= 4.
"""
import numpy as np
import matplotlib.pyplot as plt
from figstyle import *

fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), gridspec_kw={"width_ratios": [1.2, 1, 0.9]})
ax, ax2, ax3 = axes
for a_ in axes:
    clean_axes(a_)

def residues(M2):
    out = [(0.0, 1.0)]
    for i, mi in enumerate(M2):
        r = -1.0 / np.prod([1 - mi / mj for j, mj in enumerate(M2) if j != i])
        out.append((mi, r))
    return out
cases = [([1.0], BLUE, r"$n=1$"), ([1.0, 2.2, 4.0], ORANGE, r"$n=3$"), ([1.0, 1.6, 2.5, 4.0, 6.5], AQUA, r"$n=5$")]
width = 0.26
for k, (M2, col, lab) in enumerate(cases):
    res = residues(M2)
    xs = np.arange(len(res)) + (k - 1) * width
    ax.bar(xs, [r for _, r in res], width=width, color=col, zorder=3,
           label=lab + r": massive sum $%+.3f$" % sum(r for _, r in res[1:]))
ax.axhline(0, color=INK2, lw=0.7)
ax.set_xticks(range(6))
ax.set_xticklabels([r"$0$", r"$1$", r"$2$", r"$3$", r"$4$", r"$5$"], fontsize=7)
ax.set_xlabel(r"pole index", fontsize=7.4)
ax.set_ylabel("residue")
# the residues span an order of magnitude once there are five masses, so the
# scale is linear within a unit of zero and logarithmic outside it; nothing is
# cut off at the frame
ax.set_yscale("symlog", linthresh=1.0, linscale=0.7)
ax.set_ylim(-40, 90)
ax.set_yticks([-10, -1, 0, 1, 10])
ax.set_yticklabels([r"$-10$", r"$-1$", r"$0$", r"$1$", r"$10$"])
ax.legend(loc="upper right", fontsize=6.3, frameon=True, framealpha=1, edgecolor="none")
ax.text(-0.45, -24.0, "residue at zero is $+1$;\nmassive residues alternate", fontsize=6.8, color=INK2, bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.0))
panel_label(ax, "(a)", x=-0.14)

L = np.arange(1, 7)
for n, col in zip(range(0, 6), SERIES):
    ax2.plot(L, (2 + 2 * n) + (2 - 2 * n) * L, color=col, marker="o", ms=2.6, lw=1.0, label=rf"$n={n}$")
ax2.axhline(0, color=INK2, lw=0.8)
ax2.set_xlabel(r"loop order $L$")
ax2.set_ylabel(r"$\omega=(2+2n)+(2-2n)L$")
ax2.set_ylim(-40, 16); ax2.set_xlim(0.7, 6.3)
ax2.legend(loc="lower left", fontsize=6.2, ncol=2, frameon=True, framealpha=1, edgecolor="none", columnspacing=0.8, handlelength=1.2)
ax2.text(3.3, 9.5, r"$\omega(1)=4$ for every $n$", fontsize=6.8, color=INK2, bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.0))
panel_label(ax2, "(b)", x=-0.2)

ns = np.arange(0, 9)
counts = []
for n in ns:
    if n <= 1:
        counts.append(np.nan)
    else:
        counts.append(len([l for l in range(1, 60) if (2 + 2 * n) + (2 - 2 * n) * l >= 0]))
ax3.bar(ns[2:], counts[2:], color=BLUE, width=0.7, zorder=3)
for n in (0, 1):
    ax3.text(n, 0.3, r"$\infty$", ha="center", fontsize=11, color=ORANGE, bbox=dict(facecolor=SURFACE, edgecolor="none", pad=0.5))
ax3.set_xlabel(r"$n$")
ax3.set_ylabel("divergent loop orders")
ax3.set_ylim(0, 4); ax3.set_yticks([0, 1, 2, 3])
ax3.set_xticks(ns)
ax3.text(2.6, 3.55, r"$\{1,2,3\},\ \{1,2\},\ \{1\},\ \{1\},\dots$", fontsize=6.6, color=INK2, bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.0))
panel_label(ax3, "(c)")
fig.tight_layout(w_pad=1.4)
save(fig, "fig04_residues", vector=True)
