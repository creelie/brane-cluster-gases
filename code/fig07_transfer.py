"""fig07_transfer.py -- where multiplicativity comes from.

Panel (a): a mode crossing a stack of branes. The direct path contributes the
product of the single-brane transmission amplitudes; every correction contains
a round trip and so carries two reflection amplitudes.

Panel (b): the relative error of the product rule, measured against the exact
transfer-matrix result for a stack of N identical branes, as a function of the
single-brane reflectivity. The dashed guide is the bound of the lemma, which
grows like the number of pairs times |r|^2.
"""
import numpy as np
import matplotlib.pyplot as plt
from figstyle import *


def transfer_exact(t, r, delta, N):
    """Exact transmission through N identical branes by composing 2x2 transfer
    matrices, including every multiple reflection."""
    # transfer matrix of one brane in the (forward, backward) basis
    Mb = np.array([[1.0 / t, np.conj(r) / np.conj(t)],
                   [r / t, 1.0 / np.conj(t)]], dtype=complex)
    # free propagation between branes
    Mf = np.array([[np.exp(1j * delta), 0.0],
                   [0.0, np.exp(-1j * delta)]], dtype=complex)
    M = np.eye(2, dtype=complex)
    for i in range(N):
        M = M @ Mb
        if i < N - 1:
            M = M @ Mf
    return 1.0 / M[0, 0]


fig = plt.figure(figsize=(7.2, 2.75))
axA = fig.add_axes([0.035, 0.09, 0.455, 0.845])
axB = fig.add_axes([0.605, 0.205, 0.370, 0.715])

# ---------------------------------------------------------------- panel (a)
axA.set_xlim(0, 10)
axA.set_ylim(0, 5.2)
axA.axis("off")

xs = [2.2, 4.3, 6.4, 8.5]
for k, x in enumerate(xs):
    axA.add_patch(plt.Rectangle((x - 0.11, 0.75), 0.22, 3.35,
                                facecolor=BLUE, alpha=0.30,
                                edgecolor=BLUE, lw=0.9, zorder=2))
    axA.text(x, 0.44, rf"$B_{{{k+1}}}$", fontsize=7.2, ha="center",
             va="center", color=BLUE, zorder=5)
    axA.text(x, 4.35, rf"$t_{{{k+1}}}$", fontsize=7.4, ha="center",
             va="center", color=INK, zorder=5)

# the direct path
axA.annotate("", xy=(9.6, 2.95), xytext=(0.45, 2.95),
             arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.5), zorder=4)
axA.text(0.45, 3.30, "direct path", fontsize=7.0, color=ORANGE, ha="left",
         va="bottom", zorder=5)
axA.text(9.55, 3.30, r"$\prod_i t_i$", fontsize=7.8, color=ORANGE, ha="right",
         va="bottom", zorder=5)

# one round trip between branes 2 and 3
axA.annotate("", xy=(4.42, 1.72), xytext=(6.28, 1.72),
             arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.0,
                             linestyle=(0, (4, 2))), zorder=4)
axA.annotate("", xy=(6.28, 1.30), xytext=(4.42, 1.30),
             arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.0,
                             linestyle=(0, (4, 2))), zorder=4)
axA.text(5.35, 0.92, r"one round trip: $r_2' r_3$", fontsize=7.0, color=INK2,
         ha="center", va="center", zorder=5)

axA.text(5.0, 4.90, "every correction to the product carries two reflections",
         fontsize=7.2, ha="center", va="center", color=INK)
panel_label(axA, "(a)", x=0.0, y=1.00)

# ---------------------------------------------------------------- panel (b)
rvals = np.logspace(-3, -0.7, 26)
delta = 0.7
for N, col in ((3, BLUE), (5, ORANGE), (8, AQUA)):
    err = []
    for r in rvals:
        t = np.sqrt(1.0 - r ** 2)
        exact = transfer_exact(t, r, delta, N)
        # the product rule: the free phase of the stack times the single-brane
        # transmissions, with no round trips
        free = transfer_exact(1.0, 0.0, delta, N)
        approx = free * t ** N
        err.append(abs(exact - approx) / abs(exact))
    axB.loglog(rvals, err, color=col, lw=1.3, marker="o", ms=2.4,
               markerfacecolor=SURFACE, markeredgewidth=0.7, label=rf"$N={N}$")
axB.loglog(rvals, 0.5 * rvals ** 2, color=INK2, lw=0.9, ls=(0, (4, 2)),
           label=r"$\propto|r|^{2}$", zorder=1)
axB.set_xlabel(r"single-brane reflectivity $|r|$", fontsize=7.6, labelpad=1.5)
axB.set_ylabel("relative error of the product rule", fontsize=7.2, labelpad=1.5)
axB.tick_params(labelsize=6.4)
axB.grid(True, which="major", color=GRID, lw=0.5)
axB.set_axisbelow(True)
for side in ("top", "right"):
    axB.spines[side].set_visible(False)
axB.legend(fontsize=6.4, loc="upper left", frameon=True, framealpha=0.95,
           edgecolor=GRID, borderpad=0.35, handlelength=1.5)
panel_label(axB, "(b)", x=-0.20, y=1.03)

save(fig, "fig07_transfer", vector=True)
