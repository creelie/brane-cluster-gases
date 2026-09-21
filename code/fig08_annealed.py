"""fig08_annealed.py -- annealed against quenched.

Panel (a): the spin-two propagator 1/(z a(z)) along the negative real axis for
five frozen realizations of a five-brane gas, each with its own poles, against
the annealed propagator built from the averaged kinetic operator, which is
smooth because the exponential has no zeros.

Panel (b): the residue budget. For every realization the residues at the
massive poles add to -1, whatever the weights; the annealed form factor has no
massive poles, so its budget is empty.
"""
import numpy as np
import matplotlib.pyplot as plt
from figstyle import *

rng = np.random.default_rng(7)
N = 5
M2 = 1.0
NREAL = 5

# five frozen realizations: weights drawn from the same law
weights = [np.sort(np.exp(rng.uniform(np.log(0.20), np.log(2.2), N)))[::-1]
           for _ in range(NREAL)]
mean_w = np.mean([w.sum() for w in weights])


def a_poly(z, w):
    return np.prod([1.0 + wi * z / M2 for wi in w], axis=0)


def residues(w):
    """Residues of 1/(z a(z)) at the massive poles z = -M2/w_i."""
    out = []
    for i, wi in enumerate(w):
        others = [wj for j, wj in enumerate(w) if j != i]
        denom = np.prod([1.0 - wj / wi for wj in others])
        out.append(-1.0 / denom)
    return np.array(out)


fig = plt.figure(figsize=(7.2, 2.8))
axA = fig.add_axes([0.070, 0.185, 0.395, 0.735])
axB = fig.add_axes([0.605, 0.185, 0.370, 0.735])

# ---------------------------------------------------------------- panel (a)
z = np.linspace(-4.2, -0.04, 6000)
for k, w in enumerate(weights):
    prop = np.abs(1.0 / (z * a_poly(z, w)))
    axA.semilogy(z, prop, color=INK3, lw=0.8, alpha=0.9,
                 label="frozen realizations" if k == 0 else None)

annealed = np.abs(1.0 / (z * np.exp(mean_w * z / M2)))
axA.semilogy(z, annealed, color=BLUE, lw=1.8, label="annealed average",
             zorder=5)

axA.set_xlim(-4.2, 0)
axA.set_ylim(3e-2, 3e3)
axA.set_xticks([-4, -3, -2, -1, 0])
axA.set_yticks([1e-1, 1e0, 1e1, 1e2, 1e3])
axA.set_yticklabels([r"$10^{-1}$", r"$1$", r"$10$", r"$10^{2}$", r"$10^{3}$"])
axA.yaxis.set_minor_locator(plt.NullLocator())
axA.set_xlabel(r"$z=p^{2}$", fontsize=7.6, labelpad=1.5)
axA.set_ylabel(r"$|1/(z\,a(z))|$", fontsize=7.6, labelpad=1.5)
axA.tick_params(labelsize=6.4)
axA.grid(True, which="major", color=GRID, lw=0.5)
axA.set_axisbelow(True)
for side in ("top", "right"):
    axA.spines[side].set_visible(False)
axA.legend(fontsize=6.4, loc="upper center", frameon=True, framealpha=0.95,
           edgecolor=GRID, borderpad=0.35, handlelength=1.6)
axA.text(0.975, 0.055, "spikes are the poles of\neach frozen realization",
         transform=axA.transAxes, fontsize=6.4, color=INK2, ha="right",
         va="bottom", linespacing=1.4,
         bbox=dict(boxstyle="round,pad=0.25", facecolor=SURFACE,
                   edgecolor="none", alpha=0.95))
panel_label(axA, "(a)", x=-0.14, y=1.03)

# ---------------------------------------------------------------- panel (b)
for k, w in enumerate(weights):
    r = residues(w)
    axB.scatter([k] * len(r), r, s=14,
                color=[AQUA if ri > 0 else ORANGE for ri in r],
                edgecolor=SURFACE, linewidth=0.4, zorder=4)
    axB.scatter([k], [r.sum()], s=26, marker="D", color=INK, zorder=6)

axB.axhline(-1.0, color=BLUE, lw=1.2, ls=(0, (4, 2)), zorder=3)
axB.set_yscale("symlog", linthresh=1.0)
axB.set_ylim(-3000, 3000)
axB.set_yticks([-1000, -100, -10, -1, 0, 1, 10, 100, 1000])
axB.set_yticklabels([r"$-10^{3}$", r"$-10^{2}$", r"$-10$", r"$-1$", r"$0$",
                     r"$1$", r"$10$", r"$10^{2}$", r"$10^{3}$"])
axB.yaxis.set_minor_locator(plt.NullLocator())
axB.set_xticks(list(range(NREAL)) + [NREAL])
axB.set_xticklabels([rf"${i+1}$" for i in range(NREAL)] + ["ann."], fontsize=6.4)
axB.set_xlim(-0.6, NREAL + 0.6)
axB.set_xlabel("frozen realization", fontsize=7.6, labelpad=1.5)
axB.set_ylabel("residues at the massive poles", fontsize=7.2, labelpad=1.5)
axB.tick_params(labelsize=6.4)
axB.grid(True, axis="y", color=GRID, lw=0.5)
axB.set_axisbelow(True)
for side in ("top", "right"):
    axB.spines[side].set_visible(False)
axB.text(NREAL, 0.0, "no massive pole", fontsize=6.4, color=BLUE, ha="center",
         va="center", rotation=90,
         bbox=dict(boxstyle="round,pad=0.18", facecolor=SURFACE,
                   edgecolor="none", alpha=0.95))
panel_label(axB, "(b)", x=-0.19, y=1.03)

save(fig, "fig08_annealed", vector=True)
