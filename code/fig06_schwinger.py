"""fig06_schwinger.py -- the parametric domain of the one-loop bubble.

Panel (a): the quadrant of Schwinger parameters. Dressing the propagators with
exp(-k^2/M^2) raises both lower limits from 0 to 1/M^2, which deletes the corner
that carries the ultraviolet divergence of the undressed graph. The level sets
of U = s1 + s2 show that U attains 2/M^2 on the retained domain, which is the
bound of Theorem 5 for this graph.

Panel (b): the value of the integral, in units of its closed form at p = 0,
against the external momentum, for D = 5, 6, 8. The p = 0 intercepts are the
closed form of Proposition 9.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, special
from figstyle import *

M = 1.0
a = 1.0 / M ** 2


# ----------------------------------------------------------------------
def slice_integral(u, p2, a=a):
    """The integral over s1 at fixed u = s1 + s2.  Shifting to x = s1 - u/2
    turns the exponent into a Gaussian with the wrong sign, so the slice is a
    Dawson function; writing it this way avoids a nested quadrature whose
    integrand is sharply peaked at both ends once u is large."""
    if p2 == 0.0:
        return u - 2 * a
    z = np.sqrt(p2 / u) * (u / 2.0 - a)
    return 2.0 * np.sqrt(u / p2) * np.exp(p2 * (a * a / u - a)) * special.dawsn(z)


def bubble(p2, D, a=a):
    """The parametric integral of the text.  The outer variable is u = s1 + s2,
    mapped to w = 2a/u so that the semi-infinite range becomes the unit interval;
    the quadrature then needs no truncation and no upper cutoff."""
    def integrand(w):
        return (slice_integral(2 * a / w, p2, a) *
                (2 * a) ** (1 - D / 2.0) * w ** (D / 2.0 - 2))

    val, _ = integrate.quad(integrand, 0.0, 1.0,
                            epsabs=1e-14, epsrel=1e-12, limit=400)
    return val / (4 * np.pi) ** (D / 2.0)


def bubble0_closed(D, a=a):
    """Equation (35)."""
    return ((2 * a) ** (2 - D / 2.0) /
            ((D / 2.0 - 1) * (D / 2.0 - 2)) / (4 * np.pi) ** (D / 2.0))


# ----------------------------------------------------------------------
fig = plt.figure(figsize=(7.2, 2.9))
axA = fig.add_axes([0.070, 0.165, 0.360, 0.760])
axB = fig.add_axes([0.590, 0.165, 0.385, 0.760])

# ---- (a) the domain
smax = 4.2
axA.add_patch(plt.Rectangle((0, 0), smax, smax, facecolor=SURFACE,
                            edgecolor="none", zorder=0))
# excluded region: the L-shaped strip where either parameter is below 1/M^2
axA.add_patch(plt.Rectangle((0, 0), a, smax, facecolor=ORANGE, alpha=0.20,
                            edgecolor="none", zorder=1))
axA.add_patch(plt.Rectangle((0, 0), smax, a, facecolor=ORANGE, alpha=0.20,
                            edgecolor="none", zorder=1))
axA.add_patch(plt.Rectangle((a, a), smax - a, smax - a, facecolor=BLUE,
                            alpha=0.12, edgecolor=BLUE, lw=1.0, zorder=2))

# level sets of U = s1 + s2
for u in (2.0, 3.0, 4.0, 5.0, 6.0, 7.0):
    lo, hi = max(a, u - smax), min(u - a, smax)
    if hi <= lo:
        continue
    t = np.linspace(lo, hi, 200)
    axA.plot(t, u - t, color=INK3, lw=0.7, ls=(0, (4, 3)), zorder=3)

axA.plot([a, smax], [a, smax], color=INK, lw=0.9, zorder=4)
axA.scatter([a], [a], s=22, facecolor=SURFACE, edgecolor=INK, lw=1.0, zorder=6)

axA.set_xlim(0, smax)
axA.set_ylim(0, smax)
axA.set_aspect("equal")
axA.set_xticks([0, a, 2, 3, 4])
axA.set_xticklabels([r"$0$", r"$1/M^{2}$", r"$2$", r"$3$", r"$4$"])
axA.set_yticks([0, a, 2, 3, 4])
axA.set_yticklabels([r"$0$", r"$1/M^{2}$", r"$2$", r"$3$", r"$4$"])
axA.tick_params(labelsize=6.4)
axA.set_xlabel(r"$s_{1}$", fontsize=7.6, labelpad=1.0)
axA.set_ylabel(r"$s_{2}$", fontsize=7.6, labelpad=1.0)
for side in ("top", "right"):
    axA.spines[side].set_visible(False)

axA.text(2.55, 0.47, "removed by the form factor", fontsize=6.6, color=ORANGE,
         ha="center", va="center", zorder=7)
axA.text(2.85, 2.95, r"$s_{1},s_{2}\geq 1/M^{2}$", fontsize=6.8, color=BLUE,
         ha="center", va="center", zorder=7,
         bbox=dict(boxstyle="round,pad=0.20", facecolor=SURFACE,
                   edgecolor="none", alpha=0.96))
axA.annotate("", xy=(a + 0.03, a + 0.03), xytext=(1.85, 1.30),
             arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.8), zorder=7)
axA.text(1.92, 1.24, r"$U=2/M^{2}$", fontsize=6.8, color=INK, ha="left",
         va="top", zorder=7,
         bbox=dict(boxstyle="round,pad=0.20", facecolor=SURFACE,
                   edgecolor="none", alpha=0.96))
panel_label(axA, "(a)")

# ---- (b) the integral against external momentum
p = np.linspace(0.0, 3.0, 13)
for D, col in ((5, BLUE), (6, ORANGE), (8, AQUA)):
    vals = np.array([bubble(pi ** 2, D) for pi in p])
    axB.plot(p, vals / vals[0], color=col, lw=1.3, marker="o", ms=2.6,
             markerfacecolor=SURFACE, markeredgewidth=0.8,
             label=rf"$D={D}$")
axB.set_xlim(0, 3.0)
axB.set_ylim(0, 1.05)
axB.set_xlabel(r"$|p|/M$", fontsize=7.6, labelpad=1.0)
axB.set_ylabel(r"$\mathcal{I}(p)\,/\,\mathcal{I}(0)$", fontsize=7.6, labelpad=1.0)
axB.tick_params(labelsize=6.4)
axB.grid(True, color=GRID, lw=0.5)
axB.set_axisbelow(True)
for side in ("top", "right"):
    axB.spines[side].set_visible(False)
axB.legend(fontsize=6.4, loc="upper right", frameon=True, framealpha=0.95,
           edgecolor=GRID, borderpad=0.35, handlelength=1.5)
axB.text(0.055, 0.085, "finite at every $p$", fontsize=6.6, color=INK2,
         transform=axB.transAxes, ha="left", va="bottom")
panel_label(axB, "(b)")

save(fig, "fig06_schwinger", vector=True)
