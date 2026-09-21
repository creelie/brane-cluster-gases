"""
fig05_propagator.py -- the Gaussian form factor in position space.

(a) The Euclidean propagator of the Gaussian form factor in D = 3, 4, 5, 6,
    G_D(r) = gamma(D/2-1, M^2 r^2/4)/(4 pi^{D/2} r^{D-2}), against the massless
    propagator it approaches: finite at the origin in every dimension.
(b) The Newtonian potential -(Gm/r) erf(Mr/2) against -Gm/r, and the relative
    deviation erfc(Mr/2), which drops below one part in a thousand at Mr = 4.65.
(c) The weak-field metric function 1 + 2 Phi for masses below and above the
    threshold m_* = sqrt(pi)/(2GM): below it the linearised field is weak
    everywhere; above it the approximation fails inside r of order 1/M.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import special
from figstyle import *

fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7))
ax, ax2, ax3 = axes
for a_ in axes:
    clean_axes(a_)

r = np.linspace(0.02, 5, 400)
for D, col in zip((3, 4, 5, 6), SERIES):
    G = special.gammainc(D / 2 - 1, r * r / 4) * special.gamma(D / 2 - 1) / (4 * np.pi ** (D / 2) * r ** (D - 2))
    G0 = 1.0 / (2 ** (D - 1) * np.pi ** (D / 2) * (D - 2))
    ax.plot(r, G / G0, color=col, lw=1.2, label=rf"$D={D}$")
    free = special.gamma(D / 2 - 1) / (4 * np.pi ** (D / 2) * r ** (D - 2)) / G0
    ax.plot(r, free, color=col, lw=0.7, ls=(0, (3, 2)), alpha=0.7)
ax.set_ylim(0, 1.1)
ax.set_yticks([0, 0.5, 1.0]); ax.set_xticks([0, 1, 2, 3, 4, 5]); ax.set_xlim(0, 5)
ax.set_xlabel(r"$Mr$")
ax.set_ylabel(r"$G_D(r)/G_D(0)$")
ax.legend(loc="upper right", fontsize=6.6, frameon=True, framealpha=1, edgecolor="none")
ax.text(0.15, 0.08, "dashed: massless propagator", fontsize=6.6, color=INK2, bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.0))
panel_label(ax, "(a)", x=-0.16)

x = np.linspace(0.01, 8, 500)
ax2.plot(x, -special.erf(x / 2) / x, color=BLUE, lw=1.3, label=r"$-\mathrm{erf}(Mr/2)/(Mr)$")
ax2.plot(x, -1 / x, color=INK3, lw=0.9, ls=(0, (3, 2)), label=r"$-1/(Mr)$")
ax2.set_ylim(-1.2, 0.05)
ax2.set_yticks([-1.0, -0.5, 0.0]); ax2.set_xticks([0, 2, 4, 6, 8]); ax2.set_xlim(0, 8)
ax2.set_xlabel(r"$Mr$")
ax2.set_ylabel(r"$\Phi/(GmM)$")
ax2.axvline(4.654, color=ORANGE, lw=0.8)
ax2.text(4.75, -0.75, r"$Mr=4.65$:", fontsize=6.6, color=ORANGE, bbox=dict(facecolor=SURFACE, edgecolor="none", pad=0.8))
ax2.text(4.75, -0.88, r"$\mathrm{erfc}=10^{-3}$", fontsize=6.6, color=ORANGE, bbox=dict(facecolor=SURFACE, edgecolor="none", pad=0.8))
ax2.text(0.15, -0.63, r"$\Phi(0)=-GmM/\sqrt{\pi}$", fontsize=6.6, color=BLUE, bbox=dict(facecolor=SURFACE, edgecolor="none", pad=0.8))
ax2.legend(loc="lower right", fontsize=6.6, frameon=True, framealpha=1, edgecolor="none")
panel_label(ax2, "(b)", x=-0.2)

for ratio, col, lab in ((0.5, BLUE, r"$m=0.5\,m_*$"), (1.0, ORANGE, r"$m=m_*$"), (2.0, AQUA, r"$m=2\,m_*$")):
    # 1 + 2 Phi with Phi = -G m erf(Mr/2)/r and G m M = ratio * sqrt(pi)/2
    f = 1 - 2 * ratio * (np.sqrt(np.pi) / 2) * special.erf(x / 2) / x
    ax3.plot(x, f, color=col, lw=1.2, label=lab)
ax3.axhline(0, color=INK2, lw=0.7)
ax3.set_ylim(-1.2, 1.1)
ax3.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0]); ax3.set_xticks([0, 2, 4, 6, 8]); ax3.set_xlim(0, 8)
ax3.set_xlabel(r"$Mr$")
ax3.set_ylabel(r"$1+2\Phi$")
ax3.legend(loc="lower right", fontsize=6.6, frameon=True, framealpha=1, edgecolor="none")
ax3.text(0.15, -0.95, r"$m_*=\sqrt{\pi}/(2GM)$", fontsize=6.8, color=INK2, bbox=dict(facecolor=SURFACE, edgecolor="none", pad=0.8))
panel_label(ax3, "(c)")
fig.tight_layout(w_pad=1.4)
save(fig, "fig05_propagator", vector=True)
