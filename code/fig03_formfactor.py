"""
fig03_formfactor.py -- polynomial against exponential form factors.

(a) log|a(z)| over the complex z = p^2 plane for the deterministic arrangement
    of three branes, a(z) = (1+z)(1+z/2.5)(1+z/6): three logarithmic pits at the
    zeros, each a pole of the propagator.
(b) The same for the Poisson average a(z) = exp(z): a plane, no pits anywhere.
(c) Along the negative real axis: the polynomial crosses zero three times, the
    exponential never does; the crossings are the masses of the extra poles.
"""
import numpy as np
import matplotlib.pyplot as plt
from figstyle import *

fig = plt.figure(figsize=(7.2, 2.9))
X, Y = np.meshgrid(np.linspace(-8, 3, 140), np.linspace(-4, 4, 110))
Z = X + 1j * Y
M2 = [1.0, 2.5, 6.0]
apoly = np.prod([1 + Z / m for m in M2], axis=0)
aexp = np.exp(Z / 3.0)

def surf(axp, F, col, title, clip=(-6, 4)):
    L = np.log(np.abs(F))
    L = np.clip(L, *clip)
    axp.plot_surface(X, Y, L, rstride=2, cstride=2, color=col, edgecolor="none", alpha=0.95, shade=True, lightsource=LightSource(azdeg=225, altdeg=45), antialiased=True)
    axp.set_xlabel(r"$\mathrm{Re}\,z$", labelpad=-6, fontsize=7)
    axp.set_ylabel(r"$\mathrm{Im}\,z$", labelpad=-6, fontsize=7)
    axp.set_zlabel(r"$\log|a|$", labelpad=-8, fontsize=7)
    axp.tick_params(labelsize=5.5, pad=-2)
    axp.set_zlim(*clip)
    axp.view_init(elev=28, azim=-125)
    axp.text2D(0.5, 0.96, title, transform=axp.transAxes, ha="center", fontsize=8, color=INK)

ax = fig.add_subplot(1, 3, 1, projection="3d")
surf(ax, apoly, BLUE, r"$a=\prod_i(1+z/M_i^2)$: three pits")
ax.text2D(0.02, 0.97, "(a)", transform=ax.transAxes, fontsize=10, fontweight="bold")
ax2 = fig.add_subplot(1, 3, 2, projection="3d")
surf(ax2, aexp, ORANGE, r"$a=e^{z/M^2}$: no pit anywhere")
ax2.text2D(0.02, 0.97, "(b)", transform=ax2.transAxes, fontsize=10, fontweight="bold")

ax3 = fig.add_subplot(1, 3, 3)
clean_axes(ax3)
x = np.linspace(-8, 2, 500)
ax3.plot(x, np.prod([1 + x / m for m in M2], axis=0), color=BLUE, lw=1.3, label="polynomial")
ax3.plot(x, np.exp(x / 3.0), color=ORANGE, lw=1.3, label="exponential")
ax3.axhline(0, color=INK3, lw=0.7)
for m in M2:
    ax3.scatter([-m], [0], color=BLUE, s=16, zorder=5, edgecolor=SURFACE, linewidth=0.5)
ax3.set_xlabel(r"$z=p^2$ (Euclidean, negative axis)")
ax3.set_ylabel(r"$a(z)$")
ax3.set_ylim(-3, 6)
ax3.set_xlim(-8, 2)
ax3.legend(loc="upper left", fontsize=7, frameon=True, framealpha=1, edgecolor="none")
ax3.text(-7.6, -1.9, "each crossing is a pole\nof the propagator", fontsize=6.8, color=BLUE, bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.0))
panel_label(ax3, "(c)", x=-0.2)
fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.17, wspace=0.18)
save(fig, "fig03_formfactor")
