"""Figures for "Finite quantum gravity from a scale-invariant gas of branes".

Every surface and curve drawn here is a closed-form expression derived in the
paper, or a sample realization of the Poisson gas; the script only evaluates
them.  In the fine-grained gas the branes far below the resolution of a panel
are represented by the Gaussian limit of their compensated strength, as the
captions say.

Run:  python3 make_figures.py [name ...]   (writes PNG files to ../figs/)
"""
import cmath
import math
import os

import mpmath as mp

import matplotlib
matplotlib.use("Agg")
import matplotlib.ticker
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy import integrate, special

OUT = os.environ.get("FIG_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figs"))
os.makedirs(OUT, exist_ok=True)
G_E = 0.5772156649015329
rng = np.random.default_rng(7)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["cmr10", "DejaVu Serif"], "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True, "font.size": 8.5, "axes.labelsize": 9, "axes.titlesize": 9,
    "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.linewidth": 0.6,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "savefig.dpi": 400, "figure.dpi": 150, "lines.linewidth": 1.2, "axes.unicode_minus": False,
})


def ein(y):
    """Entire exponential integral for real y >= 0 (vectorized)."""
    y = np.asarray(y, dtype=float)
    out = np.empty_like(y)
    small = y < 0.5
    ys = y[small]
    term = np.ones_like(ys); acc = np.zeros_like(ys)
    for k in range(1, 30):
        term = term * ys / k
        acc += (-1) ** (k + 1) * term / k
    out[small] = acc
    yl = y[~small]
    out[~small] = np.log(yl) + G_E + special.exp1(yl)
    return out


def ein_c(w):
    """Entire exponential integral for complex w (vectorized)."""
    w = np.asarray(w, dtype=complex)
    out = np.empty_like(w)
    small = np.abs(w) < 2.0
    ws = w[small]
    term = np.ones_like(ws); acc = np.zeros_like(ws)
    for k in range(1, 60):
        term = term * ws / k
        acc += (-1) ** (k + 1) * term / k
    out[small] = acc
    wl = w[~small]
    out[~small] = np.log(wl) + G_E + special.exp1(wl)
    return out


def a_real(x, n):
    return np.exp(0.5 * n * ein(np.asarray(x) ** 2))


def label(ax, s, x=0.03, y=0.95, **kw):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, va="top", ha="left", **kw)




def style3d(ax, elev=24, azim=-58, pad=-2.5):
    """Light panes, thin grid, compact ticks: one look for every 3D panel."""
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0.975, 0.975, 0.98, 1.0))
        axis._axinfo["grid"].update(color=(0.82, 0.82, 0.85, 1.0), linewidth=0.35)
        axis.line.set_linewidth(0.6)
    ax.tick_params(axis="both", pad=pad, labelsize=6.5)
    ax.view_init(elev=elev, azim=azim)
    ax.set_facecolor((1, 1, 1, 0))


def zlab(ax, s, x=0.80, y=0.90, **kw):
    """Label for the vertical axis, placed above its top end so it never spills into a neighbour."""
    ax.text2D(x, y, s, transform=ax.transAxes, fontsize=7.5, ha="left", va="bottom", **kw)


def ribbon(ax, x, y0, z, color, zbase, alpha=0.35, lw=1.1):
    """A curve z(x) drawn at depth y0 with a translucent curtain down to zbase."""
    verts = [list(zip(x, np.full_like(x, y0), z)) + [(x[-1], y0, zbase), (x[0], y0, zbase)]]
    ax.add_collection3d(Poly3DCollection(verts, facecolor=color, edgecolor="none", alpha=alpha))
    ax.plot(x, np.full_like(x, y0), z, color=color, lw=lw)


# ---------------------------------------------------------------- the fine-grained gas
TH0 = math.log(2)


def gas_identical(rng_, n=4, tau_min=1e-6, tau_max=1.0):
    """identical branes of strength th0, intensity (n/th0) dtau/tau on [tau_min, tau_max]"""
    k = rng_.poisson(n / TH0 * math.log(tau_max / tau_min))
    return tau_min * (tau_max / tau_min) ** rng_.random(k)


def gas_fine(rng_, n=4, zeta=9.0, tau_c=0.3, tau_min=1e-9, h=0.01):
    """power law: strength th0 tau^zeta, intensity (n/th0) tau^-zeta dtau/tau.  Branes with
    tau >= tau_c are sampled exactly; below tau_c each cell of width h in log tau carries the
    Gaussian limit of its compensated strength, variance n th0 tau^zeta h."""
    lam = (n / TH0) * (tau_c ** -zeta - 1) / zeta
    N = rng_.poisson(lam); A = tau_c ** -zeta
    tau = (A - (A - 1) * rng_.random(N)) ** (-1 / zeta)
    L = np.arange(math.log(tau_min), math.log(tau_c), h) + h / 2
    tg = np.exp(L)
    S = np.sqrt(n * TH0 * tg ** zeta * h) * rng_.standard_normal(len(L))
    R_inf = (TH0 * tau ** zeta).sum() - n * math.log(1 / tau_c) + S.sum()
    return dict(tau=tau, th=TH0 * tau ** zeta, tg=tg, S=S, n=n, zeta=zeta, tau_c=tau_c, R_inf=R_inf)


def fine_minus_inf(G, w):
    """R_Pi(w) - R_inf for psi_* = 1 - exp(-w^2), complex w (any shape)"""
    w = np.asarray(w, dtype=complex); sh = w.shape; w = w.ravel()
    out = np.zeros(w.shape, dtype=complex)
    with np.errstate(over="ignore", invalid="ignore"):
        for t_, th_ in zip(np.array_split(G["tau"], max(1, len(G["tau"]) // 3000)), np.array_split(G["th"], max(1, len(G["th"]) // 3000))):
            out -= (th_[None, :] * np.exp(-(t_[None, :] * w[:, None]) ** 2)).sum(1)
        s = np.linspace(math.log(G["tau_c"]), 0, 801)
        out += G["n"] * integrate.trapezoid(np.exp(-(np.exp(s)[None, :] * w[:, None]) ** 2), s, axis=1)
        out -= (G["S"][None, :] * np.exp(-(G["tg"][None, :] * w[:, None]) ** 2)).sum(1)
    return out.reshape(sh)


def var_R_fine(x, n, zeta):
    g = lambda b, y: special.gammainc(b, y) * special.gamma(b)
    return n * TH0 * (1 / zeta - g(zeta / 2, x * x) / x ** zeta + 0.5 * g(zeta / 2, 2 * x * x) / (2 * x * x) ** (zeta / 2))


def var_f_fine(x, n, zeta):
    return 0.5 * n * TH0 * special.gammainc(zeta / 2, 2 * x * x) * special.gamma(zeta / 2) / (2 * x * x) ** (zeta / 2)


def fig_gas():
    fig = plt.figure(figsize=(7.0, 2.85))
    r_ = np.random.default_rng(12)
    n = 4
    # (a) one realization of each gas as stems: identical branes and the power law with zeta = 2
    ax = fig.add_axes([0.0, 0.04, 0.30, 0.84], projection="3d")
    ax.computed_zorder = False
    t_id = gas_identical(r_, n, 0.1, 1.0)
    zeta2 = 2.0
    lam = (n / TH0) * (0.1 ** -zeta2 - 1) / zeta2
    A = 0.1 ** -zeta2
    t_fg = (A - (A - 1) * r_.random(r_.poisson(lam))) ** (-1 / zeta2)
    for tset, th, y0, col in ((t_id, np.full(len(t_id), TH0), 1.25, "#d35400"), (t_fg, TH0 * t_fg ** zeta2, 0.0, "#1f618d")):
        lx = np.log10(tset); yy = y0 + r_.uniform(0.05, 0.95, len(tset))
        order = np.argsort(-yy)
        for i in order:
            ax.plot([lx[i], lx[i]], [yy[i], yy[i]], [0, th[i]], color=col, lw=0.55, alpha=0.85)
        ax.scatter(lx, yy, th, s=3.5 if col == "#1f618d" else 9, color=col, depthshade=False, edgecolors="none")
        verts = [[(-1, y0, 0), (0, y0, 0), (0, y0 + 1, 0), (-1, y0 + 1, 0)]]
        ax.add_collection3d(Poly3DCollection(verts, facecolor=col, edgecolor=col, alpha=0.06, linewidths=0.4))
    lxx = np.linspace(-1, 0, 50)
    ax.plot(lxx, np.full_like(lxx, 1.0), TH0 * 10 ** (zeta2 * lxx), color="#1f618d", lw=1.0, ls="--")
    ax.set_xlim(-1, 0); ax.set_ylim(0, 2.25); ax.set_zlim(0, 0.75)
    ax.set_xticks([-1, -0.5, 0]); ax.set_xticklabels(["$-1$", "$-0.5$", "$0$"]); ax.set_yticks([]); ax.set_zticks([0, 0.35, 0.69]); ax.set_zticklabels(["0", "0.35", r"$\log 2$"])
    ax.set_xlabel(r"$\log_{10}(sM^2)$", labelpad=-5)
    zlab(ax, r"strength $\vartheta$", x=0.78, y=0.86)
    style3d(ax, elev=20, azim=-66)
    fig.text(0.015, 0.95, "(a)", fontsize=9)
    fig.text(0.055, 0.955, r"identical branes, $\vartheta=\log 2$ (orange, %d);" "\n" r"power law, $\vartheta=\vartheta_0\tau^{2}$ (blue, %d)" % (len(t_id), len(t_fg)),
             fontsize=6.7, ha="left", va="top", linespacing=1.15)

    # (b) psi_*(sz) over (log s, log z): the brane switches on across sz = 1
    ax = fig.add_axes([0.318, 0.04, 0.31, 0.84], projection="3d")
    u = np.linspace(-3, 0, 151)
    v = np.linspace(-1, 3, 201)
    U, V = np.meshgrid(u, v)
    Z = -np.expm1(-(10.0 ** (U + V)) ** 2)
    ax.plot_surface(U, V, Z, cmap="magma", vmin=-0.25, vmax=1.15, rcount=120, ccount=120, linewidth=0, antialiased=False, shade=True)
    uu = np.linspace(-2.95, 0, 50)
    ax.plot(uu, -uu, np.full_like(uu, 1 - math.exp(-1)) + 0.03, color="#2e86c1", lw=1.3, ls="--", zorder=20)
    ax.set_xlabel(r"$\log_{10}(sM^2)$", labelpad=-5)
    ax.set_ylabel(r"$\log_{10}(z/M^2)$", labelpad=-5)
    ax.set_zticks([0, 0.5, 1])
    style3d(ax, elev=25, azim=-60)
    ax.text(-2.9, 2.9, 0.85, r"$sz=1$", color="#2e86c1", fontsize=7.5, zorder=30)
    fig.text(0.325, 0.95, "(b)", fontsize=9)
    fig.text(0.36, 0.955, r"$\psi_*(sz)=\vartheta(s)^{-1}\log t^{-1}$: a brane is saturated" "\n" r"for modes shorter than its size", fontsize=6.7, ha="left", va="top", linespacing=1.15)

    # (c) the exponent L_Pi(x) of eight realizations of each gas against the skeleton, as a waterfall
    ax = fig.add_axes([0.635, 0.04, 0.325, 0.84], projection="3d")
    lx = np.linspace(-1, 3, 161)
    x = 10.0 ** lx
    skel = 0.5 * n * ein(x ** 2)
    shades_o = matplotlib.colormaps["Oranges"](np.linspace(0.45, 0.85, 8))
    shades_b = matplotlib.colormaps["Blues"](np.linspace(0.5, 0.9, 8))
    rows = []
    for k in range(8):
        G = gas_fine(r_, n, 9.0)
        rows.append((2.0 + k, skel + G["R_inf"] + fine_minus_inf(G, x).real, shades_b[k]))
    for k in range(8):
        t_ = gas_identical(r_, n, 1e-6, 1.0)
        rows.append((11.0 + k, (TH0 * -np.expm1(-np.outer(t_, x) ** 2)).sum(0), shades_o[k]))
    ax.computed_zorder = False
    for y0, Lk, col in sorted(rows, key=lambda r: -r[0]):          # back to front
        ax.plot(lx, np.full_like(lx, y0), skel, color="0.6", lw=0.35, ls=":")
        ax.plot(lx, np.full_like(lx, y0), Lk, color=col, lw=0.95)
    ribbon(ax, lx, 0.0, skel, "k", 0.0, alpha=0.14, lw=1.8)
    ax.set_xlabel(r"$\log_{10}x$", labelpad=-5)
    ax.set_yticks([0, 5.5, 14.5]); ax.set_yticklabels(["skeleton", r"$\zeta=9$", "identical"], fontsize=6.0)
    ax.set_ylim(0, 19); ax.set_zlim(0, 36); ax.set_zticks([0, 10, 20, 30])
    ax.set_box_aspect((1.5, 1.5, 0.75))
    zlab(ax, r"$L_\Pi(x)$", x=0.80, y=0.86)
    style3d(ax, elev=30, azim=-74)
    fig.text(0.655, 0.95, "(c)", fontsize=9)
    fig.text(0.69, 0.955, r"skeleton $\frac{n}{2}\mathrm{Ein}(x^2)$ (black; dotted on each row)," "\n" r"power law $\zeta=9$ (blue), identical (orange)", fontsize=6.7, ha="left", va="top", linespacing=1.15)
    fig.savefig(os.path.join(OUT, "fig_gas.png"))
    plt.close(fig)


def fig_fluct():
    fig = plt.figure(figsize=(7.0, 5.9))
    n = 4
    r_ = np.random.default_rng(21)
    # (a) identical branes: twelve sample paths of R_Pi(x) and the band of Eq. (Psistar)
    ax = fig.add_axes([-0.01, 0.50, 0.5, 0.46], projection="3d")
    lx = np.linspace(-1, 4, 201); x = 10.0 ** lx
    sd = np.sqrt(n * TH0 * (ein(x ** 2) - 0.5 * ein(2 * x ** 2)))
    pal = matplotlib.colormaps["Oranges"](np.linspace(0.45, 0.9, 12))
    for sgn in (1, -1):
        Y, X_ = np.meshgrid([0.5, 12.5], lx)
        ax.plot_surface(X_, Y, np.outer(sgn * sd, [1, 1]), color="#f5cba7", alpha=0.13, linewidth=0, shade=False)
    for k in range(12):
        t_ = gas_identical(r_, n, 1e-7, 1.0)
        Rk = (TH0 * -np.expm1(-np.outer(t_, x) ** 2)).sum(0) - 0.5 * n * (ein(x ** 2) - ein((1e-7 * x) ** 2))
        ax.plot(lx, np.full_like(lx, k + 1), Rk, color=pal[k], lw=0.9)
    for y0 in (0.5, 12.5):
        ax.plot(lx, np.full_like(lx, y0), sd, color="#a04000", lw=0.8, ls="--")
        ax.plot(lx, np.full_like(lx, y0), -sd, color="#a04000", lw=0.8, ls="--")
    ax.set_xlabel(r"$\log_{10}x$", labelpad=-5); ax.set_ylabel("realization", labelpad=-5)
    ax.set_zlim(-9, 9); ax.set_zticks([-8, -4, 0, 4, 8]); ax.set_yticks([1, 4, 8, 12])
    zlab(ax, r"$R_\Pi(x)$", x=0.80, y=0.86)
    style3d(ax, elev=22, azim=-60)
    fig.text(0.01, 0.975, "(a)", fontsize=9)
    fig.text(0.05, 0.98, r"identical branes: $R_\Pi$ wanders without limit;" "\n" r"band $\pm[n\vartheta_0\Psi_*(x)]^{1/2}$ widens like $(\log x)^{1/2}$", fontsize=6.9, ha="left", va="top", linespacing=1.15)

    # (b) power law zeta = 9: paths level off at their own R_inf
    ax = fig.add_axes([0.5, 0.50, 0.5, 0.46], projection="3d")
    lx = np.linspace(-1, 2, 161); x = 10.0 ** lx
    sdv = np.sqrt(var_R_fine(x, n, 9.0))
    pal = matplotlib.colormaps["Blues"](np.linspace(0.5, 0.95, 12))
    for sgn in (1, -1):
        Y, X_ = np.meshgrid([0.5, 12.5], lx)
        ax.plot_surface(X_, Y, np.outer(sgn * sdv, [1, 1]), color="#aed6f1", alpha=0.15, linewidth=0, shade=False)
    for k in range(12):
        G = gas_fine(r_, n, 9.0)
        Rk = G["R_inf"] + fine_minus_inf(G, x).real
        ax.plot(lx, np.full_like(lx, k + 1), Rk, color=pal[k], lw=0.9)
        ax.scatter([lx[-1]], [k + 1], [G["R_inf"]], s=10, color="#c0392b", depthshade=False, zorder=20)
    for y0 in (0.5, 12.5):
        ax.plot(lx, np.full_like(lx, y0), sdv, color="#1b4f72", lw=0.8, ls="--")
        ax.plot(lx, np.full_like(lx, y0), -sdv, color="#1b4f72", lw=0.8, ls="--")
    ax.set_xlabel(r"$\log_{10}x$", labelpad=-5); ax.set_ylabel("realization", labelpad=-5)
    ax.set_zlim(-1.8, 1.8); ax.set_zticks([-1.5, -0.75, 0, 0.75, 1.5]); ax.set_yticks([1, 4, 8, 12])
    ax.set_xticks([-1, 0, 1, 2])
    zlab(ax, r"$R_\Pi(x)$", x=0.80, y=0.86)
    style3d(ax, elev=22, azim=-60)
    fig.text(0.51, 0.975, "(b)", fontsize=9)
    fig.text(0.55, 0.98, r"power law $\zeta=9$: each path levels off at its own" "\n" r"$R_\infty$ (red dots); band $\pm(\mathbb{E}R_\Pi^2)^{1/2}$, Eq. (varR)", fontsize=6.9, ha="left", va="top", linespacing=1.15)

    # (c) root-mean-square remainder over (log10 x, zeta)
    ax = fig.add_axes([-0.01, 0.01, 0.5, 0.46], projection="3d")
    lxg = np.linspace(0, 2, 121); zg = np.linspace(2, 16, 113)
    LX, ZG = np.meshgrid(lxg, zg)
    S = 0.5 * np.log10(var_f_fine(10.0 ** LX, n, ZG))
    ax.plot_surface(LX, ZG, S, cmap="viridis", vmin=-16, vmax=0.5, rcount=100, ccount=100, linewidth=0, antialiased=False, shade=True, alpha=0.95)
    ax.plot(lxg, np.full_like(lxg, 8.0), 0.5 * np.log10(var_f_fine(10.0 ** lxg, n, 8.0)) + 0.05, color="#e74c3c", lw=1.6, zorder=20)
    ax.contourf(LX, ZG, S, levels=np.linspace(-16, 0.5, 12), zdir="z", offset=-17, cmap="viridis", alpha=0.45)
    ax.contour(LX, ZG, S, levels=[-12, -9, -6, -3], zdir="z", offset=-17, colors="0.3", linewidths=0.5)
    ax.plot(lxg, np.full_like(lxg, 8.0), np.full_like(lxg, -17), color="#e74c3c", lw=0.9, ls="--")
    ax.set_zlim(-17, 1); ax.set_xticks([0, 0.5, 1, 1.5, 2])
    ax.set_xlabel(r"$\log_{10}x$", labelpad=-5); ax.set_ylabel(r"$\zeta$", labelpad=-5)
    ax.set_yticks([2, 6, 10, 14]); ax.set_zticks([-15, -10, -5, 0])
    zlab(ax, r"$\log_{10}$ rms of $R_\Pi-R_\infty$", x=0.70, y=0.86)
    style3d(ax, elev=24, azim=-52)
    fig.text(0.01, 0.48, "(c)", fontsize=9)
    fig.text(0.05, 0.485, r"$(\mathbb{E}|R_\Pi-R_\infty|^2)^{1/2}$, Eq. (varf): slope $-\zeta/2$ in $\log_{10}x$;" "\n" r"red: $\zeta=2n=8$, above which $a_\Pi-c\,x^n$ is bounded", fontsize=6.9, ha="left", va="top", linespacing=1.15)

    # (d) one realization over the plane of (log2|z|, arg z)
    ax = fig.add_axes([0.5, 0.01, 0.5, 0.46], projection="3d")
    ax.computed_zorder = False
    G = gas_fine(np.random.default_rng(5), n, 9.0)
    rho = np.linspace(0, 6, 121); ang = np.linspace(-0.4 * np.pi, 0.4 * np.pi, 121)
    RH, AN = np.meshgrid(rho, ang)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        F = np.log10(np.abs(fine_minus_inf(G, 2.0 ** RH * np.exp(1j * AN))))
    lo, hi = -9.0, 1.0
    F = np.nan_to_num(F, nan=hi + 1, posinf=hi + 1, neginf=lo)
    Fc = np.clip(F, lo, hi)
    Fm = np.where(F > hi, np.nan, Fc)
    AD = np.degrees(AN)
    norm = colors.Normalize(lo, hi)
    fl = lo - 1.5
    ax.contourf(RH, AD, Fc, levels=np.linspace(lo, hi, 11), zdir="z", offset=fl, cmap="RdYlBu_r", norm=norm, alpha=0.75, zorder=1)
    for a0 in (45, -45):
        ax.plot([0, 6], [a0, a0], [fl, fl], color="k", lw=0.9, ls="--", zorder=2)
    ax.plot_surface(RH, AD, Fm, facecolors=cm.RdYlBu_r(norm(np.nan_to_num(Fm, nan=hi))), rstride=1, cstride=1, linewidth=0,
                    antialiased=True, shade=False, alpha=0.93, zorder=3)
    rr = np.linspace(0, 6, 121)
    ax.plot(rr, 0 * rr, np.log10(np.abs(fine_minus_inf(G, 2.0 ** rr))) + 0.06, color="w", lw=1.0, zorder=10)
    f0 = np.log10(np.abs(fine_minus_inf(G, 2.0)))
    ax.plot(rr, 0 * rr, f0 - 4.5 * (rr - 1) * math.log10(2) + 1.2, color="k", lw=0.9, ls=":", zorder=11)
    ax.set_zlim(fl, hi); ax.set_xlim(0, 6); ax.set_ylim(-72, 72)
    ax.set_xticks([0, 2, 4, 6]); ax.set_yticks([-60, -45, 0, 45, 60]); ax.set_yticklabels(["", r"$-45^\circ$", r"$0$", r"$45^\circ$", ""])
    ax.set_zticks([-8, -4, 0])
    ax.set_xlabel(r"$\log_2|z/M^2|$", labelpad=-5); ax.set_ylabel(r"$\arg z$", labelpad=-5)
    zlab(ax, r"$\log_{10}|R_\Pi-R_\infty|$", x=0.72, y=0.86)
    style3d(ax, elev=26, azim=-62)
    fig.text(0.51, 0.48, "(d)", fontsize=9)
    fig.text(0.55, 0.485, r"one realization, $\zeta=9$; cone boundaries $|\arg z|=45^\circ$ dashed;" "\n" r"white: real axis; dotted: slope $-\zeta/2$, shifted up", fontsize=6.9, ha="left", va="top", linespacing=1.15)
    fig.savefig(os.path.join(OUT, "fig_fluct.png"))
    plt.close(fig)


# ---------------------------------------------------------------- form factor on the real axis
def fig_formfactor():
    fig = plt.figure(figsize=(7.0, 2.75))
    pal = {4: "#1b4f72", 5: "#2874a6", 6: "#c0392b", 7: "#7d3c98", 8: "#117864"}
    # (a) a(z) for n = 4..8 inside the band of the two-sided bound
    ax = fig.add_axes([0.0, 0.02, 0.30, 0.82], projection="3d")
    lx = np.linspace(-1.5, 1.6, 200)
    x = 10.0 ** lx
    for n, col in pal.items():
        c = math.exp(n * G_E / 2)
        lo = 0.5 * n * np.log10(1 + x ** 2)
        hi = lo + math.log10(c)
        verts = [list(zip(lx, np.full_like(lx, n), lo)) + list(zip(lx[::-1], np.full_like(lx, n), hi[::-1]))]
        ax.add_collection3d(Poly3DCollection(verts, facecolor=col, edgecolor="none", alpha=0.16))
        ax.plot(lx, np.full_like(lx, n), np.log10(a_real(x, n)), color=col, lw=1.3)
        ax.plot(lx[lx > 0], np.full_like(lx[lx > 0], n), n * lx[lx > 0] + math.log10(c), color=col, lw=0.7, ls="--")
    ax.set_xlabel(r"$\log_{10}x$", labelpad=-5); ax.set_ylabel(r"$n$", labelpad=-6); zlab(ax, r"$\log_{10}\hat a$")
    ax.set_xticks([-1, 0, 1]); ax.set_yticks([4, 5, 6, 7, 8]); ax.set_zticks([0, 4, 8, 12])
    style3d(ax, elev=20, azim=-64)
    fig.text(0.045, 0.955, r"$\hat a(z)$ (solid), band $(1+x^2)^{n/2}\ldots \hat c(1+x^2)^{n/2}$," "\n" r"monomial $\hat c\,x^n$ (dashed)", fontsize=6.8, ha="left", va="top", linespacing=1.15)
    fig.text(0.01, 0.95, "(a)", fontsize=9)

    # (b) Phi(x) - log(1+x^2)/2 for three admissible responses: bounded, with limit C_+
    ax = fig.add_axes([0.315, 0.02, 0.30, 0.82], projection="3d")
    lx = np.linspace(-2, 2.5, 300)
    x = 10.0 ** lx
    resp = [(r"$1-e^{-w^2}$", 0.5 * ein(x ** 2), G_E / 2, "#c0392b"),
            (r"$1-(1+w^2)e^{-w^2}$", 0.5 * (ein(x ** 2) - 1 + np.exp(-x ** 2)), (G_E - 1) / 2, "#1f618d"),
            (r"$1-e^{-w^4}$", 0.25 * ein(x ** 4), G_E / 4, "#117864")]
    for k, (lab, Phi, Cp, col) in enumerate(resp):
        g = Phi - 0.5 * np.log1p(x ** 2)
        ribbon(ax, lx, float(k), g, col, -0.55, alpha=0.10, lw=1.3)
        ax.plot([0.0, lx[-1]], [k, k], [Cp, Cp], color=col, lw=0.8, ls="--")
        fig.text(0.345, 0.80 - 0.045 * k, ["(i) ", "(ii) ", "(iii) "][k] + lab + fr"$:\ C_+={Cp:.3f}$", color=col, fontsize=6.4)
    ax.set_yticks([0, 1, 2]); ax.set_yticklabels(["(i)", "(ii)", "(iii)"], fontsize=6.3)
    ax.set_xlabel(r"$\log_{10}x$", labelpad=-5)
    ax.set_zlim(-0.55, 0.35); ax.set_zticks([-0.4, -0.2, 0.0, 0.2])
    style3d(ax, elev=20, azim=-58)
    fig.text(0.37, 0.955, r"$\Phi(x)-\frac{1}{2}\log(1+x^2)$ is bounded for every" "\n" r"admissible response; its limit is $C_+$ (dashed)", fontsize=6.8, ha="left", va="top", linespacing=1.15)
    fig.text(0.335, 0.95, "(b)", fontsize=9)

    # (c) z G(z) = a - 1 - c x^n for n = 4..8, compressed as sgn(v) log10(1+|v|)
    ax = fig.add_axes([0.635, 0.02, 0.30, 0.82], projection="3d")
    xs = np.linspace(1e-3, 6, 500)
    comp = lambda v: np.sign(v) * np.log10(1 + np.abs(v))
    for n, col in pal.items():
        c = math.exp(n * G_E / 2)
        v = a_real(xs, n) - 1 - c * xs ** n
        ribbon(ax, xs, n, comp(v), col, comp(-1.0), alpha=0.16, lw=1.2)
    for n in pal:
        ax.plot([0, 6], [n, n], [comp(-1.0)] * 2, color="0.45", lw=0.6, ls=":")
    ax.set_xlabel(r"$x=z/M^2$", labelpad=-5); ax.set_ylabel(r"$n$", labelpad=-6)
    zlab(ax, r"$\mathrm{sgn}(v)\log_{10}(1+|v|)$", x=0.58)
    ax.set_yticks([4, 5, 6, 7, 8]); ax.set_zticks([-0.25, 0.0, 0.5, 1.0, 1.5])
    style3d(ax, elev=22, azim=-60)
    fig.text(0.70, 0.955, r"$v=z\,\hat{\mathcal{G}}(z)=\hat a-1-\hat c\,x^n\to-1$:" "\n" r"$\hat{\mathcal{G}}$ is a symbol of order $-1$", fontsize=6.8, ha="left", va="top", linespacing=1.15)
    fig.text(0.665, 0.95, "(c)", fontsize=9)
    fig.savefig(os.path.join(OUT, "fig_formfactor.png"))
    plt.close(fig)


# ---------------------------------------------------------------- the complex plane
def fig_complex():
    fig = plt.figure(figsize=(7.0, 3.2))
    # (a) (2/n) log|a(z)| = Re Ein(z^2/M^4) over a disc, compressed as sign(v) log(1+|v|)
    ax = fig.add_axes([0.0, 0.06, 0.5, 0.84], projection="3d")
    Rmax = 2.2
    r = np.linspace(1e-3, Rmax, 160)
    th = np.linspace(0, 2 * np.pi, 361)
    Rg, Tg = np.meshgrid(r, th)
    X, Y = Rg * np.cos(Tg), Rg * np.sin(Tg)
    comp = lambda v: np.sign(v) * np.log1p(np.abs(v))
    T = comp(np.real(ein_c((X + 1j * Y) ** 2)))
    ax.plot_surface(X, Y, T, cmap="cividis", vmin=-3.6, vmax=2.3, rcount=160, ccount=180,
                    linewidth=0, antialiased=False, shade=True)
    xr = np.linspace(-Rmax, Rmax, 240)
    ax.plot(xr, 0 * xr, comp(np.real(ein_c(xr.astype(complex) ** 2))) + 0.05, color="w", lw=1.5, zorder=10)
    ax.plot(0 * xr, xr, comp(np.real(ein_c((1j * xr) ** 2))) + 0.05, color="#e67e22", lw=1.5, zorder=10)
    for t0 in (np.pi / 4, 3 * np.pi / 4, 5 * np.pi / 4, 7 * np.pi / 4):
        rr = np.linspace(0, Rmax, 120)
        zr = comp(np.real(ein_c((rr * np.exp(1j * t0)) ** 2)))
        ax.plot(rr * np.cos(t0), rr * np.sin(t0), zr + 0.05, color="#e74c3c", lw=1.0, ls="--", zorder=10)
    ax.set_xlabel(r"$\mathrm{Re}\,z/M^2$", labelpad=-4)
    ax.set_ylabel(r"$\mathrm{Im}\,z/M^2$", labelpad=-4)
    style3d(ax, elev=46, azim=-28, pad=-2)
    ax.set_zticks([-3, -2, -1, 0, 1, 2])
    ax.set_box_aspect((1, 1, 0.75))
    fig.text(0.02, 0.93, "(a)", fontsize=9)
    fig.text(0.27, 0.95, r"$\mathrm{sgn}(v)\log(1+|v|)$, $\;v=\frac{2}{n}\log|\hat a(z)|=\mathrm{Re}\,\mathrm{Ein}(z^2/M^4)$",
             fontsize=7.5, ha="center")

    # (b) accuracy of the local approximation a ~ c (z/M^2)^n as a surface, zeros of one frozen gas on the floor
    ax = fig.add_axes([0.5, 0.04, 0.5, 0.86], projection="3d")
    R = 5.0
    xs = np.linspace(-R, R, 241)
    X, Y = np.meshgrid(xs, xs)
    Z = X + 1j * Y
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        dev = np.abs(np.exp(3 * ein_c(Z ** 2) - 3 * G_E - 6 * np.log(Z)) - 1)
        L = np.log10(dev)
    L = np.clip(np.nan_to_num(L, nan=2.0, posinf=2.0, neginf=-16.0), -16, 2)
    floor = -21.0
    Lc = np.minimum(L, 0.0)
    ax.contourf(X, Y, Lc, levels=np.linspace(-16, 0, 17), zdir="z", offset=floor, cmap="RdYlBu_r", alpha=0.35)
    ax.plot_surface(X, Y, Lc, cmap="RdYlBu_r", vmin=-16, vmax=2, rcount=150, ccount=150, linewidth=0, antialiased=False, shade=True, alpha=0.97)
    for t0 in (np.pi / 4, 3 * np.pi / 4):
        ax.plot([-R * np.cos(t0), R * np.cos(t0)], [-R * np.sin(t0), R * np.sin(t0)], [floor + 0.05] * 2, color="k", lw=0.8, ls="--")
    n = 6
    smin = 0.15
    N = rng.poisson(n * math.log(1 / smin))
    sr = np.exp(np.log(smin) + rng.random(N) * math.log(1 / smin))
    zs = []
    for s_ in sr:
        for k in range(-6, 7):
            w = np.sqrt(complex(-math.log(2), -2 * math.pi * k)) / s_
            zs += [w, -w]
    zs = np.array(zs)
    zs = zs[np.abs(zs) < R * 0.98]
    ax.scatter(zs.real, zs.imag, np.full(zs.shape, floor + 0.1), s=7, facecolor="k", edgecolor="k", linewidth=0.3, depthshade=False, zorder=30)
    ax.set_zlim(floor, 0.5)
    ax.set_xlabel(r"$\mathrm{Re}\,z/M^2$", labelpad=-4); ax.set_ylabel(r"$\mathrm{Im}\,z/M^2$", labelpad=-4)
    ax.set_zlabel(r"$\log_{10}|\hat a/a_{\rm loc}-1|$", labelpad=-5)
    ax.set_zticks([-15, -10, -5, 0])
    style3d(ax, elev=30, azim=-52, pad=-2)
    fig.text(0.52, 0.93, "(b)", fontsize=9)
    fig.text(0.76, 0.95, r"error of $a_{\rm loc}=\hat c\,(z/M^2)^n$ for $n=6$, clipped at $0$;" "\n" r"floor: the same map, cones, and zeros of a resonant gas (dots)", fontsize=7.0, ha="center", va="top", linespacing=1.15)
    fig.savefig(os.path.join(OUT, "fig_complex.png"))
    plt.close(fig)


# ---------------------------------------------------------------- power counting
def omega_bar(L, n):
    return 4 * L - (2 * n + 2) * (L - 1) + 2 * max(L - n - 1, 0)


def fig_power():
    fig = plt.figure(figsize=(3.4, 2.9))
    ax = fig.add_axes([0.02, 0.0, 0.96, 0.9], projection="3d")
    Ls, ns = np.arange(1, 7), np.arange(3, 9)
    cmap = matplotlib.colormaps["GnBu_r"]
    for n in ns:
        for L in Ls:
            w = omega_bar(L, n)
            col = "#c0392b" if w > 0 else ("#f5b041" if w == 0 else cmap(0.15 + 0.6 * min(1, -w / 60)))
            h = w if w != 0 else 0.6
            ax.bar3d(L - 0.32, n - 0.32, 0, 0.64, 0.64, h, color=col, alpha=0.92, shade=True, edgecolor="k", linewidth=0.15)
    xx, yy = np.meshgrid([0.5, 6.5], [2.5, 8.5])
    ax.plot_surface(xx, yy, 0 * xx, color="0.6", alpha=0.18)
    ax.set_xlabel(r"loops $L$", labelpad=-5); ax.set_ylabel(r"$n$", labelpad=-6)
    ax.zaxis.set_rotate_label(False); ax.set_zlabel(r"$\bar\omega$", labelpad=-6, rotation=0)
    ax.set_xticks(Ls); ax.set_yticks(ns); ax.set_zticks([0, -20, -40, -60])
    style3d(ax, elev=22, azim=-52)
    fig.text(0.5, 0.955, r"$\bar\omega(L)=4L-(2n+2)(L-1)+2(L-n-1)_+$ for $m_\sigma=0$", fontsize=7.3, ha="center")
    fig.text(0.06, 0.885, "divergent ($L=1$)", color="#c0392b", fontsize=6.6)
    fig.text(0.06, 0.845, "marginal ($n=3$, $L=2$)", color="#b9770e", fontsize=6.6)
    fig.text(0.06, 0.805, "convergent", color="#1f618d", fontsize=6.6)
    fig.savefig(os.path.join(OUT, "fig_power.png"))
    plt.close(fig)



# ---------------------------------------------------------------- Newtonian limit
def inv_a_scalar(k, n):
    return float(np.exp(-0.5 * n * ein(np.array([k ** 4]))[0]))


def potential(r, n, K=40.0):
    """Phi_N(r) / (G m M) with M = 1: -(2/(pi r)) int_0^inf sin(kr)/(k a(k^2)) dk."""
    if r == 0:
        return -2 / math.pi * integrate.quad(lambda k: inv_a_scalar(k, n), 0, K, limit=400, epsabs=1e-14)[0]
    f = lambda k: inv_a_scalar(k, n) / k if k > 0 else 1.0
    val = integrate.quad(f, 0, K, weight="sin", wvar=r, limit=800, epsabs=1e-15, epsrel=1e-13)[0]
    return -2 / (math.pi * r) * val


def fig_newton():
    fig = plt.figure(figsize=(7.0, 3.0))
    # (a) the well for n = 4, with a quarter cut away to show the radial profile against Newton's funnel
    ax = fig.add_axes([-0.04, 0.0, 0.54, 0.86], projection="3d")
    n = 4
    rr = np.linspace(0, 6.0, 121)
    ph = np.array([potential(r, n) for r in rr])
    R, floor = 6.0, -1.6
    th0 = -0.5 * np.pi
    th = np.linspace(th0 + 0.5 * np.pi, th0 + 2.0 * np.pi, 91)     # three quarters; the front quarter is open
    Rg, Tg = np.meshgrid(rr, th)
    X, Y = Rg * np.cos(Tg), Rg * np.sin(Tg)
    Z = np.interp(Rg, rr, ph)
    norm = colors.Normalize(-0.55, -0.1)
    ax.plot_surface(X, Y, Z, facecolors=cm.viridis(norm(Z)), rstride=2, cstride=2, linewidth=0.1, edgecolor=(1, 1, 1, 0.15), shade=False, alpha=0.96)
    rn = np.linspace(1 / 1.6, R, 100)
    Rn, Tn = np.meshgrid(rn, th[::6])
    for r0 in (0.75, 1.0, 1.5, 2.5, 4.0, 6.0):
        t = np.linspace(th[0], th[-1], 120)
        ax.plot(r0 * np.cos(t), r0 * np.sin(t), np.full_like(t, -1 / r0), color="0.2", lw=0.45, alpha=0.75)
    for t0 in th[::10]:
        ax.plot(rn * np.cos(t0), rn * np.sin(t0), -1 / rn, color="0.2", lw=0.45, alpha=0.75)
    # the two cut faces: the profile Phi_N(r) (colored curtain) and Newton's -1/r (black)
    for t0 in (th[0], th[-1]):
        cx, cy = np.cos(t0), np.sin(t0)
        verts = [list(zip(rr * cx, rr * cy, ph)) + [(R * cx, R * cy, floor), (0, 0, floor)]]
        ax.add_collection3d(Poly3DCollection(verts, facecolor="#5dade2", edgecolor="none", alpha=0.18))
        ax.plot(rr * cx, rr * cy, ph, color="#1b4f72", lw=1.4)
        ax.plot(rn * cx, rn * cy, -1 / rn, color="k", lw=1.0, ls="--")
    lo, hi = 0.4852, 0.5303
    ax.plot([0, 0], [0, 0], [-hi, -lo], color="#c0392b", lw=3.2, solid_capstyle="butt", zorder=30)
    ax.plot([0, 0], [0, 0], [floor, -hi], color="#c0392b", lw=0.6, ls=":")
    ax.contour(X, Y, Z, levels=np.linspace(-0.5, -0.2, 7), zdir="z", offset=floor, cmap="viridis", norm=norm, linewidths=0.6)
    ax.set_zlim(floor, 0.0); ax.set_xlim(-R, R); ax.set_ylim(-R, R)
    ax.set_xticks([-6, -3, 0, 3, 6]); ax.set_yticks([-6, -3, 0, 3, 6]); ax.set_zticks([-1.5, -1.0, -0.5, 0.0])
    ax.set_xlabel(r"$Mx$", labelpad=-6); ax.set_ylabel(r"$My$", labelpad=-6)
    zlab(ax, r"$\Phi_N/(GmM)$", x=0.80, y=0.83)
    style3d(ax, elev=17, azim=-62)
    fig.text(0.01, 0.95, "(a)", fontsize=9)
    fig.text(0.05, 0.955, r"$n=4$: the well (surface, solid cut) is finite at $r=0$;" "\n"
             r"Newton's $-Gm/r$ (mesh, dashed) is not. Red bar: bracket on $\Phi_N(0)$", fontsize=6.8, ha="left", va="top", linespacing=1.15)

    # (b) waterfall of log10 |delta(r)| for n = 4..8
    ax = fig.add_axes([0.47, 0.0, 0.54, 0.86], projection="3d")
    pal = {4: "#1b4f72", 5: "#2874a6", 6: "#c0392b", 7: "#7d3c98", 8: "#117864"}
    lr = np.linspace(np.log10(2.0), np.log10(30.0), 110)
    base = -11.0
    for n, col in pal.items():
        d = np.array([abs(1 + potential(r, n) * r) for r in 10 ** lr])
        ld = np.clip(np.log10(np.maximum(d, 1e-30)), base, 0)
        ribbon(ax, lr, n, ld, col, base, alpha=0.13, lw=1.0)
    for p_, ls in ((4, ":"), (8, "-.")):
        ax.plot(lr, np.full_like(lr, 8.6), np.clip(np.log10(3e-2) - p_ * (lr - lr[0]), base, 0), color="0.3", lw=0.8, ls=ls)
        pass
    ax.set_zlim(base, 0); ax.set_ylim(3.6, 8.8)
    ax.set_xticks(np.log10([2, 5, 10, 20, 30])); ax.set_xticklabels(["2", "5", "10", "20", "30"])
    ax.set_yticks([4, 5, 6, 7, 8]); ax.set_zticks([-10, -8, -6, -4, -2, 0])
    ax.set_xlabel(r"$Mr$ (log scale)", labelpad=-5); ax.set_ylabel(r"$n$", labelpad=-6)
    zlab(ax, r"$\log_{10}|\delta(r)|$", x=0.80, y=0.83)
    style3d(ax, elev=24, azim=-60)
    fig.text(0.50, 0.95, "(b)", fontsize=9)
    fig.text(0.54, 0.955, r"$|\delta(r)|=|1+r\Phi_N/Gm|$ for $n=4,\dots,8$; the envelope falls faster" "\n"
             r"than every power of $1/Mr$; at the back $(Mr)^{-4}$ (dotted), $(Mr)^{-8}$ (dash-dot)", fontsize=6.8, ha="left", va="top", linespacing=1.15)
    fig.savefig(os.path.join(OUT, "fig_newton.png"))
    plt.close(fig)


# ---------------------------------------------------------------- chain bookkeeping
def _a_mp(z, n):
    """a(z) - 1 - c z^n and a(z) for real z >= 0, M = 1, in mpmath."""
    z = mp.mpf(z)
    c = mp.e ** (n * mp.euler / 2)
    if z * z < 30:
        ein_ = mp.e1(z * z) + mp.log(z * z) + mp.euler if z > 0 else mp.mpf(0)
        return mp.e ** (n * ein_ / 2) - 1 - c * z ** n
    return c * z ** n * mp.expm1(n * mp.e1(z * z) / 2) - 1


def _G(z, n):
    z = mp.mpf(z)
    if z == 0:
        return mp.mpf(0)
    return _a_mp(z, n) / z


def _dd(f, zs):
    """Divided difference f[z_0, ..., z_m] by the recursive formula (distinct points)."""
    if len(zs) == 1:
        return f(zs[0])
    return (_dd(f, zs[1:]) - _dd(f, zs[:-1])) / (zs[-1] - zs[0])


def fig_chain3d():
    mp.mp.dps = 60
    n = 4
    L1 = np.linspace(0.0, 4.0, 33)
    L2 = np.linspace(0.0, 4.0, 33) + 0.0125   # keeps q_1^2 and q_2^2 apart on the diagonal
    A, B = np.meshgrid(L1, L2, indexing="ij")
    up = np.empty_like(A); low = np.empty_like(A)
    for i, l1 in enumerate(L1):
        for j, l2 in enumerate(L2):
            lam1, lam2 = 10.0 ** l1, 10.0 ** l2
            q0 = np.array([0, 0, 1.0]); ell1 = np.array([lam1, 0, 0]); ell2 = np.array([-lam1, lam2, 0])
            q1 = q0 + ell1; q2 = q1 + ell2
            n0, n1, n2 = (np.linalg.norm(v) for v in (q0, q1, q2))
            Y = 1 + n0; X = 1 + n2
            B1 = (n0 + np.linalg.norm(ell1)) ** 2; B2 = (n1 + np.linalg.norm(ell2)) ** 2
            g = abs(_dd(lambda z: _G(z, n), [mp.mpf(n0 ** 2), mp.mpf(n1 ** 2), mp.mpf(n2 ** 2)]))
            size = Y * B1 * B2 * X
            legs = (1 + np.linalg.norm(ell1)) * (1 + np.linalg.norm(ell2))
            up[i, j] = math.log10(size)
            low[i, j] = math.log10(float(g) * size / legs)
    fig = plt.figure(figsize=(3.5, 3.0))
    ax = fig.add_axes([-0.06, -0.02, 1.05, 0.92], projection="3d")
    ax.plot_surface(A, B, up, cmap="magma_r", vmin=-4, vmax=up.max(), rstride=1, cstride=1, linewidth=0.15, edgecolor=(1, 1, 1, 0.25), alpha=0.9)
    ax.plot_surface(A, B, low, color="#e67e22", rstride=1, cstride=1, linewidth=0.15, edgecolor=(0.4, 0.2, 0, 0.35), alpha=0.95, shade=True)
    ax.contour(A, B, up, levels=[4, 8, 12, 16, 20], zdir="z", offset=-4, colors="0.55", linewidths=0.5)
    ax.set_zlim(-4, 22)
    ax.set_xlabel(r"$\log_{10}\lambda_1$", labelpad=-6); ax.set_ylabel(r"$\log_{10}\lambda_2$", labelpad=-6)
    ax.set_xticks([0, 1, 2, 3, 4]); ax.set_yticks([0, 1, 2, 3, 4]); ax.set_zticks([0, 5, 10, 15, 20])
    zlab(ax, r"$\log_{10}$", x=0.86, y=0.86)
    style3d(ax, elev=20, azim=-128)
    fig.text(0.02, 0.965, r"upper: $|Y|\,|B_1|\,|B_2|\,|X|$, up to $10^{%.0f}$" % up.max(), fontsize=7, va="top", color="#7b2d26")
    fig.text(0.02, 0.905, r"lower: full term$/\prod_a(1+|\ell_a|)$, in $[10^{%.1f},10^{%.1f}]$" % (low.min(), low.max()), fontsize=7, va="top", color="#a04000")
    fig.savefig(os.path.join(OUT, "fig_chain3d.png"))
    plt.close(fig)
    print("chain: upper max %.2f, lower range [%.3f, %.3f]" % (up.max(), low.min(), low.max()))


# ---------------------------------------------------------------- admissible class
def fig_class():
    fig = plt.figure(figsize=(7.0, 2.75))
    resp = [(r"(a) $1-\psi=e^{-w^2}$", lambda w: np.exp(-w ** 2), np.pi / 4, 3.0),
            (r"(b) $1-\psi=(1+w^2)e^{-w^2}$", lambda w: (1 + w ** 2) * np.exp(-w ** 2), np.pi / 4, 3.0),
            (r"(c) $1-\psi=e^{-w^4}$", lambda w: np.exp(-w ** 4), np.pi / 8, 1.7)]
    lo, hi = -8.0, 3.0
    norm = colors.Normalize(lo, hi)
    for k, (title, f, th, box) in enumerate(resp):
        g = np.linspace(-box, box, 181)
        U, V = np.meshgrid(g, g)
        W = U + 1j * V
        ax = fig.add_axes([0.005 + 0.33 * k, 0.0, 0.33, 0.84], projection="3d")
        with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
            Z = np.log10(np.abs(f(W)) + 1e-300)
        Zc = np.clip(Z, lo, hi)
        Zm = np.where(Z > hi, np.nan, Zc)       # the surface stops where the response is far from saturation
        ax.plot_surface(U, V, Zm, facecolors=cm.RdYlBu_r(norm(np.nan_to_num(Zm, nan=hi))), rstride=2, cstride=2,
                        linewidth=0, antialiased=True, shade=False, alpha=0.92)
        fl = lo - 3.5
        ax.contourf(U, V, Zc, levels=np.linspace(lo, hi, 12), zdir="z", offset=fl, cmap="RdYlBu_r", norm=norm, alpha=0.85)
        rr = np.linspace(0, box * 1.0, 20)
        for sgn in (1, -1):
            for ang in (th, np.pi - th):
                ax.plot(rr * np.cos(sgn * ang), rr * np.sin(sgn * ang), np.full_like(rr, fl), color="k", lw=0.9, ls="--", zorder=20)
        ax.set_zlim(fl, hi); ax.set_xlim(-box, box); ax.set_ylim(-box, box)
        tk = [-2, 0, 2] if box > 2 else [-1.5, 0, 1.5]
        ax.set_xticks(tk); ax.set_yticks(tk); ax.set_zticks([-8, -4, 0])
        ax.set_xlabel(r"$\mathrm{Re}\,w$", labelpad=-6); ax.set_ylabel(r"$\mathrm{Im}\,w$", labelpad=-6)
        zlab(ax, r"$\log_{10}|1-\psi|$", x=0.70, y=0.86)
        style3d(ax, elev=27, azim=-62)
        fig.text(0.02 + 0.33 * k, 0.955, title, fontsize=7.6, ha="left", va="top")
        fig.text(0.02 + 0.33 * k, 0.885, r"sector $|\arg(\pm w)|<%s$ (dashed)" % (r"\pi/4" if k < 2 else r"\pi/8"), fontsize=6.6, ha="left", va="top", color="0.25")
    fig.savefig(os.path.join(OUT, "fig_class.png"))
    plt.close(fig)


# ---------------------------------------------------------------- vacuum energy
def _remainder(W, p):
    """int_0^inf Ein(w) r(w/W) dw minus the growing terms, r(u) = exp(-u^p); p = inf is the sharp cutoff."""
    if np.isinf(p):
        return 1 - math.exp(-W) + W * special.exp1(W)
    umax = 60.0 ** (1 / p)
    f = lambda u: ein(np.array([W * u]))[0] * math.exp(-u ** p)
    pts = [0.5, 1.0, 1.5] if umax > 1.5 else [0.5, 1.0]
    I = W * integrate.quad(f, 0, umax, points=pts, limit=400, epsabs=1e-12, epsrel=1e-12)[0]
    rt1 = special.gamma(1 / p) / p
    rtp1 = special.digamma(1 / p) * special.gamma(1 / p) / p ** 2
    return I - rt1 * W * math.log(W) - (rtp1 + G_E * rt1) * W


def fig_vacuum():
    fig = plt.figure(figsize=(7.0, 3.0))
    # (a) |Gamma(s)/s| over the complex s plane
    ax = fig.add_axes([-0.03, 0.0, 0.52, 0.86], projection="3d")
    ax.computed_zorder = False
    x = np.linspace(-3.6, 2.2, 233); y = np.linspace(-2.0, 2.0, 161)
    X, Y = np.meshgrid(x, y)
    S = X + 1j * Y
    with np.errstate(all="ignore"):
        Z = np.log10(np.abs(special.gamma(S) / S))
    lo, hi = -2.0, 2.5
    Z = np.clip(np.nan_to_num(Z, nan=hi, posinf=hi), lo, hi)
    norm = colors.Normalize(lo, hi)
    floor = lo - 1.5
    ax.contourf(X, Y, Z, levels=np.linspace(lo, hi, 10), zdir="z", offset=floor, cmap="Spectral_r", norm=norm, alpha=0.55, zorder=1)
    strip = [[(-1, -2, floor), (0, -2, floor), (0, 2, floor), (-1, 2, floor)]]
    ax.add_collection3d(Poly3DCollection(strip, facecolor="#f5b041", edgecolor="#b9770e", alpha=0.55, linewidths=0.8, zorder=2))
    ax.plot_surface(X, Y, Z, facecolors=cm.Spectral_r(norm(Z)), rstride=2, cstride=2, linewidth=0, antialiased=True, shade=False, alpha=0.82, zorder=3)
    ax.plot(x, np.zeros_like(x), np.clip(np.log10(np.abs(special.gamma(x) / x)), lo, hi), color="0.15", lw=0.8, zorder=4)
    ax.plot([1, 1], [0, 0], [floor, 0.0], color="#c0392b", lw=1.0, zorder=5)
    ax.scatter([1], [0], [0.0], color="#c0392b", s=22, depthshade=False, zorder=6)
    ax.set_zlim(floor, hi); ax.set_xlim(-3.6, 2.2); ax.set_ylim(-2, 2)
    ax.set_xticks([-3, -2, -1, 0, 1, 2]); ax.set_yticks([-2, 0, 2]); ax.set_zticks([-2, -1, 0, 1, 2])
    ax.set_xlabel(r"$\mathrm{Re}\,s$", labelpad=-5); ax.set_ylabel(r"$\mathrm{Im}\,s$", labelpad=-5)
    zlab(ax, r"$\log_{10}|\Gamma(s)/s|$", x=0.80, y=0.84)
    style3d(ax, elev=24, azim=-58)
    fig.text(0.01, 0.95, "(a)", fontsize=9)
    fig.text(0.05, 0.955, r"Mellin transform of Ein: converges in $-1<\mathrm{Re}\,s<0$ (orange band)," "\n"
             r"double pole at $0$, simple poles at $-1,-2,\ldots$; red: $\Gamma(1)=1$ at $s=1$", fontsize=6.8, ha="left", va="top", linespacing=1.15)

    # (b) remainder after subtraction, for r(u) = exp(-u^p)
    ax = fig.add_axes([0.47, 0.0, 0.54, 0.86], projection="3d")
    lW = np.linspace(-1.0, 3.0, 41)
    ip = np.linspace(0.0, 1.0, 21)     # 1/p; the edge 1/p = 0 is the sharp cutoff
    A, B = np.meshgrid(lW, ip, indexing="ij")
    R = np.empty_like(A)
    for i, l in enumerate(lW):
        for j, q in enumerate(ip):
            R[i, j] = _remainder(10 ** l, np.inf if q == 0 else 1 / q)
    norm = colors.Normalize(-0.2, 1.4)
    ax.plot_surface(A, B, R, facecolors=cm.viridis(norm(R)), rstride=1, cstride=1, linewidth=0.15, edgecolor=(1, 1, 1, 0.3), shade=False, alpha=0.92)
    ax.plot(lW, np.zeros_like(lW), R[:, 0], color="#c0392b", lw=1.4)
    pl = [[(-1, 0, 1), (3, 0, 1), (3, 1, 1), (-1, 1, 1)]]
    ax.add_collection3d(Poly3DCollection(pl, facecolor="0.5", edgecolor="0.3", alpha=0.10, linewidths=0.5))
    ax.set_xlabel(r"$\log_{10}W$", labelpad=-5); ax.set_ylabel(r"$1/p$", labelpad=-5)
    ax.set_xticks([-1, 0, 1, 2, 3]); ax.set_yticks([0, 0.5, 1]); ax.set_zticks([0, 0.5, 1.0])
    zlab(ax, "remainder", x=0.82, y=0.84)
    style3d(ax, elev=24, azim=-122)
    fig.text(0.50, 0.95, "(b)", fontsize=9)
    fig.text(0.54, 0.955, r"cutoffs $r(u)=e^{-u^p}$: the remainder tends to $1$ (gray plane)" "\n"
             r"for every $p$; red edge, $p=\infty$: $1-e^{-W}+W E_1(W)$", fontsize=6.8, ha="left", va="top", linespacing=1.15)
    fig.savefig(os.path.join(OUT, "fig_vacuum.png"))
    plt.close(fig)
    print("vacuum remainder at W=1e3: min %.5f max %.5f" % (R[-1].min(), R[-1].max()))


# ---------------------------------------------------------------- unitarity
def fig_unitarity():
    fig = plt.figure(figsize=(3.5, 3.2))
    ax = fig.add_axes([-0.06, 0.02, 1.06, 0.86], projection="3d")
    n, kk = 4, 1.2
    g = np.linspace(-1.8, 1.8, 361)     # contains the poles k0 = +-1.2 exactly
    U, V = np.meshgrid(g, g)
    K0 = U + 1j * V
    z = kk ** 2 - K0 ** 2
    with np.errstate(all="ignore"):
        L = -np.log10(np.abs(z)) - 0.5 * n * np.real(ein_c(z * z)) / math.log(10)
        amp = 0.5 * n * np.abs(special.exp1(z * z)) / math.log(10)   # size of the oscillating part of log10|a|
    wild = ~(amp < 6) & (np.abs(z) > 0.5)
    L = np.clip(np.nan_to_num(L, nan=6, posinf=6, neginf=-6), -6, 6)
    Lm = np.where(wild, np.nan, L)
    norm = colors.Normalize(-6, 6)
    ax.plot_surface(U, V, Lm, facecolors=cm.coolwarm(norm(np.nan_to_num(Lm, nan=0))), rstride=2, cstride=2, linewidth=0, antialiased=True, shade=False, alpha=0.92)
    ax.contourf(U, V, Lm, levels=np.linspace(-6, 6, 13), zdir="z", offset=-10, cmap="coolwarm", norm=norm, alpha=0.8)
    ax.contourf(U, V, wild.astype(float), levels=[0.5, 1.5], zdir="z", offset=-10, colors=["0.35"], alpha=0.45)
    t = np.linspace(-1.8, 1.8, 80)
    ax.plot(np.zeros_like(t), t, np.full_like(t, -10), color="white", lw=2.0, zorder=20)
    ax.plot(np.zeros_like(t), t, np.clip(-np.log10(kk ** 2 + t ** 2) - 0.5 * n * ein(((kk ** 2 + t ** 2)) ** 2) / math.log(10), -6, 6), color="white", lw=1.4, zorder=21)
    ax.plot(t, np.zeros_like(t), np.full_like(t, -10), color="#c0392b", lw=1.4, zorder=20)
    ax.scatter([kk, -kk], [0, 0], [-10, -10], color="k", s=10, depthshade=False, zorder=22)
    ax.set_zlim(-10, 6)
    ax.set_xticks([-1.5, 0, 1.5]); ax.set_yticks([-1.5, 0, 1.5]); ax.set_zticks([-6, -3, 0, 3, 6])
    ax.set_xlabel(r"$\mathrm{Re}\,k^0/M$", labelpad=-6); ax.set_ylabel(r"$\mathrm{Im}\,k^0/M$", labelpad=-6)
    zlab(ax, r"$\log_{10}|z\,\hat a(z)|^{-1}$", x=0.70, y=0.86)
    style3d(ax, elev=30, azim=-58)
    fig.text(0.02, 0.965, "(a)", fontsize=9, va="top")
    fig.text(0.10, 0.965, r"$n=4$, $|\mathbf{k}|=1.2M$, $z=\mathbf{k}^2-(k^0)^2$" "\n" r"white: Euclidean contour; red: real axis;" "\n" r"gray wedges: $|\hat a|^{\pm1}$ oscillates beyond $10^{\pm6}$", fontsize=6.8, va="top", linespacing=1.15)
    fig.savefig(os.path.join(OUT, "fig_unitarity.png"))
    plt.close(fig)


# ---------------------------------------------------------------- why (G) needs zeta > 4
def cut_var(lam, zeta):
    """lam^(4-zeta) int_0^lam v^(zeta-5) (1-exp(-v^2))^2 dv: the variance of the realization's part
    of the zero-point energy with a sharp cutoff, lam = (Lambda/M)^2, in units K^2 n th0 (psi_*)."""
    lam = np.atleast_1d(np.asarray(lam, dtype=float))
    x = np.linspace(-9.0, math.log(lam.max()), 20001)
    f = np.exp((zeta - 4) * x) * (-np.expm1(-np.exp(2 * x))) ** 2
    cum = integrate.cumulative_trapezoid(f, x, initial=0.0) + math.exp((zeta) * -9.0) / zeta
    return lam ** (4 - zeta) * np.interp(np.log(lam), x, cum)


def C_zeta(zeta):
    """closed form of the limit for 0 < zeta < 4, Eq. (cutvar)"""
    if abs(zeta - 2) < 1e-12:
        return math.log(2)
    b = zeta / 2 - 2
    return 0.5 * special.gamma(b) * (2 ** (-b) - 2)


def rho_paths(rng_, zeta, lam, n=4, tau_c=0.3, tau_min=1e-9, h=0.01):
    """delta rho_Lambda / K for one realization: exact branes above tau_c, Gaussian shells below."""
    lam = np.asarray(lam, dtype=float)
    m = lambda U: -np.expm1(-U ** 2)
    N = rng_.poisson((n / TH0) * (tau_c ** -zeta - 1) / zeta)
    A = tau_c ** -zeta
    tau = (A - (A - 1) * rng_.random(N)) ** (-1 / zeta)
    g = -TH0 * tau ** (zeta - 2)
    out = (g[:, None] * m(np.outer(tau, lam))).sum(0)
    s = np.linspace(math.log(tau_c), 0.0, 4001); ts = np.exp(s)
    out += n * integrate.trapezoid(ts[None, :] ** -2 * m(np.outer(lam, ts)), s, axis=1)   # minus the mean
    L = np.arange(math.log(tau_min), math.log(tau_c), h) + h / 2
    tg = np.exp(L)
    S = np.sqrt(n * TH0 * tg ** (zeta - 4) * h) * rng_.standard_normal(len(L))
    out += (S[:, None] * m(np.outer(tg, lam))).sum(0)
    return out


def fig_necessity():
    fig = plt.figure(figsize=(7.0, 2.75))
    n = 4
    # (a) log10 of the standard deviation over (log10 Lambda/M, zeta)
    ax = fig.add_axes([0.0, 0.0, 0.34, 0.84], projection="3d")
    ll = np.linspace(0.0, 4.0, 81); zg = np.linspace(2.0, 9.0, 71)
    S = np.array([0.5 * np.log10(cut_var(10.0 ** (2 * ll), z)) for z in zg]).T
    LL, ZG = np.meshgrid(ll, zg, indexing="ij")
    norm = colors.Normalize(-0.5, 4.0)
    ax.plot_surface(LL, ZG, S, facecolors=cm.plasma(norm(S)), rstride=2, cstride=2, linewidth=0.1, edgecolor=(1, 1, 1, 0.25), shade=False, alpha=0.93)
    ax.plot(ll, np.full_like(ll, 4.0), 0.5 * np.log10(cut_var(10.0 ** (2 * ll), 4.0)) + 0.04, color="#c0392b", lw=1.6, zorder=20)
    for z in (2.0, 3.0):
        ax.plot(ll, np.full_like(ll, z), 0.5 * np.log10(C_zeta(z)) + (4 - z) * ll, color="k", lw=0.8, ls=":", zorder=21)
    for z in (6.0, 9.0):
        ax.plot(ll, np.full_like(ll, z), np.full_like(ll, -0.5 * math.log10(z - 4)), color="w", lw=0.8, ls="--", zorder=21)
    ax.set_xlabel(r"$\log_{10}(\Lambda/M)$", labelpad=-5); ax.set_ylabel(r"$\zeta$", labelpad=-5)
    ax.set_xticks([0, 1, 2, 3, 4]); ax.set_yticks([2, 4, 6, 8]); ax.set_zticks([0, 2, 4, 6, 8])
    zlab(ax, r"$\log_{10}$ s.d. of $\delta\rho_\Lambda$", x=0.62, y=0.86)
    style3d(ax, elev=22, azim=-62)
    fig.text(0.01, 0.95, "(a)", fontsize=9)
    fig.text(0.05, 0.955, r"s.d. in units $K(n\vartheta_0)^{1/2}$: slope $4-\zeta$ for $\zeta<4$" "\n" r"(dotted, Eq. (cutvar)); red: $\zeta=4$, $\log$ growth;" "\n" r"white: limit $(\zeta-4)^{-1/2}$ for $\zeta>4$", fontsize=6.6, ha="left", va="top", linespacing=1.15)

    # (b), (c) sample paths for zeta = 6 and zeta = 3
    lx = np.linspace(0.0, 3.0, 151); lam = 10.0 ** (2 * lx)
    r_ = np.random.default_rng(31)
    comp = lambda v: np.sign(v) * np.log10(1 + np.abs(v))
    for k, (zeta, x0, lab, cmapn, band, edge) in enumerate([(6.0, 0.33, "(b)", "Blues", "#aed6f1", "#1b4f72"),
                                                             (3.0, 0.655, "(c)", "Oranges", "#f5cba7", "#a04000")]):
        ax = fig.add_axes([x0, 0.0, 0.33, 0.84], projection="3d")
        sd = np.sqrt(n * TH0 * cut_var(lam, zeta))
        pal = matplotlib.colormaps[cmapn](np.linspace(0.45, 0.92, 12))
        for sgn in (1, -1):
            Y, X_ = np.meshgrid([0.5, 12.5], lx)
            ax.plot_surface(X_, Y, np.outer(comp(sgn * sd), [1, 1]), color=band, alpha=0.15, linewidth=0, shade=False)
        for j in range(12):
            p = rho_paths(r_, zeta, lam, n)
            ax.plot(lx, np.full_like(lx, j + 1), comp(p), color=pal[j], lw=0.9)
            if zeta > 4:
                ax.scatter([lx[-1]], [j + 1], [comp(p[-1])], s=9, color="#c0392b", depthshade=False, zorder=20)
        for y0 in (0.5, 12.5):
            ax.plot(lx, np.full_like(lx, y0), comp(sd), color=edge, lw=0.8, ls="--")
            ax.plot(lx, np.full_like(lx, y0), comp(-sd), color=edge, lw=0.8, ls="--")
        ax.set_xlabel(r"$\log_{10}(\Lambda/M)$", labelpad=-5); ax.set_ylabel("realization", labelpad=-5)
        ax.set_xticks([0, 1, 2, 3]); ax.set_yticks([1, 4, 8, 12])
        ax.set_zlim(-4, 4); ax.set_zticks([-4, -2, 0, 2, 4])
        zlab(ax, r"$\mathrm{sgn}(v)\log_{10}(1+|v|)$, $v=\delta\rho_\Lambda/K$", x=0.42, y=0.86)
        style3d(ax, elev=22, azim=-60)
        fig.text(x0 + 0.015, 0.95, lab, fontsize=9)
        txt = (r"$\zeta=6$: every path settles to its own" "\n" r"finite $\delta\rho_\Pi$ (red dots)") if zeta > 4 else \
              (r"$\zeta=3$: the band widens like $\Lambda/M$;" "\n" r"the paths wander without limit")
        fig.text(x0 + 0.055, 0.955, txt, fontsize=6.6, ha="left", va="top", linespacing=1.15)
    fig.savefig(os.path.join(OUT, "fig_necessity.png"))
    plt.close(fig)
    print("C_3 = %.6f (closed form), check %.6f" % (C_zeta(3.0), cut_var(1e12, 3.0)[0] / 1e12))


# ---------------------------------------------------------------- the one-loop bubble
def bubble_logF2(xi, n=4):
    """log |F(xi)|^2 for the skeleton propagator F = 1/(xi a(xi)), M = 1"""
    return -n * np.real(ein_c(xi * xi)) - 2 * np.log(np.abs(xi))


def ein_point(w):
    """entire exponential integral at one complex point"""
    if abs(w) < 2.0:
        acc = 0j; term = 1 + 0j
        for k in range(1, 50):
            term *= w / k; acc += (term if k % 2 else -term) / k
        return acc
    return cmath.log(w) + G_E + complex(special.exp1(w))


def bubble_ReA(E, n=4):
    """Re A for the skeleton bubble at energy E (M = 1), Eq. (bubbledec) with the Pauli-Villars mass
    mu = 2E + 2: closed-form A_mu, plus the real section in polar coordinates about the focus xi = 0
    of the region P_s, where it ends at R = s/(2(1 - cos phi)), plus the residue term.  Returns log10 Re A."""
    s = E * E; mu2 = (2 * E + 2) ** 2
    F = lambda z: cmath.exp(-0.5 * n * ein_point(z * z)) / z
    Fp = lambda z: 1 / z - 1 / (z + mu2)

    def inner(ph):
        Rmax = s / (2 * (1 - math.cos(ph)))
        def f(R):
            xi = R * cmath.exp(1j * ph)
            rho = math.sqrt(max(R * math.cos(ph) + s / 4 - (R * math.sin(ph)) ** 2 / s, 0.0))
            return rho * (abs(F(xi)) ** 2 - abs(Fp(xi)) ** 2) * R
        Rc = min(Rmax, 20.0 * s + 20.0)
        val = integrate.quad(f, 0, Rc, limit=200, epsabs=1e-12, epsrel=1e-10)[0]
        if Rmax > Rc:
            val += integrate.quad(f, Rc, Rmax, limit=200, epsabs=1e-12, epsrel=1e-10)[0]
        return val
    sect = 2 * (2 * math.pi / E) * integrate.quad(inner, 0, math.pi, limit=200, epsabs=1e-11, epsrel=1e-9)[0]
    res = (2 * math.pi ** 2 / s) * integrate.quad(lambda w: (w + s) * (F(complex(w)) - Fp(complex(w))).real, -s, 0, limit=200)[0]
    P = mp.mpc(-s, 1e-40)
    g = lambda x: (-mp.log(x * (1 - x) * P) + mp.log(x * (1 - x) * P + x * mu2) + mp.log(x * (1 - x) * P + (1 - x) * mu2)
                   - mp.log(x * (1 - x) * P + mu2))
    apv = float(mp.re(mp.pi ** 2 * mp.quad(g, [0, 0.5, 1])))
    return math.log10(apv + sect + res)


def ein_minus(X):
    return float(mp.ei(X) - mp.log(X) - mp.euler)


def bubble_lower(E, n=4):
    """log10 of the rigorous lower bound, maximized over delta (the power-law remainder is dropped)"""
    s = E * E
    best = -np.inf
    for d in np.linspace(0.01, 0.6, 240):
        v = math.log(math.pi ** 2 * math.sqrt(7) * d ** 2.5 / (64 * (1 - d) ** 2)) + n * ein_minus((1 - d) ** 2 * s * s / 4)
        best = max(best, v)
    return best / math.log(10)


def fig_bubble():
    fig = plt.figure(figsize=(7.0, 2.9))
    n = 4
    # (a) the real section in the xi plane at E = 1.7 M: log10 of the weight rho |F|^2 over the region P_s
    ax = fig.add_axes([-0.06, 0.03, 0.64, 0.88], projection="3d")
    ax.computed_zorder = False
    E = 1.7; s = E * E
    X = np.linspace(-0.85, 1.6, 246); Y = np.linspace(0.0, 2.6, 261)
    XX, YY = np.meshgrid(X, Y, indexing="ij")
    rho2 = XX + s / 4 - YY ** 2 / s
    xi = XX + 1j * YY
    with np.errstate(all="ignore"):
        Z = np.log10(2 * np.pi / E * np.sqrt(np.clip(rho2, 0, None))) + bubble_logF2(xi, n) / math.log(10)
    lo, hi = -6.0, 9.0
    Z = np.where(rho2 > 0, np.clip(np.nan_to_num(Z, nan=lo, neginf=lo, posinf=hi), lo, hi), np.nan)
    norm = colors.Normalize(lo, hi)
    fl = lo - 1.5
    ax.contourf(XX, YY, np.nan_to_num(Z, nan=lo - 5), levels=np.linspace(lo, hi, 14), zdir="z", offset=fl, cmap="magma", norm=norm, alpha=0.7, zorder=1)
    yb = np.linspace(0, 2.6, 200)
    ax.plot(yb ** 2 / s - s / 4, yb, np.full_like(yb, fl), color="k", lw=1.0, zorder=2)
    ax.plot([0, 0], [0, s / 2], [fl, fl], color="#2e86c1", lw=1.6, zorder=3)
    d = 0.2; c0 = (1 - d) * s / 2; rad = d * s / 8
    ph = np.linspace(0, 2 * np.pi, 100)
    ax.plot(rad * np.cos(ph), c0 + rad * np.sin(ph), np.full_like(ph, fl), color="#27ae60", lw=1.4, zorder=4)
    ax.scatter([0], [0], [fl], color="w", edgecolor="k", s=16, depthshade=False, zorder=5)
    ax.plot_surface(XX, YY, Z, facecolors=cm.magma(norm(np.nan_to_num(Z, nan=lo))), rstride=2, cstride=2, linewidth=0, antialiased=True, shade=False, alpha=0.92, zorder=6)
    # the segment xi = iy and the disk of the bound, lifted onto the surface
    ys = np.linspace(0.12, s / 2 - 0.01, 120)
    zs = np.log10(2 * np.pi / E * np.sqrt(s / 4 - ys ** 2 / s)) + bubble_logF2(1j * ys, n) / math.log(10)
    ax.plot(np.zeros_like(ys), ys, np.clip(zs, lo, hi) + 0.08, color="#5dade2", lw=1.6, zorder=7)
    xd, yd = rad * np.cos(ph), c0 + rad * np.sin(ph)
    zd = np.log10(2 * np.pi / E * np.sqrt(xd + s / 4 - yd ** 2 / s)) + bubble_logF2(xd + 1j * yd, n) / math.log(10)
    ax.plot(xd, yd, zd + 0.08, color="#58d68d", lw=1.6, zorder=8)
    ax.set_zlim(fl, hi); ax.set_xlim(-0.85, 1.6); ax.set_ylim(0, 2.6)
    ax.set_xticks([-0.5, 0, 0.5, 1, 1.5]); ax.set_yticks([0, 1, 2]); ax.set_zticks([-6, -3, 0, 3, 6, 9])
    ax.set_xlabel(r"$\mathrm{Re}\,\xi/M^2$", labelpad=-5); ax.set_ylabel(r"$\mathrm{Im}\,\xi/M^2$", labelpad=-5)
    zlab(ax, r"$\log_{10}\,\frac{2\pi}{E}\rho\,|F(\xi)|^2$", x=0.73, y=0.84)
    style3d(ax, elev=27, azim=-58)
    fig.text(0.01, 0.95, "(a)", fontsize=9)
    fig.text(0.05, 0.955, r"$E=1.7M$: integrand of the real section over $P_s$, clipped to $[10^{-6},10^{9}]$" "\n"
             r"(black: edge $\mathrm{Re}\,\xi=(\mathrm{Im}\,\xi)^2/s-s/4$; blue: $\xi=iy$, $0<y<s/2$; green: disk" "\n"
             r"of the lower bound, $\delta=0.2$; white dot: graviton pole $\xi=0$)", fontsize=6.6, ha="left", va="top", linespacing=1.15)

    # (b) Re A against E, with the rigorous lower bound, |Im A| and the scales where the graph beats the tree
    ax = fig.add_axes([0.625, 0.15, 0.36, 0.70])
    Es = np.round(np.arange(0.25, 1.951, 0.05), 3)
    lv = np.array([bubble_ReA(e, n) for e in Es])
    ax.plot(Es, lv, color="#1b4f72", lw=1.4, marker="o", ms=2.3, label=r"$\mathrm{Re}\,A$ (quadrature)")
    El = np.linspace(1.2, 1.95, 60)
    ax.plot(El, [bubble_lower(e, n) for e in El], color="#27ae60", lw=1.1, ls="--", label=r"lower bound (leading term)")
    ax.axhline(math.log10(math.pi ** 3), color="#c0392b", lw=1.0, ls="-.", label=r"$|\mathrm{Im}\,A|=\pi^3$ (local cut)")
    Eg = np.linspace(0.8, 1.95, 200)
    for k, ratio in enumerate([3, 19, 40]):
        y = 2 * ratio - np.log10(Eg ** 2 * a_real(Eg ** 2, n))
        ax.plot(Eg, y, color="0.45", lw=0.7, ls=":")
        ax.text(1.0, y[np.argmin(np.abs(Eg - 1.0))] + 1.5, r"$M/M_{\rm P}=10^{-%d}$" % ratio, fontsize=6.0, color="0.35")
    ax.set_xlim(0.2, 2.0); ax.set_ylim(-3, 165)
    ax.set_xlabel(r"$E/M$"); ax.set_ylabel(r"$\log_{10}\mathrm{Re}\,A$")
    ax.legend(loc="upper left", frameon=False, fontsize=6.3)
    fig.text(0.575, 0.955, "(b)", fontsize=9)
    fig.text(0.61, 0.955, r"dotted: $Gs\,a(-s)\,\mathrm{Re}\,A=1$, where the graph reaches" "\n" r"the tree amplitude (up to angular factors)", fontsize=6.6, ha="left", va="top", linespacing=1.15)
    fig.savefig(os.path.join(OUT, "fig_bubble.png"))
    plt.close(fig)
    print("bubble: log10 Re A at E =", dict(zip(Es.tolist(), np.round(lv, 3).tolist())))


ALL = ["gas", "fluct", "formfactor", "complex", "power", "newton", "chain3d", "class", "vacuum", "unitarity", "necessity", "bubble"]

if __name__ == "__main__":
    import sys
    for w in sys.argv[1:] or ALL:
        globals()["fig_" + w]()
        print("wrote", w)
