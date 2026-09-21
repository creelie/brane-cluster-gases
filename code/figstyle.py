"""
figstyle.py -- drawing conventions shared by every figure script of the paper,
and a label collision test that each figure has to pass before it is written
to disk.

Palette: a small fixed categorical set (validated for colour-vision deficiency
on a white surface), assigned in a fixed order and never cycled; one sequential
hue for magnitudes; neutral inks for text and axes.  Labels are placed by hand.
"""
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")

# Embedding a font in the vector output makes fontTools read the header of the
# system font file and report on the date stamp it finds there.  That has no
# bearing on the figure, so the message is kept out of the console.
logging.getLogger("fontTools").setLevel(logging.ERROR)
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import LightSource, to_rgb
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

# categorical slots, fixed order
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
MAGENTA = "#e87ba4"
VIOLET = "#4a3aa7"
SERIES = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, VIOLET]

INK = "#0b0b0b"
INK2 = "#52514e"
INK3 = "#8b8a85"
GRID = "#d8d7d2"
SURFACE = "#ffffff"
PANEL = "#f4f3ef"

DPI = 600

rcParams.update({
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 8.5,
    "axes.edgecolor": INK2,
    "axes.labelcolor": INK,
    "axes.linewidth": 0.7,
    "axes.titlesize": 9,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "text.color": INK,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "legend.frameon": False,
    "legend.fontsize": 7.5,
    "lines.linewidth": 1.4,
    "lines.solid_capstyle": "round",
    "pdf.fonttype": 42,
})


def panel_label(ax, s, x=-0.02, y=1.03):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=10, fontweight="bold", ha="right", va="bottom", color=INK)


def clean_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, axis="y", zorder=0)
    ax.set_axisbelow(True)


def blank_3d(ax):
    ax.set_axis_off()
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass


def shade(color, ls, normals, base=0.55, spread=0.5):
    """flat shading of a colour by a light direction on unit normals"""
    c = np.array(to_rgb(color))
    lam = np.clip(normals @ ls, 0, 1)
    f = base + spread * lam
    out = np.clip(c[None, :] * f[:, None], 0, 1)
    return out


def surface_faces(X, Y, Z):
    """quads and unit normals of a parametrised surface given on a grid"""
    faces = []
    normals = []
    m, n = X.shape
    for i in range(m - 1):
        for j in range(n - 1):
            p = np.array([[X[i, j], Y[i, j], Z[i, j]], [X[i + 1, j], Y[i + 1, j], Z[i + 1, j]],
                          [X[i + 1, j + 1], Y[i + 1, j + 1], Z[i + 1, j + 1]], [X[i, j + 1], Y[i, j + 1], Z[i, j + 1]]])
            nrm = np.cross(p[1] - p[0], p[3] - p[0])
            nn = np.linalg.norm(nrm)
            normals.append(nrm / nn if nn > 0 else np.array([0, 0, 1.0]))
            faces.append(p)
    return faces, np.array(normals)


def add_surface(ax, X, Y, Z, color, light=(0.3, -0.6, 0.75), alpha=1.0, edge=None, base=0.55, spread=0.5, zorder=1):
    faces, normals = surface_faces(X, Y, Z)
    ls = np.array(light) / np.linalg.norm(light)
    cols = shade(color, ls, np.abs(normals) if False else normals, base, spread)
    # two-sided lighting: use |n.l| so the back of a sheet is not black
    lam = np.abs(normals @ ls)
    c = np.array(to_rgb(color))
    cols = np.clip(c[None, :] * (base + spread * lam)[:, None], 0, 1)
    if alpha < 1.0:
        rgba = np.column_stack([cols, np.full(len(cols), alpha)])
        coll = Poly3DCollection(faces, facecolors=rgba, edgecolors="none", linewidths=0.0, zorder=zorder)
    else:
        coll = Poly3DCollection(faces, facecolors=cols, edgecolors=edge if edge else cols, linewidths=0.15, zorder=zorder)
    ax.add_collection3d(coll)
    return coll


def tube(ax, pts, radius, color, nseg=10, light=(0.3, -0.6, 0.75), alpha=1.0, closed=True):
    """draw a curve as a shaded tube; pts is (N,3)"""
    pts = np.asarray(pts)
    N = len(pts)
    if closed:
        P = np.vstack([pts, pts[:1]])
    else:
        P = pts
    T = np.gradient(P, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    # parallel-ish transport of a normal frame
    n0 = np.cross(T[0], [0, 0, 1.0])
    if np.linalg.norm(n0) < 1e-6:
        n0 = np.cross(T[0], [0, 1.0, 0])
    n0 /= np.linalg.norm(n0)
    Ns = [n0]
    for k in range(1, len(P)):
        v = Ns[-1] - np.dot(Ns[-1], T[k]) * T[k]
        v /= np.linalg.norm(v)
        Ns.append(v)
    Ns = np.array(Ns)
    Bs = np.cross(T, Ns)
    th = np.linspace(0, 2 * np.pi, nseg + 1)
    X = P[:, None, 0] + radius * (np.cos(th)[None, :] * Ns[:, None, 0] + np.sin(th)[None, :] * Bs[:, None, 0])
    Y = P[:, None, 1] + radius * (np.cos(th)[None, :] * Ns[:, None, 1] + np.sin(th)[None, :] * Bs[:, None, 1])
    Z = P[:, None, 2] + radius * (np.cos(th)[None, :] * Ns[:, None, 2] + np.sin(th)[None, :] * Bs[:, None, 2])
    return add_surface(ax, X, Y, Z, color, light=light, alpha=alpha, base=0.5, spread=0.55, zorder=3)


# ----------------------------------------------------------------------------
# collision test
# ----------------------------------------------------------------------------
INSET = 1.5
EDGE_FRACTION = 0.05
EDGE_LEVEL = 20


def _texts(fig):
    r = fig.canvas.get_renderer()
    out = list(fig.texts)
    for ax in fig.get_axes():
        out.extend(ax.texts)
        if getattr(ax, "axison", True) and not hasattr(ax, "get_zlim"):
            out += [ax.title, ax.xaxis.label, ax.yaxis.label]
            try:
                bb = ax.get_window_extent(renderer=r)
            except Exception:
                bb = None
            for t in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
                try:
                    tb = t.get_window_extent(renderer=r)
                except Exception:
                    continue
                if bb is None or (tb.x1 >= bb.x0 - 50 and tb.x0 <= bb.x1 + 50 and tb.y1 >= bb.y0 - 50 and tb.y0 <= bb.y1 + 50):
                    out.append(t)
        leg = ax.get_legend()
        if leg is not None:
            out.extend(leg.get_texts())
    keep = []
    for t in out:
        try:
            if t.get_visible() and isinstance(t.get_text(), str) and t.get_text().strip():
                keep.append(t)
        except Exception:
            pass
    return keep


def _boxes(fig, texts):
    r = fig.canvas.get_renderer()
    out = []
    for t in texts:
        try:
            bb = t.get_window_extent(renderer=r)
        except Exception:
            continue
        if bb.width <= 0 or bb.height <= 0:
            continue
        out.append((t, bb.x0 + INSET, bb.y0 + INSET, bb.x1 - INSET, bb.y1 - INSET))
    return out


def check_overlap(fig, name):
    """returns the number of collisions found; prints each one"""
    fig.canvas.draw()
    texts = _texts(fig)
    boxes = _boxes(fig, texts)
    hits = 0
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a = boxes[i]; b = boxes[j]
            if a[1] < b[3] and b[1] < a[3] and a[2] < b[4] and b[2] < a[4]:
                hits += 1
                print(f"  [{name}] text/text: '{a[0].get_text()[:30]}' vs '{b[0].get_text()[:30]}'")
    # text against ink
    full = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].astype(float).mean(axis=2)
    vis = [(t, t.get_visible()) for t in texts]
    for t, _ in vis:
        t.set_visible(False)
    fig.canvas.draw()
    bare = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].astype(float).mean(axis=2)
    for t, v in vis:
        t.set_visible(v)
    fig.canvas.draw()
    H = bare.shape[0]
    for (t, x0, y0, x1, y1) in boxes:
        if t.get_bbox_patch() is not None and t.get_bbox_patch().get_alpha() not in (0, 0.0) and t.get_bbox_patch().get_facecolor()[3] > 0.5:
            continue
        r0, r1 = int(H - y1), int(H - y0)
        c0, c1 = int(x0), int(x1)
        if r1 - r0 < 3 or c1 - c0 < 3:
            continue
        patch = bare[max(r0, 0):r1, max(c0, 0):c1]
        if patch.size == 0:
            continue
        gx = np.abs(np.diff(patch, axis=1)); gy = np.abs(np.diff(patch, axis=0))
        edges = (gx > EDGE_LEVEL).mean() + (gy > EDGE_LEVEL).mean()
        if edges > EDGE_FRACTION:
            hits += 1
            print(f"  [{name}] text/ink: '{t.get_text()[:40]}' edge density {edges:.3f}")
    return hits


def default_outdir():
    """Where the figures belong, resolved from the location of this file so
    that the scripts run from any working directory and in either the flat
    submission layout (figs beside code) or the repository layout (figs under
    paper)."""
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    flat = os.path.normpath(os.path.join(here, os.pardir, "figs"))
    nested = os.path.normpath(os.path.join(here, os.pardir, "paper", "figs"))
    if os.path.isdir(nested) and not os.path.isdir(flat):
        return nested
    return flat


def save(fig, name, outdir=None, vector=False, dpi=DPI):
    import os
    hits = check_overlap(fig, name)
    outdir = default_outdir() if outdir is None else outdir
    os.makedirs(outdir, exist_ok=True)
    if vector:
        fig.savefig(os.path.join(outdir, name + ".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(os.path.join(outdir, name + ".png"), dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    print(f"{name}: {hits} collisions")
    plt.close(fig)
    return hits


def front_mask(pts, ax, center=(0, 0, 0)):
    """NaN out the points of a curve on a convex body that face away from the camera"""
    pts = np.asarray(pts, dtype=float).copy()
    az = np.deg2rad(ax.azim); el = np.deg2rad(ax.elev)
    view = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    n = pts - np.asarray(center)
    back = (n @ view) < 0.02
    pts[back] = np.nan
    return pts
