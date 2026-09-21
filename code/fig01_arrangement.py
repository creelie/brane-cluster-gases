"""
fig01_arrangement.py -- a brane arrangement and its strata.

(a) Three transverse hypersurfaces (planes) in a cube of three-space.  Pairs meet
    in lines, the triple meets in a point: the strata of codimension one, two and
    three, each carrying a homology class of the ambient space.
(b) A Poisson gas of planes in the same cube: a random number of branes with
    random positions and orientations.  The intersection strata of the gas are
    again lines and points, and their expected numbers are the factorial moments
    of the Poisson process.
"""
import numpy as np
import matplotlib.pyplot as plt
from figstyle import *

rng = np.random.default_rng(5)

def plane_patch(n, d, size=1.0, res=2):
    """quad of the plane n.x = d clipped to the cube [-1,1]^3, as a polygon via corner sampling"""
    n = np.asarray(n, float); n /= np.linalg.norm(n)
    # build an orthonormal frame in the plane
    a = np.cross(n, [0, 0, 1.0])
    if np.linalg.norm(a) < 1e-6:
        a = np.cross(n, [0, 1.0, 0])
    a /= np.linalg.norm(a); b = np.cross(n, a)
    p0 = d * n
    # polygon = intersection of plane with cube: sample a big square then clip by convex hull of points inside cube edges
    pts = []
    # intersect plane with the 12 cube edges
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
    edges = [(i, j) for i in range(8) for j in range(i + 1, 8) if np.sum(corners[i] != corners[j]) == 1]
    for i, j in edges:
        P, Q = corners[i], corners[j]
        fp, fq = n @ P - d, n @ Q - d
        if fp * fq < 0:
            t = fp / (fp - fq)
            pts.append(P + t * (Q - P))
    if len(pts) < 3:
        return None
    pts = np.array(pts)
    # order by angle in the plane
    c = pts.mean(axis=0)
    ang = np.arctan2((pts - c) @ b, (pts - c) @ a)
    return pts[np.argsort(ang)]

def draw_planes(ax, planes, cols, alpha=0.45):
    for (n, d), col in zip(planes, cols):
        poly = plane_patch(n, d)
        if poly is None:
            continue
        nn = np.asarray(n, float); nn /= np.linalg.norm(nn)
        ls = np.array([0.4, -0.5, 0.75]); ls /= np.linalg.norm(ls)
        shade_f = 0.65 + 0.35 * abs(nn @ ls)
        c = np.clip(np.array(to_rgb(col)) * shade_f, 0, 1)
        coll = Poly3DCollection([poly], facecolors=[np.append(c, alpha)], edgecolors=[np.append(c, 0.9)], linewidths=0.5)
        ax.add_collection3d(coll)

def line_in_cube(p, v, ax, col, lw=1.6):
    v = np.asarray(v, float); v /= np.linalg.norm(v)
    ts = []
    for k in range(3):
        if abs(v[k]) > 1e-9:
            ts += [(-1 - p[k]) / v[k], (1 - p[k]) / v[k]]
    ts = sorted(t for t in ts if np.all(np.abs(p + t * v) <= 1 + 1e-9))
    if len(ts) >= 2:
        A = p + ts[0] * v; B = p + ts[-1] * v
        ax.plot([A[0], B[0]], [A[1], B[1]], [A[2], B[2]], color=col, lw=lw, zorder=6)

def cube(ax):
    for s in ([-1, -1], [-1, 1], [1, -1], [1, 1]):
        ax.plot([-1, 1], [s[0]] * 2, [s[1]] * 2, color=GRID, lw=0.6)
        ax.plot([s[0]] * 2, [-1, 1], [s[1]] * 2, color=GRID, lw=0.6)
        ax.plot([s[0]] * 2, [s[1]] * 2, [-1, 1], color=GRID, lw=0.6)

fig = plt.figure(figsize=(7.2, 3.5))
ax = fig.add_subplot(1, 2, 1, projection="3d")
blank_3d(ax); ax.computed_zorder = False
ax.view_init(elev=22, azim=-52)
cube(ax)
planes = [((1.0, 0.25, 0.15), 0.1), ((0.2, 1.0, 0.1), -0.05), ((0.1, -0.3, 1.0), 0.15)]
draw_planes(ax, planes, [BLUE, ORANGE, AQUA])
# pairwise intersection lines and the triple point
N = np.array([np.asarray(p[0], float) / np.linalg.norm(p[0]) for p in planes])
D = np.array([p[1] for p in planes])
x0 = np.linalg.solve(N, D)
for (i, j) in ((0, 1), (0, 2), (1, 2)):
    v = np.cross(N[i], N[j])
    line_in_cube(x0, v, ax, INK, lw=1.3)
ax.scatter([x0[0]], [x0[1]], [x0[2]], color=INK, s=34, zorder=9, depthshade=False)
ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2); ax.set_zlim(-1.25, 1.35)
ax.text2D(0.02, 0.97, "(a)", transform=ax.transAxes, fontsize=10, fontweight="bold")
ax.text2D(0.5, 0.0, r"three branes: $K=1$ walls, $K=2$ lines, $K=3$ point", transform=ax.transAxes, ha="center", fontsize=8, color=INK2)
ax.text2D(0.02, 0.90, r"strata by type: $\binom{3}{1},\binom{3}{2},\binom{3}{3}=3,3,1$", transform=ax.transAxes, fontsize=8, color=INK)

ax2 = fig.add_subplot(1, 2, 2, projection="3d")
blank_3d(ax2); ax2.computed_zorder = False
ax2.view_init(elev=22, azim=-52)
cube(ax2)
Np = rng.poisson(6)
while Np < 5 or Np > 8:
    Np = rng.poisson(6)
gas = []
for _ in range(Np):
    n = rng.normal(size=3); n /= np.linalg.norm(n)
    d = rng.uniform(-0.45, 0.45)
    gas.append((n, d))
draw_planes(ax2, gas, [BLUE] * len(gas), alpha=0.22)
# draw pairwise intersection lines inside the cube (only a few, thin) and triple points
Ng = np.array([g[0] for g in gas]); Dg = np.array([g[1] for g in gas])
cnt = 0
for i in range(len(gas)):
    for j in range(i + 1, len(gas)):
        v = np.cross(Ng[i], Ng[j])
        # a point on the line: solve with a third equation v.x = 0
        try:
            p = np.linalg.solve(np.vstack([Ng[i], Ng[j], v]), [Dg[i], Dg[j], 0.0])
        except np.linalg.LinAlgError:
            continue
        line_in_cube(p, v, ax2, INK2, lw=0.6)
tri = 0
for i in range(len(gas)):
    for j in range(i + 1, len(gas)):
        for k in range(j + 1, len(gas)):
            try:
                p = np.linalg.solve(np.vstack([Ng[i], Ng[j], Ng[k]]), [Dg[i], Dg[j], Dg[k]])
            except np.linalg.LinAlgError:
                continue
            if np.all(np.abs(p) <= 1):
                ax2.scatter([p[0]], [p[1]], [p[2]], color=ORANGE, s=14, zorder=9, depthshade=False)
                tri += 1
ax2.set_xlim(-1.2, 1.2); ax2.set_ylim(-1.2, 1.2); ax2.set_zlim(-1.25, 1.35)
ax2.text2D(0.02, 0.97, "(b)", transform=ax2.transAxes, fontsize=10, fontweight="bold")
ax2.text2D(0.5, 0.0, f"a Poisson gas: {Np} branes, {tri} triple points in the box", transform=ax2.transAxes, ha="center", fontsize=8, color=INK2)
ax2.text2D(0.02, 0.90, r"expected $\#\{K$-fold strata$\}=\Lambda^{K}/K!$", transform=ax2.transAxes, fontsize=8, color=INK)
fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.05, wspace=0.02)
save(fig, "fig01_arrangement")
