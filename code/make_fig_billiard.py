"""Fig. 3: billiard picture of the hidden conserved quantity.

    python make_fig_billiard.py    ->   figs/fig_billiard.pdf      (a few seconds)

(a) A ball moves along great circles inside the spherical triangle T and bounces off the three capture walls q_j + q_k = 0.
(b) Squared components of its angular momentum L = q x v along the axes q1, q2, q3: each bounce exchanges two of them, so
L1^4 + L2^4 + L3^4 stays constant (the script prints it for every segment of the trajectory)."""
import numpy as np
from figstyle import *
from lamblions import CEN, VS, stereographic as proj

C1, C2, C3 = BLUE, ORANGE, GREEN
# wall q_j + q_k = 0: inward unit normal and the pair (j, k) of components of L which it exchanges
walls = [(np.array([0, 1, 1])/np.sqrt(2), (1, 2)), (np.array([1, 0, 1])/np.sqrt(2), (0, 2)), (np.array([1, 1, 0])/np.sqrt(2), (0, 1))]
def run(q, v, nb):
    """nb bounces of the ball starting at q with unit velocity v: list of (q, v, length, L, wall) for each segment"""
    segs = []
    for b in range(nb):
        best = None
        for w, (n, pair) in enumerate(walls):
            a, bb = n @ q, n @ v; s = np.arctan2(a, -bb)          # the great circle q cos s + v sin s reaches the wall at this s
            while s < 1e-7: s += np.pi
            if best is None or s < best[0]: best = (s, w)
        s, w = best
        segs.append((q.copy(), v.copy(), s, np.cross(q, v), w))
        qh = q*np.cos(s) + v*np.sin(s); vh = -q*np.sin(s) + v*np.cos(s); n = walls[w][0]
        v = vh - 2*(n @ vh)*n; q = qh                              # specular reflection
    return segs
# choose a generic, readable trajectory: well separated L_a^2, bounces away from the corners, the three walls visited
best = None; rng = np.random.default_rng(3)
for trial in range(60000):
    L = rng.normal(size=3); L /= np.linalg.norm(L); sq = np.sort(L**2)
    if min(sq[1] - sq[0], sq[2] - sq[1]) < 0.15 or sq[0] < 0.05: continue
    u = np.cross(L, rng.normal(size=3)); u /= np.linalg.norm(u); q0 = u
    if min(n @ q0 for n, p in walls) < 0.15: continue
    v0 = np.cross(L, q0); segs = run(q0, v0, 7)
    hits = [s[0]*np.cos(s[2]) + s[1]*np.sin(s[2]) for s in segs]
    dcorner = min(np.arccos(np.clip(h @ V, -1, 1)) for h in hits for V in VS)
    lens = [s[2] for s in segs]; ws = [s[4] for s in segs]
    if len(set(ws)) < 3: continue
    score = min(dcorner, 0.35) + 0.5*min(min(lens), 0.6)
    if best is None or score > best[0]: best = (score, q0, v0, dcorner, min(lens))
print("trajectory: distance to the nearest corner %.3f, shortest segment %.3f" % (best[3], best[4]))
score, q0, v0 = best[:3]; segs = run(q0, v0, 7)

fig = plt.figure(figsize=(3.375, 3.36)); gs = fig.add_gridspec(2, 1, height_ratios=[1.35, 0.82], hspace=0.14)
ax = fig.add_subplot(gs[0]); bx = fig.add_subplot(gs[1])
def arc(a, b, **kw):
    s = np.linspace(0, 1, 120); P = np.array([proj((1 - t)*a + t*b) for t in s]); ax.plot(P[:, 0], P[:, 1], **kw)
for i in range(3): arc(VS[i], VS[(i+1) % 3], color=INK, lw=1.4, solid_capstyle='round')
for k, (q, v, s, L, w) in enumerate(segs):
    t = np.linspace(0, s, 80); P = np.array([proj(q*np.cos(x) + v*np.sin(x)) for x in t])
    ax.plot(P[:, 0], P[:, 1], color=INK2, lw=0.9)
    m = len(t)//2; ax.annotate('', xy=P[m+2], xytext=P[m-2], arrowprops=dict(arrowstyle='-|>', color=INK2, lw=0.9, mutation_scale=7))
    hx, hy = proj(q*np.cos(s) + v*np.sin(s)); ax.plot(hx, hy, 'o', ms=3.6, color=INK, mec='white', mew=0.6, zorder=6)
    r = np.hypot(hx, hy); ax.text(hx*(1 + 0.13/r), hy*(1 + 0.13/r), str(k + 1), color=INK, fontsize=7.5, ha='center', va='center')
sx, sy = proj(q0); ax.plot(sx, sy, 's', ms=3.6, color='white', mec=INK, mew=0.9, zorder=6); ax.text(sx + 0.08, sy + 0.10, 'start', color=INK2, fontsize=7.5, ha='left', va='center')
for (n, pair), lab in zip(walls, [r'$L_2^2\leftrightarrow L_3^2$', r'$L_1^2\leftrightarrow L_3^2$', r'$L_1^2\leftrightarrow L_2^2$']):
    e = np.eye(3)[[i for i in range(3) if i not in pair][0]]      # middle of the wall: the axis which is not exchanged
    px, py = proj(e); r = np.hypot(px, py); ax.text(px*(1 + 0.42/r), py*(1 + 0.42/r), lab, color=INK2, fontsize=7.5, ha='center', va='center')
ax.set_aspect('equal'); ax.set_xlim(-1.75, 1.75); ax.set_ylim(-1.38, 1.38); ax.axis('off'); ax.text(-1.72, 1.25, '(a)', fontsize=9, color=INK)
# panel (b)
S = np.concatenate([[0], np.cumsum([s[2] for s in segs])]); Ls = np.array([s[3] for s in segs])**2
for a, col, lab in zip(range(3), (C1, C2, C3), (r'$L_1^2$', r'$L_2^2$', r'$L_3^2$')):
    xs = np.repeat(S, 2)[1:-1]; ys = np.repeat(Ls[:, a], 2)
    bx.plot(xs, ys, color=col, lw=1.5, label=lab, solid_joinstyle='miter')
O4 = (Ls**2).sum(1); bx.plot(np.repeat(S, 2)[1:-1], np.repeat(O4, 2), color=INK, lw=1.2, ls=(0, (4, 2)), label=r'$L_1^4+L_2^4+L_3^4$')
for k in range(len(segs) - 1):
    bx.axvline(S[k+1], color=MUTED, lw=0.4, ls=(0, (2, 2)), zorder=0); bx.text(S[k+1], 1.02, str(k + 1), color=INK, fontsize=7.5, ha='center', va='bottom', transform=bx.get_xaxis_transform())
bx.set_xlim(0, S[-1]); bx.set_ylim(0, 1.12); bx.set_yticks([0, 0.25, 0.5, 0.75]); bx.set_xlabel('distance travelled'); bx.set_ylabel('squared components')
bx.legend(frameon=False, fontsize=7.5, ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.02), handlelength=1.3, columnspacing=1.0, handletextpad=0.4, labelcolor=INK2)
for sp in ('top', 'right'): bx.spines[sp].set_visible(False)
bx.text(-0.2, 1.0, '(b)', fontsize=9, color=INK, transform=bx.transAxes, va='bottom')
print("L1^4 + L2^4 + L3^4 on the successive segments:", np.round(O4, 12))
save(fig, 'fig_billiard', bbox_inches='tight', pad_inches=0.02)
