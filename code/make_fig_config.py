"""Fig. 2: configuration space of the lamb and the lions in the center-of-mass frame.

    python make_fig_config.py    ->   figs/fig_config.pdf      (about a minute, most of it to select the sample trajectory of panel (a))

(a) two lions: the wedge of opening angle 2 pi/3 with a sample Brownian trajectory, and the arc coloured by sin(3 phi/2);
(b) three lions: the cone cut by the unit ball, seen in three dimensions (flat walls meeting at the apex O, curved lid T);
(c) the lid T flattened by stereographic projection, with its ground state m_1 (series of Eq. (21), from lamblions.py);
bottom: positions of the animals on the line at the marked points."""
import numpy as np
import matplotlib.tri as mtri
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb
from figstyle import *
from lamblions import CEN as c, VS as Vs, MS as Ms, EX as ex, EY as ey, great_arc as garc, inside_T, ground_state

def unit3(v): return np.asarray(v, float)/np.linalg.norm(v)

def ground_state_grid(n=301):
    """ground state m_1 on a grid of the stereographic plane centred on c: X, Y, the points Q of the sphere, the values U
    normalized by the maximum (nan outside T) and the mask of the points inside T"""
    g = np.linspace(-1.55, 1.55, n); X, Y = np.meshgrid(g, g)
    r2 = (X*X + Y*Y)/4; f = 2/(1 + r2); Q = ((f - 1)[..., None]*c + (f*X/2)[..., None]*ex + (f*Y/2)[..., None]*ey).reshape(-1, 3)
    inside = inside_T(Q, 1e-9); u = ground_state().sphere(Q[inside]); u = u/u[np.argmax(np.abs(u))]
    U = np.full(n*n, np.nan); U[inside] = u
    return X, Y, Q, U.reshape(n, n), inside.reshape(n, n)

def draw_solid(ax, gs, delta=80.0, az=180.0, Rext=1.38, light=(-0.5, 0.35, 0.8)):
    """the solid 'cone cut by the unit ball' seen from outside, in orthographic projection along the direction d which makes
    the angle delta with c: flat walls meeting at the apex, curved lid T coloured by the ground state"""
    X, Y, Q, U, inside = gs
    th = np.radians(delta); a = np.radians(az); tilt = np.cos(a)*ex + np.sin(a)*ey
    d = np.cos(th)*c + np.sin(th)*tilt; up = c - (c @ d)*d; e2 = up/np.linalg.norm(up); e1 = np.cross(e2, d)
    P = lambda q: (np.asarray(q) @ e1, np.asarray(q) @ e2)
    Lv = light[0]*e1 + light[1]*e2 + light[2]*d; Lv /= np.linalg.norm(Lv)
    t = np.linspace(0, 2*np.pi, 721)
    ax.fill(np.cos(t), np.sin(t), color='#f4f3f0', lw=0, zorder=0); ax.plot(np.cos(t), np.sin(t), color=MUTED, lw=0.6, ls=(0, (1.2, 2.0)), zorder=1)
    normals = []
    for i in range(3):
        n = np.cross(Vs[i], Vs[(i + 1) % 3]); n /= np.linalg.norm(n)
        if n @ c > 0: n = -n
        normals.append(n)                           # outward normal of the wall between the corners i and i + 1
    wallvis = [n @ d > 1e-9 for n in normals]
    # hidden edges first (dashed)
    for i in range(3):
        A = garc(Vs[i], Vs[(i + 1) % 3], 200); hid = ~((A @ d > 0) | wallvis[i]); x, y = P(A)
        ax.plot(np.where(hid, x, np.nan), np.where(hid, y, np.nan), color=MUTED, lw=0.6, ls=(0, (3, 2.2)), zorder=2)
        if not (wallvis[i] or wallvis[(i - 1) % 3]):
            x1, y1 = P(Vs[i]); ax.plot([0, x1], [0, y1], color=MUTED, lw=0.6, ls=(0, (3, 2.2)), zorder=2)
    # visible flat walls, shaded by their orientation
    lo, hi = np.array(to_rgb('#a9b6ca')), np.array(to_rgb('#e6ebf2')); lams = [normals[i] @ Lv for i in range(3)]
    vis_l = [lams[i] for i in range(3) if wallvis[i]]; l0, l1 = (min(vis_l), max(vis_l)) if vis_l else (0, 1)
    for i in range(3):
        if not wallvis[i]: continue
        A = garc(Vs[i], Vs[(i + 1) % 3], 120); poly = np.vstack([np.zeros(3), A]); x, y = P(poly)
        w = 0.5 if l1 - l0 < 1e-6 else (lams[i] - l0)/(l1 - l0)
        ax.fill(x, y, color=lo + (hi - lo)*w, lw=0, zorder=3)
    # curved lid T (visible part) with the ground state
    tri = mtri.Triangulation(X.ravel(), Y.ravel()); bad = (~inside.ravel()) | (Q @ d <= 0.0); tri.set_mask(bad[tri.triangles].any(axis=1))
    px, py = P(Q); tri3 = mtri.Triangulation(px, py, tri.triangles, mask=tri.mask)
    cf = ax.tricontourf(tri3, np.nan_to_num(U.ravel()), levels=np.linspace(0, 1, 11), cmap=cmap, zorder=4)
    limb = np.outer(np.cos(t), e1) + np.outer(np.sin(t), e2); ok = inside_T(limb, -1e-12)
    ax.plot(np.where(ok, np.cos(t), np.nan), np.where(ok, np.sin(t), np.nan), color=INK, lw=1.0, zorder=6)
    # visible edges
    for i in range(3):
        A = garc(Vs[i], Vs[(i + 1) % 3], 200); vis = (A @ d > 0) | wallvis[i]; x, y = P(A)
        ax.plot(np.where(vis, x, np.nan), np.where(vis, y, np.nan), color=INK, lw=1.4, solid_capstyle='round', zorder=7)
        if wallvis[i] or wallvis[(i - 1) % 3]:
            x1, y1 = P(Vs[i]); ax.plot([0, x1], [0, y1], color=INK, lw=1.0, solid_capstyle='round', zorder=7)
    cornervis = [(Vs[i] @ d > 0) or wallvis[i] or wallvis[(i - 1) % 3] for i in range(3)]
    for i, v in enumerate(Vs):
        x1, y1 = P(v)
        if cornervis[i]:
            x2, y2 = P(Rext*v); ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle='-|>,head_width=0.12,head_length=0.28', lw=0.7, color=INK2, shrinkA=0, shrinkB=0), zorder=8)
            ax.plot(x1, y1, 'o', ms=4.5, color=INK, mec=SURF, mew=0.8, zorder=9)
    for mm in Ms:                                   # middles of the walls (axes q_a)
        if (mm @ d > 0) or any(wallvis[k] for k in range(3) if abs(normals[k] @ mm) < 1e-9):
            x1, y1 = P(mm); ax.plot(x1, y1, 's', ms=3.6, color=SURF, mec=INK, mew=0.8, zorder=9)
    if c @ d > 0.02:
        x1, y1 = P(c); ax.plot(x1, y1, 'o', ms=3.4, color=SURF, mec=INK, mew=0.8, zorder=9)
    return dict(P=P, d=d, cf=cf, wallvis=wallvis, cornervis=cornervis, normals=normals)

FW, FH = 7.0, 2.30
fig = plt.figure(figsize=(FW, FH))
def axes_in(x, y, w, h): return fig.add_axes([x/FW, y/FH, w/FW, h/FH])
# ------------------------------------------------------------------ (a) two lions: the wedge
WA, HA = 2.30, 1.26; XM = WA/2; Y0, Y1 = -0.07, HA - 0.07
aw = axes_in(0.02, FH - HA - 0.02, WA, HA); aw.set_xlim(-XM, XM); aw.set_ylim(Y0, Y1); aw.set_aspect('equal'); aw.axis('off')
s3 = np.sqrt(3.0); yw = XM/s3
aw.fill([0, XM, XM, -XM, -XM], [0, yw, Y1, Y1, yw], color='#eef4fd', lw=0, zorder=0)
aw.plot([0, XM], [0, yw], color=INK, lw=1.4, solid_capstyle='butt', zorder=3); aw.plot([0, -XM], [0, yw], color=INK, lw=1.4, solid_capstyle='butt', zorder=3)
aw.plot([0, 0], [0, Y1], color=INK2, lw=0.7, ls=(0, (4, 3)), zorder=2)
Ra = 0.86; ang = np.radians(np.linspace(30, 150, 241)); Pa = np.column_stack([Ra*np.cos(ang), Ra*np.sin(ang)])
seg = np.stack([Pa[:-1], Pa[1:]], axis=1); val = np.sin(1.5*(0.5*(ang[:-1] + ang[1:]) - np.pi/6))         # two-lion ground state sin(3 phi/2)
for dr in (-0.033, 0.033): aw.plot((Ra + dr)*np.cos(ang), (Ra + dr)*np.sin(ang), color=MUTED, lw=0.4, zorder=3)
lc = LineCollection(seg, cmap=cmap, array=val, linewidths=4.2, zorder=3, capstyle='butt'); lc.set_clim(0, 1); aw.add_collection(lc)
R0 = 0.30
def make_path(seed, r0=R0, h=0.0085):
    """Brownian path in the wedge started on its axis at distance r0 from the apex, up to its first passage through a wall"""
    rng = np.random.default_rng(seed); p = np.array([0.0, r0]); pts = [p.copy()]
    for _ in range(40000):
        q = p + h*rng.standard_normal(2)
        if q[1] <= abs(q[0])/s3:
            lo, hi = 0.0, 1.0
            for _ in range(30):                     # locate the crossing point on the last step
                mid = 0.5*(lo + hi); m = p + mid*(q - p)
                if m[1] <= abs(m[0])/s3: hi = mid
                else: lo = mid
            pts.append(p + lo*(q - p)); return np.array(pts)
        p = q; pts.append(p.copy())
    return None
best = None
for seed in range(1, 2500):                         # keep a readable sample: it stays inside the arc and ends on the right-hand wall
    pth = make_path(seed)
    if pth is None: continue
    r = np.hypot(pth[:, 0], pth[:, 1]); end = pth[-1]; n = len(pth)
    if not (2500 < n < 9000): continue
    if r.max() > 0.76 or pth[:, 1].max() > 0.76: continue
    if not (0.27 < np.hypot(*end) < 0.40) or end[0] < 0: continue
    if np.abs(pth[:, 0]).max() < 0.34: continue
    if (np.hypot(pth[:, 0] + 0.25, pth[:, 1] - R0 - 0.05) < 0.12).any(): continue
    if (pth[:, 0] < -0.02).mean() > 0.25: continue                         # keep the 'lions meet' label readable
    score = np.abs(pth[:, 0]).max() + pth[:, 1].max() - 0.0001*n
    if best is None or score > best[0]: best = (score, seed, pth)
print("trajectory: seed %d, %d steps, ends at (%.3f, %.3f)" % (best[1], len(best[2]), best[2][-1, 0], best[2][-1, 1]))
pth = best[2]; aw.plot(pth[:, 0], pth[:, 1], color=INK2, lw=0.45, zorder=4, solid_joinstyle='round')
aw.plot(0, R0, 'o', ms=4.0, color=SURF, mec=INK, mew=0.9, zorder=6); aw.text(-0.055, R0, 'start', color=INK, fontsize=7.5, ha='right', va='center', zorder=6)
aw.plot(pth[-1, 0], pth[-1, 1], 'o', ms=4.2, color=ORANGE, mec=INK, mew=0.6, zorder=6)
aw.annotate('capture', xy=(pth[-1, 0], pth[-1, 1]), xytext=(pth[-1, 0] + 0.17, pth[-1, 1] - 0.17), color=INK, fontsize=7.5, ha='center', va='center',
            arrowprops=dict(arrowstyle='-', lw=0.5, color=INK2, shrinkA=1, shrinkB=3), zorder=6)
aw.plot(0, 0, 'v', ms=5.2, color=INK, mec=SURF, mew=0.8, zorder=6)
a2 = np.radians(np.linspace(30, 150, 60)); aw.plot(0.14*np.cos(a2), 0.14*np.sin(a2), color=INK2, lw=0.6, zorder=3)
aw.text(-0.185, 0.185, '2π/3', color=INK2, fontsize=7, ha='center', va='center', zorder=5)
def along(xc, sign, txt, off):
    yc = abs(xc)/s3; nx, ny = (0.5*sign, -s3/2)
    aw.text(xc + off*nx, yc + off*ny, txt, color=INK, fontsize=7.2, ha='center', va='center', rotation=30*sign, rotation_mode='anchor', zorder=5)
along(0.76, +1, 'lamb meets lion 1', 0.095); along(-0.76, -1, 'lamb meets lion 2', 0.095)
aw.text(-0.05, 0.575, 'lions meet', color=INK2, fontsize=7, ha='center', va='center', rotation=90, zorder=5)
aw.text(-XM + 0.03, Y1 - 0.02, '(a)', color=INK, fontsize=9, ha='left', va='top')
ox, oy, al = 0.66, 0.90, 0.19
for dx, dy in ((al, 0), (0, al)):
    aw.annotate('', xy=(ox + dx, oy + dy), xytext=(ox, oy), arrowprops=dict(arrowstyle='-|>,head_width=0.11,head_length=0.22', lw=0.6, color=INK2, shrinkA=0, shrinkB=0))
aw.text(ox + al + 0.03, oy, '$x_1-x_2$', color=INK2, fontsize=6.5, ha='left', va='center'); aw.text(ox - 0.04, oy + al - 0.01, '$2x_0-x_1-x_2$', color=INK2, fontsize=6.5, ha='right', va='center')
# ------------------------------------------------------------------ (b) three lions: the cone cut by the unit ball, in 3D
gs = ground_state_grid(301)
WB, HB = 2.25, 1.55
ab = axes_in(2.40, FH - HB - 0.02, WB, HB); ab.set_aspect('equal'); ab.set_xlim(-1.654, 1.654); ab.set_ylim(-1.07, 1.21); ab.axis('off')
r3 = draw_solid(ab, gs, delta=86.0, az=-9.0, Rext=1.36); P3 = r3['P']
ab.text(-1.654 + 0.04, 1.21 - 0.03, '(b)', color=INK, fontsize=9, ha='left', va='top')
ab.plot(0, 0, 'v', ms=5.6, color=INK, mec=SURF, mew=0.8, zorder=10); ab.text(0.0, -0.13, 'O', color=INK, fontsize=8.5, ha='center', va='center', zorder=10)
xc, yc = P3(c); ab.text(xc + 0.11, yc + 0.07, '$c$', color=INK, fontsize=8.5, ha='center', va='center')
xT, yT = P3(unit3(0.55*c + 0.45*r3['d'])); ab.text(xT, yT, '$T$', color=SURF, fontsize=10, ha='center', va='center', zorder=10)
for i in range(3):
    if not r3['wallvis'][i]: continue
    lion = int(np.argmin(np.abs(r3['normals'][i]))) + 1; xm, ym = P3(0.60*unit3(Vs[i] + Vs[(i + 1) % 3])); sgn = 1 if xm > 0 else -1
    ab.annotate('lamb meets lion %d' % lion, xy=(xm, ym), xytext=(sgn*0.86, -0.42), color=INK, fontsize=7.2, ha='center', va='center',
                arrowprops=dict(arrowstyle='-', lw=0.5, color=INK2, shrinkA=2, shrinkB=1), zorder=10)
for i, v in enumerate(Vs):
    if r3['cornervis'][i]:
        xv, yv = P3(v); dx, dy = (0.13, -0.10) if abs(xv) < 0.4 else (np.sign(xv)*0.13, -0.12); ab.text(xv + dx, yv + dy, 'V', color=INK, fontsize=8.5, ha='center', va='center', zorder=10)
ab.text(0.0, -0.80, 'unit sphere', color=MUTED, fontsize=7.2, ha='center', va='center')
# ------------------------------------------------------------------ (c) the lid T flattened
X, Y, Q, U, inside = gs
WC = 1.95; HC = WC*(2.84/3.05)
ax = axes_in(4.60, 0.24, WC, HC)
cf = ax.contourf(X, Y, U, levels=np.linspace(0, 1, 11), cmap=cmap); cf.set_edgecolor('face')
ax.contour(X, Y, U, levels=np.linspace(0.1, 0.9, 9), colors=SURF, linewidths=0.3)
def proj(q):
    q = np.asarray(q, float); q = q/np.linalg.norm(q); dd = 1 + q @ c; return 2*(q @ ex)/dd, 2*(q @ ey)/dd
def arcp(a, b, **kw):
    A = garc(np.array(a, float), np.array(b, float), 200); Pp = np.array([proj(q) for q in A]); ax.plot(Pp[:, 0], Pp[:, 1], **kw)
for i in range(3): arcp(Vs[i], Vs[(i + 1) % 3], color=INK, lw=1.4, solid_capstyle='round')                # capture walls
for v, m in zip(Vs, [(0, 0, 1), (1, 0, 0), (0, 1, 0)]): arcp(v, m, color=INK2, lw=0.7, ls=(0, (4, 3)))    # mirrors: two lions meet
for v, off in zip(Vs, [(0.14, 0.0), (-0.09, 0.13), (-0.09, -0.13)]):
    px, py = proj(v); ax.plot(px, py, 'o', ms=4.5, color=INK, mec=SURF, mew=0.8, zorder=5); ax.text(px + off[0], py + off[1], 'V', color=INK, fontsize=8.5, ha='center', va='center')
for m, lab in zip(Ms, ['$q_1$', '$q_2$', '$q_3$']):
    px, py = proj(m); ax.plot(px, py, 's', ms=4, color=SURF, mec=INK, mew=0.9, zorder=5)
    r = np.hypot(px, py); ax.text(px*(1 + 0.22/r), py*(1 + 0.22/r), lab, color=INK, fontsize=8.5, ha='center', va='center')
ax.plot(0, 0, 'o', ms=3.8, color=SURF, mec=INK, mew=0.9, zorder=5); ax.text(0.12, 0.12, '$c$', color=SURF, fontsize=8.5, ha='center', va='center')
px, py = proj(Vs[0]); ax.text(px - 0.40, py, '2π/3', color=INK2, fontsize=7, ha='center', va='center', bbox=dict(boxstyle='round,pad=0.12', fc='#eef4fd', ec='none'))
ax.set_aspect('equal'); ax.set_xlim(-1.45, 1.6); ax.set_ylim(-1.42, 1.42); ax.axis('off')
fig.text(4.62/FW, (FH - 0.04)/FH, '(c)', color=INK, fontsize=9, ha='left', va='top')
pos = ax.get_position(); cax = fig.add_axes([6.56/FW, pos.y0 + 0.10*pos.height, 0.075/FW, 0.80*pos.height])
cb = fig.colorbar(cf, cax=cax, ticks=[0, 0.5, 1]); cb.outline.set_edgecolor(MUTED); cb.set_label('ground state (normalized)', color=INK2, fontsize=7.5); cb.ax.tick_params(labelsize=7, width=0.5); cb.outline.set_linewidth(0.5)
# ------------------------------------------------------------------ pictograms for the marked points
HS = 0.66; sx = axes_in(0.0, 0.0, 4.66, HS); sx.set_xlim(0, 4.66); sx.set_ylim(0, HS); sx.axis('off')
def picto(x, y, groups, wid=0.50, dy=0.085):
    """positions on the line: groups = [(position between 0 and 1, number of lions there, lamb there?), ...]"""
    sx.annotate('', xy=(x + wid + 0.09, y), xytext=(x - 0.04, y), arrowprops=dict(arrowstyle='-|>,head_width=0.12,head_length=0.25', lw=0.6, color=MUTED, shrinkA=0, shrinkB=0))
    for p, nl, lamb in groups:
        k = nl + (1 if lamb else 0); ys = y + dy*(np.arange(k) - (k - 1)/2)
        for i in range(nl): sx.plot(x + p*wid, ys[i], 'o', ms=4.3, color=BLUE, mec=SURF, mew=0.5, zorder=4, clip_on=False)
        if lamb: sx.plot(x + p*wid, ys[-1], 'D', ms=4.0, color=ORANGE, mec=SURF, mew=0.5, zorder=5, clip_on=False)
yb = 0.22; col = [0.55, 1.65, 2.75, 3.95]
sx.plot(col[0] - 0.30, yb, 'v', ms=5.6, color=INK, mec=SURF, mew=0.8); sx.text(col[0] - 0.17, yb, 'O', color=INK, fontsize=8.5, ha='center', va='center')
picto(col[0], yb, [(0.5, 3, True)])
sx.plot(col[1] - 0.30, yb, 'o', ms=3.8, color=SURF, mec=INK, mew=0.9); sx.text(col[1] - 0.19, yb, '$c$', color=INK, fontsize=9, ha='center', va='center')
picto(col[1], yb, [(0.12, 3, False), (0.88, 0, True)])
sx.plot(col[2] - 0.30, yb, 'o', ms=4.5, color=INK, mec=SURF, mew=0.8); sx.text(col[2] - 0.18, yb, 'V', color=INK, fontsize=8.5, ha='center', va='center')
picto(col[2], yb, [(0.12, 1, False), (0.88, 2, True)])
sx.plot(col[3] - 0.36, yb, 's', ms=4, color=SURF, mec=INK, mew=0.9); sx.text(col[3] - 0.21, yb, '$q_a$', color=INK, fontsize=9, ha='center', va='center')
picto(col[3], yb, [(0.12, 2, False), (0.88, 1, True)])
yl = 0.56
sx.text(0.22, yl, 'Positions of the animals on the line at the marked points of (b) and (c)', color=INK2, fontsize=7.5, ha='left', va='center')
sx.plot(3.72, yl, 'D', ms=4.0, color=ORANGE, mec=SURF, mew=0.5); sx.text(3.79, yl, 'lamb', color=INK2, fontsize=7.5, ha='left', va='center')
sx.plot(4.12, yl, 'o', ms=4.3, color=BLUE, mec=SURF, mew=0.5); sx.text(4.19, yl, 'lion', color=INK2, fontsize=7.5, ha='left', va='center')
save(fig, 'fig_config')
