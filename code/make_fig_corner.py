"""Fig. 4: zoom on a corner V of the spherical triangle T, with the corner coordinates (theta, phi).

    python make_fig_corner.py    ->   figs/fig_corner.pdf      (a few seconds)

Azimuthal equidistant projection centred on V: the distance to V in the plane is theta and the polar angle is phi (measured from
the wall on which the lamb meets lion 1), so the two walls are the straight lines phi = 0 and phi = 2 pi/3.
Colours: ground state m_1, series of Eq. (21) about this corner with the coefficients of Eqs. (23)-(25) (lamblions.py), with the
same colour scale as in panel (c) of Fig. 2.  Inset: the whole triangle in stereographic projection, with the zoomed region."""
import numpy as np
from matplotlib.patches import Arc, Polygon
from figstyle import *
from lamblions import CEN, VS as Vs, THETA_C as thc, great_arc as garc, stereographic as proj, corner_frame, corner_to_sphere, ground_state

g = ground_state()                              # m_1 with c_0 = 1: g.corner(theta, phi)
print("nu_1 = %.10f, kappa_1 = %.10f, m_1(c) = %.10f" % (g.nu, g.kappa, g.center))

# ------------------------------------------------------------------ frame of the corner
V, eb, ep = corner_frame(0)                     # V = (1,1,-1)/sqrt3; phi = 0 on the wall q2 + q3 = 0 (lamb meets lion 1), phi = 2 pi/3 on q1 + q3 = 0 (lion 2)
def sph(th, ph): return corner_to_sphere(th, ph, 0)
assert abs(sph(np.array(thc), np.array(np.pi/3)) - CEN).max() < 1e-12       # the centre c is at cos(theta) = 1/3, phi = pi/3
ROT = np.pi/6                                   # phi = 0 is drawn at 30 degrees, so that the mirror phi = pi/3 is vertical
def xy(th, ph): return th*np.cos(ph + ROT), th*np.sin(ph + ROT)

# ------------------------------------------------------------------ ground state on a polar grid
THMAX = 1.47
th = np.linspace(1e-4, THMAX, 220); ph = np.linspace(0, 2*np.pi/3, 241); TH, PH = np.meshgrid(th, ph)
U = np.clip(g.corner(TH, PH)/g.center, 0, 1)     # normalized by the value at the centre c, where m_1 is maximal
print("near the corner m_1/(m_1(c) theta^1.5 sin(3 phi/2)) -> %s at theta = 0.02, 0.05   (C/m_1(c) = %.4f, Eq. (19))"
      % ([round(float(g.corner(t, np.pi/3))/g.center/t**1.5, 4) for t in (0.02, 0.05)], np.sqrt(2/np.pi)/3/g.center))

FW, FH = 3.40, 2.52
fig = plt.figure(figsize=(FW, FH)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_aspect('equal'); ax.axis('off')
XL = 1.35; YB = -0.46; ax.set_xlim(-XL, XL); ax.set_ylim(YB, YB + 2*XL*FH/FW)
X, Y = xy(TH, PH)
cf = ax.contourf(X, Y, U, levels=np.linspace(0, 1, 11), cmap=cmap, zorder=1)
ax.contour(X, Y, U, levels=np.linspace(0.1, 0.9, 9), colors=SURF, linewidths=0.3, zorder=2)
# fade at the outer edge: the triangle continues beyond the zoom
t = np.linspace(0, 2*np.pi/3, 121)
for k, a in enumerate(np.linspace(0.0, 0.85, 12)):
    r0, r1 = THMAX - 0.20 + 0.20*k/12, THMAX - 0.20 + 0.20*(k + 1)/12 + 0.004
    xo, yo = xy(np.full_like(t, r1), t); xi, yi = xy(np.full_like(t, r0), t[::-1])
    ax.fill(np.r_[xo, xi], np.r_[yo, yi], color='white', alpha=a, lw=0, zorder=3)
# circles of constant theta and rays of constant phi (the coordinate grid)
for r in (0.4, 0.8, 1.2):
    xa, ya = xy(np.full_like(t, r), t); ax.plot(xa, ya, color=SURF, lw=0.5, ls=(0, (1, 2.2)), zorder=3)
# walls and mirror
RW = THMAX - 0.03
for p in (0.0, 2*np.pi/3):
    x1, y1 = xy(RW, p); ax.plot([0, x1], [0, y1], color=INK, lw=1.5, solid_capstyle='butt', zorder=5)
    x2, y2 = xy(THMAX + 0.10, p); ax.plot([x1, x2], [y1, y2], color=INK, lw=1.5, ls=(0, (0.6, 1.6)), zorder=5)
xm, ym = xy(THMAX - 0.02, np.pi/3); ax.plot([0, xm], [0, ym], color=INK2, lw=0.7, ls=(0, (4, 3)), zorder=4)
# the point (theta, phi)
thP, phP = 0.98, 0.50
xa, ya = xy(np.full_like(t, thP), t); ax.plot(xa, ya, color=INK, lw=0.5, ls=(0, (2, 2)), zorder=4)
xP, yP = xy(thP, phP); ax.plot([0, xP], [0, yP], color=INK, lw=0.9, zorder=6)
ax.plot(xP, yP, 'o', ms=4.0, color=INK, mec=SURF, mew=0.7, zorder=8)
ax.text(xP + 0.045, yP + 0.075, r'$(\theta,\phi)$', color=INK, fontsize=8.5, ha='left', va='center', zorder=8,
        bbox=dict(boxstyle='round,pad=0.10', fc='white', ec='none', alpha=0.80))
xmid, ymid = xy(0.60*thP, phP); nrm = np.array([-np.sin(phP + ROT), np.cos(phP + ROT)])
ax.text(xmid - 0.075*nrm[0], ymid - 0.075*nrm[1], r'$\theta$', color=INK, fontsize=9.5, ha='center', va='center', zorder=8)
ax.add_patch(Arc((0, 0), 0.50, 0.50, angle=0, theta1=np.degrees(ROT), theta2=np.degrees(ROT + phP), color=INK, lw=0.8, zorder=7))
xl, yl = xy(0.335, 0.50*phP); ax.text(xl, yl, r'$\phi$', color=INK, fontsize=9.5, ha='center', va='center', zorder=8)
# corner and centre
ax.plot(0, 0, 'o', ms=5.0, color=INK, mec=SURF, mew=0.8, zorder=9); ax.text(0, -0.12, 'V', color=INK, fontsize=9, ha='center', va='center')
xc, yc = xy(thc, np.pi/3); ax.plot(xc, yc, 'o', ms=4.0, color=SURF, mec=INK, mew=0.9, zorder=9)
ax.text(xc + 0.07, yc + 0.02, r'$c$', color=SURF, fontsize=9.5, ha='left', va='center', zorder=9)
ax.text(xc - 0.07, yc + 0.005, r'$\cos\theta=\frac{1}{3}$', color=SURF, fontsize=7.5, ha='right', va='center', zorder=9)
# labels of the walls and of the mirror
def along(r, p, off, s, **kw):
    x, y = xy(r, p); a = p + ROT; n = np.array([-np.sin(a), np.cos(a)]); rot = np.degrees(a); rot = rot - 180 if rot > 90 else rot
    ax.text(x + off*n[0], y + off*n[1], s, rotation=rot, rotation_mode='anchor', ha='center', va='center', zorder=8, **kw)
along(0.80, 0.0, -0.095, r'$\phi=0$   (lamb meets lion 1)', color=INK, fontsize=7.6)
along(0.80, 2*np.pi/3, 0.095, r'(lamb meets lion 2)   $\phi=2\pi/3$', color=INK, fontsize=7.6)
ax.text(-0.045, 0.56, r'$\phi=\pi/3$', color=INK, fontsize=7.6, rotation=90, ha='center', va='center', zorder=8)
# two-lion form near the corner
ax.text(-1.31, YB + 0.115, r'near V:  $m_1\simeq C\,\theta^{3/2}\sin\frac{3\phi}{2}$', color=INK, fontsize=8.2, ha='left', va='center')
# ------------------------------------------------------------------ inset: the whole triangle, with the zoomed region
ai = fig.add_axes([0.715, 0.025, 0.275, 0.275*FW/FH*0.94]); ai.set_aspect('equal'); ai.axis('off')
# zoomed region = points of T within THMAX of V: boundary made of the two walls and the arc theta = THMAX
tz = np.linspace(0, 2*np.pi/3, 80); arcq = sph(np.full_like(tz, THMAX - 0.02), tz); w1 = sph(np.linspace(0, THMAX - 0.02, 40), np.zeros(40)); w2 = sph(np.linspace(THMAX - 0.02, 0, 40), np.full(40, 2*np.pi/3))
poly = np.vstack([w1, arcq, w2]); px, py = proj(poly); ai.add_patch(Polygon(np.column_stack([px, py]), closed=True, fc='#9ec5f4', ec='none', zorder=1))
for i in range(3):
    A = garc(Vs[i], Vs[(i + 1) % 3], 120); px, py = proj(A); ai.plot(px, py, color=INK, lw=0.9, solid_capstyle='round', zorder=3)
px, py = proj(V); ai.plot(px, py, 'o', ms=3.4, color=INK, mec=SURF, mew=0.5, zorder=5); ai.text(px + 0.26, py, 'V', color=INK, fontsize=7.5, ha='center', va='center')
ai.plot(0, 0, 'o', ms=2.8, color=SURF, mec=INK, mew=0.7, zorder=5); ai.text(-0.22, 0.02, r'$c$', color=INK, fontsize=7.5, ha='center', va='center')
ai.text(-0.52, -0.62, r'$T$', color=INK, fontsize=8, ha='center', va='center')
ai.set_xlim(-1.30, 1.75); ai.set_ylim(-1.38, 1.38)
save(fig, 'fig_corner')
