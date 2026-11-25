"""Capture of a diffusing lamb by three diffusing lions: numerical companion to the Letter (double precision).

This module implements the exact solution as it is written in the Letter and is used by the figure scripts:

    kappa(nu), c_k(nu)        continued fraction of Eq. (24) solved by iteration, coefficients of Eq. (25)
    solve_nu1()               root search on the condition at the center, Eq. (26)   ->   nu_1 = 2 beta_3
    m1_corner, m1_sphere      ground state, series of Eq. (21) about the nearest corner of the triangle T
    symmetric_modes()         excited states which are symmetric in the lions, with their amplitudes A_j of Eq. (15)
    survival()                survival probability S_3(t) at all times, Eq. (15)

    python lamblions.py       prints the numbers quoted in the Letter and in its Supplemental Material

Requirements: numpy, scipy.  Equation numbers refer to the Letter, (S..) to the Supplemental Material."""
import numpy as np
from scipy.special import hyp2f1, hyp1f1, gamma
from scipy.optimize import brentq

SQRT2 = np.sqrt(2.0)
THETA_C = np.arccos(1.0/3.0)                 # angular distance between a corner V and the center c of T
NU1 = 1.8257570041744562                     # nu_1 = 2 beta_3 (solve_nu1() recomputes it)

# ----------------------------------------------------------------------------------------------------------------------
# Geometry of the spherical triangle T in the coordinates (q1, q2, q3) of Eq. (17)
# ----------------------------------------------------------------------------------------------------------------------
CEN = np.array([1.0, 1.0, 1.0])/np.sqrt(3)                                              # center c
VS = [np.array(v, float)/np.sqrt(3) for v in ((1, 1, -1), (-1, 1, 1), (1, -1, 1))]      # corners V
MS = [np.array(m, float) for m in ((1, 0, 0), (0, 1, 0), (0, 0, 1))]                    # middles of the walls (axes q_a)
EX = np.array([1.0, 1.0, -2.0])/np.sqrt(6); EY = np.array([-1.0, 1.0, 0.0])/np.sqrt(2)  # axes of the stereographic plane

def inside_T(q, eps=0.0):
    """True for the points q (n x 3) of the cone q_j + q_k > 0, i.e. x_0 > x_i for the three lions (Eq. (S1))"""
    q = np.atleast_2d(q); return (q[:, 1] + q[:, 2] > eps) & (q[:, 0] + q[:, 2] > eps) & (q[:, 0] + q[:, 1] > eps)

def great_arc(a, b, n=80):
    """n points on the arc of great circle from a to b"""
    a = a/np.linalg.norm(a); b = b/np.linalg.norm(b); s = np.linspace(0, 1, n)[:, None]; P = (1 - s)*a + s*b
    return P/np.linalg.norm(P, axis=1)[:, None]

def stereographic(q):
    """stereographic projection centred on c of the points q (.. x 3): the plane coordinates (x, y)"""
    q = np.asarray(q, float); q = q/np.linalg.norm(q, axis=-1, keepdims=True); d = 1 + q @ CEN
    return 2*(q @ EX)/d, 2*(q @ EY)/d

def corner_frame(i):
    """(V, e_b, e_p): corner V_i, unit tangent at V_i along one of its two walls (phi = 0) and the orthogonal tangent, such
    that the other wall is phi = 2 pi/3.  For i = 0 the wall phi = 0 is the one on which the lamb meets lion 1."""
    V = VS[i]; W0, W1 = VS[(i + 2) % 3], VS[(i + 1) % 3]
    tang = lambda P: (P - (P @ V)*V)/np.linalg.norm(P - (P @ V)*V)
    eb, ea = tang(W0), tang(W1); ep = ea - (ea @ eb)*eb; ep /= np.linalg.norm(ep)
    return V, eb, ep

def corner_to_sphere(th, ph, i=0):
    """unit vector of corner coordinates (theta, phi) about the corner V_i"""
    V, eb, ep = corner_frame(i); th, ph = np.asarray(th, float), np.asarray(ph, float)
    return np.cos(th)[..., None]*V + np.sin(th)[..., None]*(np.cos(ph)[..., None]*eb + np.sin(ph)[..., None]*ep)

def sphere_to_corner(q, i=0):
    """corner coordinates (theta, phi) about the corner V_i of the unit vectors q (.. x 3)"""
    V, eb, ep = corner_frame(i); q = np.asarray(q, float)
    return np.arccos(np.clip(q @ V, -1, 1)), np.arctan2(q @ ep, q @ eb)

# ----------------------------------------------------------------------------------------------------------------------
# The tri-diagonal matrix of Eq. (22) and the continued fraction of Eq. (24)
# ----------------------------------------------------------------------------------------------------------------------
def mu_k(K):
    """orders mu_k = 3 k + 3/2, k = 0, ..., K - 1"""
    return 3.0*np.arange(K) + 1.5

def D_diag(nu, mu):
    """diagonal elements D(mu) = -7/6 mu^4 + (lambda - 5/6) mu^2 + lambda^2/2, with lambda = nu (nu + 1)"""
    lam = nu*(nu + 1); return -7.0/6.0*mu**4 + (lam - 5.0/6.0)*mu**2 + lam**2/2

def Lambda(nu, mu):
    """Lambda(mu) = prod_{i = 0, 1, 2} (nu - mu - i)(nu + mu + 1 + i)"""
    return (nu - mu)*(nu - mu - 1)*(nu - mu - 2)*(nu + mu + 1)*(nu + mu + 2)*(nu + mu + 3)

def ratios(nu, kap, K):
    """rho_k = c_k/c_{k-1} from Eq. (23), started from rho_K = 0 (K rows of the matrix); returns rho_0 = 1, rho_1, ..., rho_{K-1}"""
    mu = mu_k(K); D = D_diag(nu, mu); L = Lambda(nu, mu); rho = np.zeros(K + 1); rho[0] = 1.0
    for k in range(K - 1, 0, -1):
        rho[k] = SQRT2*k*L[k - 1]/(D[k] - kap - SQRT2*(k + 1)*rho[k + 1])
    return rho[:K]

def kappa(nu, K=24, tol=1e-14, itmax=200):
    """kappa(nu): solution of the continued fraction of Eq. (24), by iteration from kappa = D_0 as described in the Letter.
    It is the largest eigenvalue of the matrix of Eq. (22)."""
    D0 = D_diag(nu, 1.5); kap = D0
    for _ in range(itmax):
        new = D0 - SQRT2*ratios(nu, kap, K)[1] if K > 1 else D0
        if abs(new - kap) <= tol*max(1.0, abs(new)): return new
        kap = new
    return kap

def coefficients(nu, kap, K=24):
    """c_k = rho_1 rho_2 ... rho_k, Eq. (25), with c_0 = 1"""
    return np.cumprod(ratios(nu, kap, K))

def matrix_eigenvalues(nu, K=24):
    """real eigenvalues of the upper-left K x K block of the matrix of Eq. (22), in decreasing order.  The matrix is first
    balanced: only the products of the two elements which couple k and k + 1, 2 (k + 1)^2 Lambda_k, matter."""
    mu = mu_k(K); B = np.diag(D_diag(nu, mu)); p = 2*(np.arange(K - 1) + 1.0)**2*Lambda(nu, mu[:-1]); s = np.sqrt(np.abs(p))
    B[np.arange(K - 1), np.arange(1, K)] = s; B[np.arange(1, K), np.arange(K - 1)] = np.sign(p)*s
    ev = np.linalg.eigvals(B); ev = ev[np.abs(ev.imag) <= 1e-9*np.maximum(1.0, np.abs(ev.real))].real
    return np.sort(ev)[::-1]

# ----------------------------------------------------------------------------------------------------------------------
# Ferrers functions and the series about a corner, Eq. (21)
# ----------------------------------------------------------------------------------------------------------------------
def ferrers(nu, mu, th):
    """Ferrers function P_nu^{-mu}(cos theta), regular at theta = 0, Eq. (S14)"""
    return np.tan(th/2)**mu/gamma(1 + mu)*hyp2f1(-nu, nu + 1, 1 + mu, np.sin(th/2)**2)

def dferrers(nu, mu, th):
    """derivative of P_nu^{-mu}(cos theta) with respect to theta, from the first relation of Eq. (S19)"""
    return ferrers(nu, mu - 1, th) - mu/np.tan(th)*ferrers(nu, mu, th)

def series(nu, c, th, ph):
    """sum_k c_k P_nu^{-mu_k}(cos theta) sin(mu_k phi), Eq. (21); converges for tan(theta/2) < sqrt 2"""
    th, ph = np.asarray(th, float), np.asarray(ph, float); mu = mu_k(len(c))
    return sum(c[k]*ferrers(nu, mu[k], th)*np.sin(mu[k]*ph) for k in range(len(c)))

def center_condition(nu, kap=None, K=24):
    """left-hand side of Eq. (26): theta-derivative of the series at the center c (cos theta = 1/3, phi = pi/3)"""
    if kap is None: kap = kappa(nu, K)
    c = coefficients(nu, kap, K); mu = mu_k(K)
    return sum((-1)**k*c[k]*dferrers(nu, mu[k], THETA_C) for k in range(K))

def solve_nu1(K=24):
    """smallest positive root of Eq. (26), by a root search: nu_1 = 2 beta_3"""
    return brentq(lambda nu: center_condition(nu, K=K), 1.6, 2.0, xtol=1e-15, rtol=1e-15)

# ----------------------------------------------------------------------------------------------------------------------
# Modes of T which are symmetric in the lions (ground state and excited states)
# ----------------------------------------------------------------------------------------------------------------------
class Mode:
    """a mode of T symmetric in the lions: nu, kappa, the coefficients c_k of Eq. (21) and, once normalize() is called,
    its value at the center, its integrals over T and its amplitude A in Eq. (15)"""
    def __init__(self, nu, kap, K=24, rank=0):
        self.nu, self.kappa, self.K, self.rank = nu, kap, K, rank; self.c = coefficients(nu, kap, K)
    def corner(self, th, ph):
        """value at the corner coordinates (theta, phi) about a corner"""
        return series(self.nu, self.c, th, ph)
    def sphere(self, q):
        """value at the unit vectors q (n x 3) inside T, from the expansion about the nearest corner"""
        q = np.atleast_2d(np.asarray(q, float)); q = q/np.linalg.norm(q, axis=1)[:, None]
        near = np.argmax(np.stack([q @ V for V in VS], axis=1), axis=1); out = np.zeros(len(q))
        for i in range(3):
            sel = near == i
            if sel.any():
                th, ph = sphere_to_corner(q[sel], i); out[sel] = self.corner(th, ph)
        return out
    def mirror_defect(self):
        """relative asymmetry under the exchange of two lions which does not leave the corner fixed (mirror q2 = q3).
        It vanishes for a mode which is symmetric in the three lions, and is of order one for a spurious root of Eq. (26)."""
        n = np.array([0.0, 1.0, -1.0])/SQRT2; d = 0.2; worst = 0.0; scale = abs(float(self.corner(THETA_C, np.pi/3)))
        for s in (0.25, 0.5, 0.75, 1.0):
            p = (1 - s)*MS[0] + s*CEN; p /= np.linalg.norm(p)
            vals = [float(self.corner(*sphere_to_corner(np.cos(d)*p + sg*np.sin(d)*n, 0))) for sg in (1, -1)]
            worst = max(worst, abs(vals[0] - vals[1])); scale = max(scale, abs(vals[0]), abs(vals[1]))
        return worst/scale
    def integrate(self, weight=lambda th, ph: 1.0, power=1, n=96):
        """integral over T of weight * mode^power, for integrands with the full symmetry of T: 6 times the integral over the
        cell 0 < phi < pi/3, 0 < theta < arctan(sqrt 2/cos phi) of the points which are closer to this corner and to its wall
        phi = 0.  The substitution theta = theta_max s^2 makes the integrand analytic at the corner."""
        x, w = np.polynomial.legendre.leggauss(n); tot = 0.0
        for xp, wp in zip(x, w):
            ph = (xp + 1)/2*np.pi/3; thmax = np.arctan(SQRT2/np.cos(ph)); s = (x + 1)/2; th = thmax*s**2
            f = weight(th, ph)*self.corner(th, np.full_like(th, ph))**power*np.sin(th)*2*thmax*s
            tot += wp*np.pi/6*np.sum(w/2*f)
        return 6*tot
    def normalize(self):
        """value at the center, integrals over T and amplitude A_j = B_j m_j(c) of Eq. (15), with B_j given by Eq. (14)"""
        self.center = float(self.corner(THETA_C, np.pi/3)); self.int1 = self.integrate(power=1); self.int2 = self.integrate(power=2)
        self.A = gamma((self.nu + 3)/2)/gamma(self.nu + 1.5)*self.int1*self.center/self.int2
        return self

def ground_state(K=24, nu=NU1):
    """ground state m_1 with c_0 = 1"""
    return Mode(nu, kappa(nu, K), K, 0).normalize()

def symmetric_modes(nu_max=14.2, K=24, step=0.02, defect_tol=1e-6, verbose=False):
    """all the modes of T which are symmetric in the lions with nu < nu_max, sorted by increasing nu.
    For each eigenvalue of the matrix of Eq. (22) (rank 0 = largest) the roots of Eq. (26) are bracketed on a grid and refined.
    The eigenvalue of rank n belongs to the allowed projection mu_n = 3 n + 3/2 <= nu.  Eq. (26) is necessary but not
    sufficient: the roots which are not symmetric under the exchange of the third lion are discarded (mirror_defect)."""
    def F(nu, rank):
        ev = matrix_eigenvalues(nu, K)
        return center_condition(nu, ev[rank], K) if rank < len(ev) else np.nan
    modes = []; rank = 0
    while 3*rank + 1.5 < nu_max:
        grid = np.arange(3*rank + 1.5 + 1e-3, nu_max, step); vals = np.array([F(v, rank) for v in grid])
        for a, b, fa, fb in zip(grid[:-1], grid[1:], vals[:-1], vals[1:]):
            if not (np.isfinite(fa) and np.isfinite(fb)) or fa*fb > 0: continue
            nu = brentq(lambda v: F(v, rank), a, b, xtol=1e-14, rtol=1e-15)
            m = Mode(nu, matrix_eigenvalues(nu, K)[rank], K, rank); defect = m.mirror_defect()
            if verbose: print("   rank %d  root nu = %.10f  kappa = %.6f  mirror defect %.1e  %s" % (rank, nu, m.kappa, defect, "mode" if defect < defect_tol else "spurious"))
            if defect < defect_tol: modes.append(m.normalize())
        rank += 1
    return sorted(modes, key=lambda m: m.nu)

def survival(alpha, modes):
    """S_3(t) of Eq. (15) for three lions starting together a distance l behind the lamb; alpha = 3 l^2/(16 D t)"""
    alpha = np.asarray(alpha, float)
    return sum(m.A*alpha**(m.nu/2)*hyp1f1(m.nu/2, m.nu + 1.5, -alpha) for m in modes)

# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    K = 24
    nu1 = solve_nu1(K); kap1 = kappa(nu1, K); g = ground_state(K, nu1)
    print("Ground state (Eqs. (24)-(27))")
    print("   nu_1    = %.13f      (1.8257570041745)" % nu1)
    print("   beta_3  = %.13f      (0.9128785020872)" % (nu1/2))
    print("   kappa_1 = %.10f        (17.3744631043)" % kap1)
    print("   c_k     = " + ", ".join("%.8g" % v for v in g.c[:7]))
    print("   terms of Eq. (21) at the center, relative to the first: " + ", ".join("%.4e" % ((-1)**k*g.c[k]*ferrers(nu1, 3*k + 1.5, THETA_C)/ferrers(nu1, 1.5, THETA_C)) for k in range(1, 7)))
    print("   m_1(c) = %.10f,  int m_1 = %.10f,  int m_1^2 = %.10f      (0.1874437902, 0.2636859555, 0.0315200692)" % (g.center, g.int1, g.int2))
    print("   A_1 = %.6f,   A = A_1 (3/16)^beta_3 = %.6f      (0.712696, 0.154612)" % (g.A, g.A*(3/16)**(nu1/2)))
    print("Truncation of the matrix to K rows (Table S1)")
    for Kt in (1, 2, 3, 4, 5, 6, 8, 10):
        print("   K = %2d   beta_3 = %.10f   kappa = %.7f" % (Kt, brentq(lambda v: center_condition(v, K=Kt), 1.6, 2.0, xtol=1e-14)/2, kappa(nu1, Kt)))
    a = brentq(lambda a: np.tan(a*THETA_C) - 6*SQRT2*a/(19 - 16*a*a), 2.2, 2.4)
    print("   one-term equation (S29): beta_3 = %.7f" % ((a - 0.5)/2))
    # configuration of a surviving lamb: mean lead over the nearest lion, Eq. (S9), in units of sqrt(D t)
    dist = g.integrate(weight=lambda th, ph: np.sin(th)*np.sin(ph))/g.int1          # mean distance of the direction omega to the nearest wall
    print("Mean lead of a surviving lamb = %.5f sqrt(D t)      (1.91837);   <r^2> = %.4f D t      (9.6515)"
          % (SQRT2*2*gamma((nu1 + 4)/2)/gamma((nu1 + 3)/2)*dist, 2*(nu1 + 3)))
    print("Modes which are symmetric in the lions (Table S3 and exact curve of Fig. 5)")
    modes = symmetric_modes(14.2, K, verbose=False)
    print("    j   rank        nu_j              kappa_j              A_j")
    for j, m in enumerate(modes): print("   %2d    %d    %13.10f   %16.6f   %+.8e" % (j + 1, m.rank, m.nu, m.kappa, m.A))
    print("   S_3 at alpha = 1.08, 0.27, 0.0084375 :", ", ".join("%.7f" % survival(x, modes) for x in (1.08, 0.27, 0.0084375)))
