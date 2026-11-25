"""Fig. 1: typical trajectories of a lamb chased by three lions, up to the capture.

    python make_fig_sketch.py    ->   figs/fig_sketch.pdf      (a few seconds)

Four independent Brownian paths are drawn for many seeds; the lamb starts a distance ell ahead of the lions.  Among the
samples in which the capture occurs in the last third of the window, the most readable one is kept (well separated lions
at the end, one near miss earlier)."""
import numpy as np
from figstyle import *

n, dt, ell = 3000, 1.0/3000, 0.45
def paths(seed):
    rng = np.random.default_rng(seed)
    X = np.cumsum(rng.normal(0, np.sqrt(dt), size=(4, n)), axis=1); X = np.concatenate([np.zeros((4, 1)), X], axis=1); X[1:] -= ell
    return X                                    # X[0]: lamb, X[1:]: lions
best = None
for seed in range(4000):
    X = paths(seed); gap = X[0][None, :] - X[1:]; hit = np.where((gap <= 0).any(axis=0))[0]
    if len(hit) == 0: continue
    tc = hit[0]; who = int(np.argmax(gap[:, tc] <= 0))
    if not (0.68*n < tc < 0.80*n): continue
    g = gap[:, :tc]
    near = (g.min(axis=0)[int(0.1*n):int(0.5*n)]).min(); spread = np.ptp(X[1:, tc])
    score = spread - 3*abs(near - 0.12) + 0.5*(X[0, :tc].max() - X[0, :tc].min())
    if best is None or score > best[0]: best = (score, seed, tc, who)
score, seed, tc, who = best
X = paths(seed); t = np.arange(n + 1)*dt
print("seed", seed, "capture step", tc, "by lion", who + 1)

fig, ax = plt.subplots(figsize=(3.375, 2.25))
for i in (1, 2, 3):
    ax.plot(t[:tc+1], X[i, :tc+1], color=BLUE, lw=0.8, alpha=0.95, label='lions' if i == 1 else None)
ax.plot(t[:tc+1], X[0, :tc+1], color=ORANGE, lw=1.1, label='lamb')
ax.plot(t[tc], X[0, tc], 'o', ms=6, mfc='none', mec=INK, mew=1.0, zorder=6)
ax.annotate('capture', xy=(t[tc], X[0, tc]), xytext=(t[tc] + 0.025, X[0, tc] + 0.30), color=INK, fontsize=8, ha='left', arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.6))
ax.annotate('', xy=(-0.012, 0), xytext=(-0.012, -ell), arrowprops=dict(arrowstyle='<->', color=INK2, lw=0.7, shrinkA=0, shrinkB=0))
ax.text(-0.03, -ell/2, r'$\ell$', color=INK, fontsize=9, ha='right', va='center')
ax.legend(frameon=False, fontsize=8, loc='lower left', bbox_to_anchor=(0.06, 0.02), labelcolor=INK2, handlelength=1.6)
ax.set_xlim(-0.07, t[tc] + 0.17); ax.set_xticks([0, t[tc]]); ax.set_xticklabels(['0', r'$t_{\rm c}$']); ax.set_yticks([])
ax.set_xlabel('time'); ax.set_ylabel('position')
for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
ax.axvline(t[tc], color=MUTED, lw=0.5, ls=(0, (3, 3)), zorder=0)
fig.tight_layout(pad=0.3); save(fig, 'fig_sketch')
