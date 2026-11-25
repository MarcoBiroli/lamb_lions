"""Fig. 5: survival probability of the lamb for three lions, exact result against Monte Carlo simulations.

    python make_fig_survival.py    ->   figs/fig_survival.pdf      (a few seconds)

Exact curve: Eq. (15), summed over the modes of T which are symmetric in the lions with nu_j < 14.2 (eleven modes); they are
computed by lamblions.symmetric_modes() from the matrix of Eq. (22) and the condition at the center, Eq. (26).
Symbols: data/survival_mc.json, written by simulate_survival.py (lattice walks, D = 1/2, ell = 12, 8 x 10^6 realizations)."""
import json
import numpy as np
from figstyle import *
from lamblions import symmetric_modes, survival

modes = symmetric_modes(14.2)
print("%d modes, nu_j = %s" % (len(modes), ", ".join("%.4f" % m.nu for m in modes)))
mc = json.load(open(os.path.join(HERE, 'data', 'survival_mc.json'))); rows = np.array(mc['rows']); ell, D = mc['ell'], mc['D']
T, s, err = rows.T
x = D*T/ell**2                                   # D t/l^2
alpha_mc = 3*ell**2/(16*D*T); th = survival(alpha_mc, modes)
alpha = np.geomspace(1.15, 0.0017, 300); xg = 3/(16*alpha)
Sg = survival(alpha, modes); Lg = modes[0].A*alpha**(modes[0].nu/2)          # all modes, and leading term A (l^2/D t)^beta_3
print("   D t/l^2    S (Monte Carlo)           S (exact)     ratio")
for xi, si, ei, ti in zip(x, s, err, th): print("  %8.3f   %.6f +- %.6f   %.6f   %.4f +- %.4f" % (xi, si, ei, ti, si/ti, ei/ti))

fig, (a1, a2) = plt.subplots(2, 1, figsize=(3.375, 2.80), sharex=True, gridspec_kw=dict(height_ratios=[3, 1.1], hspace=0.08))
a1.loglog(xg, Lg, color=ORANGE, lw=1.3, ls=(0, (5, 3)), label='leading term  $A(\\ell^2/Dt)^{\\beta_3}$')
a1.loglog(xg, Sg, color=BLUE, lw=1.4, label='exact result')
a1.loglog(x, s, 'o', ms=4.6, color=INK, mec=SURF, mew=0.8, label='Monte Carlo, $8\\times10^6$ runs', zorder=5)
a1.set_ylabel('survival probability  $S_3(t)$'); a1.set_ylim(1.5e-3, 1.0); a1.grid(True, which='major', color='#e6e5e1', lw=0.4); a1.set_axisbelow(True)
a1.legend(frameon=False, fontsize=7.5, loc='lower left', labelcolor=INK2)
a1.text(xg[60]*1.25, Lg[60]*1.5, 'slope $-\\beta_3=-%.5f$' % (modes[0].nu/2), color=INK2, fontsize=7.5)
a2.axhline(1, color=BLUE, lw=1.2)
a2.errorbar(x, s/th, yerr=err/th, fmt='o', ms=4.2, color=INK, mec=SURF, mew=0.7, ecolor=INK2, elinewidth=0.7, capsize=1.6, zorder=5)
a2.set_xscale('log'); a2.set_ylim(0.985, 1.015); a2.set_yticks([0.99, 1.0, 1.01]); a2.set_ylabel('MC / exact'); a2.set_xlabel('time  $Dt/\\ell^2$')
a2.grid(True, which='major', color='#e6e5e1', lw=0.4); a2.set_axisbelow(True)
for a in (a1, a2):
    for sp in ('top', 'right'): a.spines[sp].set_visible(False)
fig.align_ylabels([a1, a2]); save(fig, 'fig_survival', bbox_inches='tight', pad_inches=0.02)
