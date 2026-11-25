"""Monte Carlo simulation of the lamb and the lions on a lattice: data of Fig. 5.

    python simulate_survival.py            ->  data/survival_mc.json   (8 x 10^6 realizations, a few minutes on one core)

N + 1 random walkers on the integer lattice all hop by +-1 at every time step, so that D = 1/2.  The lions start at the
origin and the lamb at x = ell.  Only the gaps x_0 - x_i between the lamb and the lions are stored; the lamb is captured as
soon as one of them vanishes (ell is even, so a gap cannot jump over zero).  Captured realizations are dropped, which makes
the cost of a time step proportional to the number of survivors.  The survival probability is recorded at a list of times.
The generator is seeded, so the output is reproducible."""
import json, os, sys, time
import numpy as np

def simulate(N, ell, ntraj, checkpoints, seed=7, chunk=500_000):
    """fraction of the ntraj realizations in which the lamb is still alive after the numbers of steps listed in checkpoints"""
    rng = np.random.default_rng(seed)
    surv = np.zeros(len(checkpoints))
    for start in range(0, ntraj, chunk):
        n0 = min(chunk, ntraj - start); gaps = np.full((n0, N), ell, dtype=np.int32); t = 0
        for k, T in enumerate(checkpoints):
            while t < T and gaps.shape[0] > 0:
                n = gaps.shape[0]
                # hop of the lamb (one per realization) minus hop of each lion
                gaps += (rng.integers(0, 2, size=(n, 1), dtype=np.int8)*2 - 1) - (rng.integers(0, 2, size=(n, N), dtype=np.int8)*2 - 1)
                t += 1; dead = (gaps <= 0).any(axis=1)
                if dead.any(): gaps = gaps[~dead]
            surv[k] += gaps.shape[0]
    return surv/ntraj

if __name__ == '__main__':
    N, ELL, NTRAJ, SEED = 3, 12, 8_000_000, 7
    STEPS = [50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600]
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'survival_mc.json')
    t0 = time.time(); S = simulate(N, ELL, NTRAJ, STEPS, seed=SEED)
    rows = [[T, float(s), float(np.sqrt(s*(1 - s)/NTRAJ))] for T, s in zip(STEPS, S)]
    json.dump(dict(description="lamb and N lions on a lattice, all walkers hop by +-1 at each step (D = 1/2); rows = [number of steps, survival probability, standard error]",
                   N=N, ell=ELL, D=0.5, ntraj=NTRAJ, seed=SEED, rows=rows), open(out, 'w'), indent=1)
    print("%d realizations in %.0f s  ->  %s" % (NTRAJ, time.time() - t0, out))
    for T, s, e in rows: print("  steps %6d   D t/ell^2 = %8.3f   S = %.6f +- %.6f" % (T, 0.5*T/ELL**2, s, e))
