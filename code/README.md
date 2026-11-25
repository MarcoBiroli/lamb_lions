# Code for the figures

Python scripts which produce the five figures of the Letter, and a small module, `lamblions.py`, which implements the
exact solution as it is written in the Letter (in double precision). Equation numbers below refer to the Letter,
(S..) to its Supplemental Material.

## Requirements

Python 3 with `numpy`, `scipy` and `matplotlib` (the STIX fonts used in the figures ship with matplotlib).
The figures of the manuscript were produced with Python 3.13.18, numpy 2.5.3, scipy 1.18.1 and matplotlib 3.11.2.

## How to run

Run the scripts from this folder. Each one writes its figure to `../figs/`, the folder read by `lamb_lions.tex`.

```
python make_fig_sketch.py      # Fig. 1  ->  figs/fig_sketch.pdf      ~2 s
python make_fig_config.py      # Fig. 2  ->  figs/fig_config.pdf      ~30 s
python make_fig_billiard.py    # Fig. 3  ->  figs/fig_billiard.pdf    ~2 s
python make_fig_corner.py      # Fig. 4  ->  figs/fig_corner.pdf      ~2 s
python make_fig_survival.py    # Fig. 5  ->  figs/fig_survival.pdf    ~3 s
python lamblions.py            # prints the numbers quoted in the Letter (a few seconds)
python simulate_survival.py    # regenerates data/survival_mc.json     ~80 s on one core
```

## Files

| File | What it does |
|---|---|
| `lamblions.py` | The solution: `kappa(nu)` from the continued fraction of Eq. (24) by iteration, coefficients `c_k` of Eq. (25), root search on the condition at the center, Eq. (26), ground state from the series of Eq. (21), excited modes which are symmetric in the lions with their amplitudes `A_j`, and `S_3(t)` of Eq. (15). |
| `figstyle.py` | Colours, fonts and output folder shared by the figure scripts. |
| `make_fig_sketch.py` | Fig. 1: sample trajectories of the lamb and of three lions up to the capture. |
| `make_fig_config.py` | Fig. 2: wedge for two lions with a sample trajectory, cone for three lions in three dimensions, triangle `T` in stereographic projection with its ground state. |
| `make_fig_billiard.py` | Fig. 3: a billiard trajectory in `T` and the squared components of its angular momentum; prints `L1^4 + L2^4 + L3^4` on each segment. |
| `make_fig_corner.py` | Fig. 4: zoom on a corner with the coordinates (theta, phi) and the ground state. |
| `make_fig_survival.py` | Fig. 5: `S_3(t)` from Eq. (15) with the eleven symmetric modes with nu_j < 14.2, against the Monte Carlo data. |
| `simulate_survival.py` | Monte Carlo simulation behind Fig. 5: lattice walks with D = 1/2, l = 12, 8 x 10^6 realizations, seeded. |
| `data/survival_mc.json` | Output of `simulate_survival.py`: number of steps, survival probability, standard error. |

## What `python lamblions.py` prints

- nu_1 = 1.8257570041745, beta_3 = 0.9128785020872, kappa_1 = 17.3744631043 (Eqs. (24)-(27));
- the coefficients c_k and the ratios of the terms of the series at the center (Table S2);
- m_1(c), the integrals of m_1 and m_1^2 over T, A_1 = 0.712696 and A = 0.154612 (Eq. (1), Eqs. (S30)-(S31));
- beta_3 with K rows of the matrix (Table S1) and the root of the one-term equation (S29);
- the mean lead of a surviving lamb, 1.91837 sqrt(D t) (Eq. (S9));
- the eleven modes symmetric in the lions with nu_j < 14.2: nu_j, kappa_j, A_j (Table S3 lists the first five).

## Notes

- Excited modes. For each real eigenvalue of the matrix of Eq. (22) (rank 0 is the largest) the roots of Eq. (26) are
  bracketed on a grid in nu and refined. Eq. (26) is necessary but not sufficient, so each root is tested for the symmetry
  under the exchange of the third lion (`Mode.mirror_defect`) and the spurious ones are discarded.
- Random numbers. The sample trajectories of Figs. 1, 2(a) and 3 are selected by seeded searches, and the Monte Carlo
  run is seeded. With the versions listed above the scripts reproduce the figures of the manuscript and the file
  `data/survival_mc.json` exactly. Another version of numpy may produce other random streams, hence other (equally valid)
  samples.
