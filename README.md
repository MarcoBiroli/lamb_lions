# Capture of a diffusing lamb by three diffusing lions: hidden symmetry and an exact solution

Manuscript, Supplemental Material, figures and code for the Letter by M. Biroli.

A lamb and three lions diffuse on a line, and the lamb survives up to time `t` with a probability which decays as
`t^(-beta_3)`. The Letter shows that this problem has a hidden conserved quantity, which gives the exponent
`beta_3 = 0.912878502087...` as the root of an explicit equation, and the survival probability at all times.

This is a draft which is still being revised.

## Contents

| Path | What it is |
|---|---|
| `lamb_lions.tex` | The Letter. It inputs `supplement.tex` after the bibliography. |
| `supplement.tex` | The Supplemental Material. |
| `refs.bib` | Bibliography. |
| `lamb_lions.pdf` | Compiled Letter and Supplemental Material. |
| `figs/` | The five figures of the Letter. |
| `code/` | Python code for the solution, the figures and the simulations. See `code/README.md`. |

## Using the code

The module `code/lamblions.py` implements the solution as it is written in the Letter, in double precision. It needs
Python 3 with `numpy`, `scipy` and, for the figures, `matplotlib`.

```
cd code
python lamblions.py            # prints the numbers quoted in the Letter
python make_fig_survival.py    # one of the five figure scripts
python simulate_survival.py    # reruns the Monte Carlo simulation of Fig. 5
```

`code/README.md` lists every script, what it produces and how long it takes.

## Compiling the paper

```
latexmk -pdf lamb_lions.tex
```
