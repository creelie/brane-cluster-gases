# Ghosts in Poisson brane gases

Source, figures and checks for the paper *Ghosts in Poisson Brane Gases* by
Deep Bhattacharjee and Ushashi Bhattacharya.

The paper computes the propagator of a graviton in a Poisson gas of branes,
averaged over the gas, exactly: for intensity measure `nu` it is
`E[1/(z F(z))] = exp(-phi(z))/z` with `phi(z) = int z/(z+t) nu(dt)`. It is not
`1/(z E F(z))`, the expression used in the two earlier manuscripts of this
repository. The averaged propagator has no positive Kallen-Lehmann
representation; for `nu(dt) = lambda(t) dt/t` its massive spectrum consists of
ghosts alone exactly when `lambda <= 1`, and every average with a finite
four-dimensional tadpole has states of both signs. All proofs are in the paper,
by hand.

## Journal version

The journal version, for Physical Review D (no author charge: PRD articles in
high-energy physics are covered by SCOAP3), is `paper/paper1/GhostsBraneGases.pdf`
with its LaTeX source in `paper/paper2/GhostsBraneGases-tex.zip`. The paper
cites release 3.3.0 of this repository,
https://github.com/creelie/brane-cluster-gases/releases/tag/v3.3.0.

## Layout

```
paper/   main.tex (REVTeX, PRD style), main.pdf, figures/ (PNG),
         figsrc/ (TikZ sources of the figures), figdata/ (plotted tables)
code/    verify_ghosts.py (19 checks), residues.c (5 checks),
         verify.jl (9 checks), make_figure_data.py
lean/    BraneGhosts.lean (Lean 4 with Mathlib), Axioms.lean
```

## Running the checks

```
cd code
python3 verify_ghosts.py                 # needs numpy, scipy, mpmath
cc -O2 -o residues residues.c -lm && ./residues
julia verify.jl
cd ../lean && lake build && lake env lean Axioms.lean
```

Each script prints `N checks passed, 0 failed`. The Lean file proves the sign
of the residues, the sum rule, the elementary inequalities of Appendix A and
the sign of the spectral density; every theorem depends only on `propext`,
`Classical.choice` and `Quot.sound`.

## Building the paper and figures

```
cd code && python3 make_figure_data.py ../paper/figdata
cd ../paper/figsrc && sh build.sh        # pdflatex, then PNG at 300 dpi
cd .. && latexmk -pdf main.tex
```
