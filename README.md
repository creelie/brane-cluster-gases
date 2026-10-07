# Brane cluster gases and ghost-free form factors for quantum gravity

Source, figures and verification code for two manuscripts by Deep Bhattacharjee,
prepared for submission to *Physical Review D*. Contact: <itsdeep@live.com>.

1. *Brane cluster gases and ghost-free form factors for quantum gravity*
   (`paper/`, `code/`).
2. *Finite quantum gravity from a scale-invariant gas of branes*
   (`uv-finite/`, added in release 2.0.0). See
   [the section below](#finite-quantum-gravity-from-a-scale-invariant-gas-of-branes).

## What the paper does

A finite arrangement of branes coupled multiplicatively to the graviton dresses
the graviton kinetic operator by a polynomial form factor, one linear factor per
brane. Every zero of that polynomial is a pole of the spin-two propagator, and
the residues at the massive poles sum to `-1`, so they cannot all belong to
healthy states. Replacing the arrangement by a Poisson gas replaces the product
by its expectation, which is the probability generating functional of the
process and therefore the exponential of an entire function; an exponential has
no zeros, so the averaged propagator retains a single pole, at `p^2 = 0`, with
residue `+1`.

The word *ghost free* is used in the paper for the pole content of the
tree-level propagator. It carries no claim about unitarity of the interacting
theory; that limitation is stated in the introduction and again in the
discussion.

## Layout

```
paper/      brane-gases.tex, the compiled PDF, and the eight figures it includes
code/       the verification suite, the eight figure scripts, and the shared style
docs/       the DOI verification record
uv-finite/  the second manuscript: uv-finite-brane-gas.tex, its PDF, figs/,
            and code/ (verify.py, formfactor_check.c, verify.jl,
            PowerCounting.lean, make_figures.py)
```

## Building the paper

```
cd paper
pdflatex brane-gases.tex
pdflatex brane-gases.tex
pdflatex brane-gases.tex
```

Requires REVTeX 4.2 together with `fontenc`, `inputenc`, `amsmath`, `amssymb`,
`amsfonts`, `mathtools`, `amsthm`, `enumitem`, `graphicx`, `orcidlink`, `xurl`
and `hyperref`, all of which are in TeX Live. The bibliography is inline, so no
BibTeX run is needed. The build is clean: no errors, no warnings, no overfull or
underfull boxes, and no undefined references or citations.

## Reproducing the numbers

```
pip install -r requirements.txt
cd code
python3 checks_clusters.py
```

The suite runs 171 checks in eleven groups, matching the eleven items of Sec. IX of
the paper, and prints one line per check followed by

```
171 checks passed, 0 failed.
```

No check uses a formula proved in the paper as an input; each one rebuilds the
object from its definition and measures the quantity directly.

## Reproducing the figures

```
cd code
python3 fig01_arrangement.py
python3 fig02_torus.py
python3 fig03_formfactor.py
python3 fig04_residues.py
python3 fig05_propagator.py
python3 fig06_schwinger.py
python3 fig07_transfer.py
python3 fig08_annealed.py
```

Each script writes `paper/figs/<name>.png` at 600 dpi, writes a vector PDF where
the figure is line art, and runs the label collision test in `figstyle.py`
before saving. Every figure reports `0 collisions`.

## Archiving this repository and citing the archive

The Data Availability section of the manuscript cites this repository through a
switch and a macro, both defined near the top of `paper/brane-gases.tex`:

```tex
\newif\ifzenodo
\zenodofalse
\newcommand{\zenododoi}{10.5281/zenodo.0000000}
```

While the switch is false the section describes the code as supplied with the
article and makes no claim about a repository, so no placeholder identifier can
reach the compiled PDF. To complete the link:

1. Push this repository to GitHub.
2. Enable it in Zenodo under *GitHub* in your account settings.
3. Create a release on GitHub. Zenodo archives it and mints a DOI, reading the
   title, authors, ORCIDs, licence and keywords from `.zenodo.json`.
4. Put the concept DOI Zenodo returns, the one that always resolves to the
   newest release, in `\zenododoi`, change `\zenodofalse` to `\zenodotrue`, and
   recompile.

Nothing else in the source needs to change.

## Finite quantum gravity from a scale-invariant gas of branes

The second manuscript derives the form factor from a Poisson gas of branes with
no preferred size, each multiplying the graviton kinetic operator by a passive,
saturating factor without resonances, and works with one frozen realization of
the gas rather than an average. Under its two stated premises (the single-brane
conditions and the fine graining of the gas) almost every realization gives an
entire form factor without zeros, and for `n >= 4` the paper proves
ultraviolet finiteness at every order of perturbation theory around flat space,
unitarity at every loop order in the Efimov-Pius-Sen prescription, and a finite
graviton zero-point energy, so the theory is ultraviolet complete in
perturbation theory (its Sec. VII D says exactly in what sense, and what a
nonperturbative definition would still need). Infrared divergences are physical
and are not removed. The open points are listed in its Sec. IX.

Every result is derived analytically in the text. The code only re-evaluates
the formulas at sample points and samples realizations of the gas:

```
pip install -r requirements.txt
cd uv-finite/code
python3 verify.py                  # 178 checks passed, 0 failed
cc -O2 -std=c99 -o formfactor_check formfactor_check.c -lm
./formfactor_check                 # 80 checks passed, 0 failed
julia verify.jl                    # 106 checks passed, 0 failed (Julia 1.10, stdlib only)
lean PowerCounting.lean            # compiles with no output (Lean 4.15, core only)
python3 make_figures.py            # writes the ten PNG figures to ../figs/
```

The Python, C and Julia programs share no code. The Julia program works in
256-bit arithmetic with its own special functions and quadrature. The Lean file
proves the integer statements behind the power counting and the one-loop tuning
for all values of their arguments; the analytic estimates are proved in the
paper and are not formalized.

To build the paper, run `pdflatex uv-finite-brane-gas.tex` three times in
`uv-finite/`. It uses REVTeX 4.2 and TikZ from TeX Live, and the bibliography
is inline.

## References

Every digital object identifier in the bibliography was resolved against its
registration record before inclusion; `docs/doi-verification.md` lists the 42
identifiers and what each one resolves to.

## Statements

Funding: none. Conflict of interest: none. Both appear as back-matter sections
of the manuscript.

## Licence

The code in `code/` and `uv-finite/code/` is released under the MIT Licence
(see `LICENSE`). The manuscripts and figures in `paper/` and `uv-finite/` are
released under CC BY 4.0.
