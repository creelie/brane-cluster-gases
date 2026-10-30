# Brane cluster gases and ghost-free form factors for quantum gravity

Source, figures and verification code for three manuscripts by Deep Bhattacharjee
and Ushashi Bhattacharya, prepared for submission to *Physical Review D*.
Contact: <d.bhattacharjee@erl-forschung.de>, <itsdeep@live.com>; GitHub-related contact: <egspl@proton.me>.

1. *Brane cluster gases and ghost-free form factors for quantum gravity*
   (`paper/`, `code/`).
2. *Perturbative finiteness of gravity in a scale-invariant gas of branes*
   (`uv-finite/`, added in release 2.0.0 as "Finite quantum gravity from a
   scale-invariant gas of branes", retitled in 3.1.0 and 3.4.0). See
   [the section below](#perturbative-finiteness-of-gravity-in-a-scale-invariant-gas-of-branes).
3. *Ghosts in Poisson Brane Gases* (`ghosts/`, added in release 3.0.0). See
   [the section below](#ghosts-in-poisson-brane-gases).

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
ghosts/     the third manuscript: paper/, code/ and lean/ (see ghosts/README.md)
uv-finite/  the second manuscript: uv-finite-brane-gas.tex, its PDF, figs/,
            code/ (verify.py, formfactor_check.c, verify.jl,
            PowerCounting.lean, make_figures.py) and audit/ (five
            independent re-derivations, see audit/README.md)
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

The suite runs 176 checks in eleven groups, matching the eleven items of Sec. IX of
the paper, and prints one line per check followed by

```
176 checks passed, 0 failed.
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

## Ghosts in Poisson brane gases

The third manuscript, in `ghosts/`, computes the gas average of the
propagator itself, `E[1/(z F(z))] = exp(-phi(z))/z`, and shows that it differs
from `1/(z E F(z))`: the averaged propagator has total massive
spectral weight `exp(-|nu|) - 1 < 0`, and its spectrum is made of ghosts alone
when the brane density per unit `log t` is at most one. See
[`ghosts/README.md`](ghosts/README.md).

## Releases

Releases are published by the `release` workflow: merging a change to
`.github/release/notes.md` into `main` publishes the release for the version in
`CITATION.cff`, with the PDF, LaTeX source and arXiv package of all three
manuscripts and the second manuscript's figures attached (built by
`.github/release/build.sh`). Every release is listed at
https://github.com/creelie/brane-cluster-gases/releases.

- [v3.4.0](https://github.com/creelie/brane-cluster-gases/releases/tag/v3.4.0): the second manuscript corrected and retitled "Perturbative finiteness of gravity in a scale-invariant gas of branes", with a short abstract, the preprint id hexagonmath.org 2610.00342v2 on the title page, and the unitarity statement made explicit about its input (the cutting-rule theorem of Pius and Sen and of Briscese and Modesto, written out for scalar fields).
- [v3.3.0](https://github.com/creelie/brane-cluster-gases/releases/tag/v3.3.0): the third manuscript in its journal form.
- [v3.2.0](https://github.com/creelie/brane-cluster-gases/releases/tag/v3.2.0): the second manuscript in the form submitted to the
  Journal of High Energy Physics, with its PDF and source in `paper/paper1/`
  and `paper/paper2/`.
- [v3.1.0](https://github.com/creelie/brane-cluster-gases/releases/tag/v3.1.0): the second manuscript retitled, with both authors,
  three new figures and the audit scripts.
- [v3.0.0](https://github.com/creelie/brane-cluster-gases/releases/tag/v3.0.0): the third manuscript, "Ghosts in Poisson Brane Gases".
- [v2.2.0](https://github.com/creelie/brane-cluster-gases/releases/tag/v2.2.0): the corrected first manuscript.

The second manuscript cites release v3.4.0, which contains its code, figures
and audit, through its GitHub link. Its two-column REVTeX source is
`uv-finite/uv-finite-brane-gas.tex`; the one-column JHEP source,
`uv-finite/uv-finite-brane-gas-jhep.tex` (with `jheppub.sty`), is the version in
`paper/paper1/` (`uv-finite-brane-gas-jhep.pdf`) and `paper/paper2/`
(`uv-finite-brane-gas-jhep-tex.zip`).

## Perturbative finiteness of gravity in a scale-invariant gas of branes

The second manuscript derives the form factor from a Poisson gas of branes with
no preferred size and works with one frozen realization of the gas rather than
an average. The single-brane factor is derived from a model of a brane as a
layer of independent cells that shift the proper time of a crossing graviton,
and the fine graining of the gas is shown to be necessary, up to slowly varying
factors, for a zero-point energy that does not depend on the regulator. Almost
every realization then gives an entire form factor without zeros, and for
`n >= 4` the paper proves ultraviolet finiteness at every order of perturbation
theory around flat space, unitarity at every loop order (the Lorentzian
amplitudes are the unique analytic continuation of the Euclidean ones), and a
finite graviton zero-point energy, so the theory is ultraviolet complete in
perturbation theory. A one-loop bubble computed in closed form confirms the
cutting rule, but it grows faster than any exponential of the energy and
overtakes the tree amplitude between 1.4M and 1.9M, so the loop expansion is
an expansion in a small number only up to energies of order `M`. Infrared
divergences are physical and are not removed. The paper's discussion lists the
open points.

Every result is derived analytically in the text. The code only re-evaluates
the formulas at sample points and samples realizations of the gas:

```
pip install -r requirements.txt
cd uv-finite/code
python3 verify.py                  # 195 checks passed, 0 failed
cc -O2 -std=c99 -o formfactor_check formfactor_check.c -lm
./formfactor_check                 # 84 checks passed, 0 failed
julia verify.jl                    # 114 checks passed, 0 failed (Julia 1.10, stdlib only)
lean PowerCounting.lean            # compiles with no output (Lean 4.15, core only)
python3 make_figures.py            # writes the fifteen PNG figures to ../figs/
cd ../audit
python3 run_all.py                 # five audit scripts, each PASS/FAIL per check
```

The audit scripts in `uv-finite/audit/` re-derive, by methods different from
those of the paper, the one-loop pole matrix and its tuning (exact rational
arithmetic), the superficial degree (exhaustive enumeration of vertex
degrees), the Mellin transform of the vacuum energy and its regulator
independence, the failure of reflection positivity (a Hankel test on the
time-slice covariance) and the one-loop bubble table (adaptive quadrature in
polar coordinates about the graviton pole, with the tail to infinity);
`audit/README.md` records what each one found.

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
identifiers and what each one resolves to. The bibliography of the second
manuscript was checked again in October 2026: all 95 identifiers resolve, through
the Crossref registration record, to the cited title and authors.

## Statements

Funding: none. Conflict of interest: none. Both appear as back-matter sections
of the manuscript.

## Licence

The code in `code/` and `uv-finite/code/` is released under the MIT Licence
(see `LICENSE`). The manuscripts and figures in `paper/` and `uv-finite/` are
released under CC BY 4.0.
