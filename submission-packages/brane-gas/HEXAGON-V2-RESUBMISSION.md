# Hexagon (hexagonmath.org) — version 2 of the deposit, with resubmission notes

Files to upload: the same two files as for JHEP, `uv-finite-brane-gas-jhep.pdf` (the compiled
paper, 83 pages, one column) and `uv-finite-brane-gas-jhep-tex.zip` (the source:
`uv-finite-brane-gas-jhep.tex` with the bibliography inside, `jheppub.sty` and `figs/`; three
`pdflatex` passes compile it on its own; no bad box, no undefined reference). Hexagon does not accept code; the verification
suite and the audit scripts are linked from the Data availability section.

## Fields (unchanged from version 1 unless stated)

- Title: Cancellation of all ultraviolet divergences of quantum gravity by a scale-invariant gas
  of branes
- Contributors: 1. Deep Bhattacharjee — ORCID 0000-0003-0466-750X — itsdeep@live.com — Formerly,
  Electro-Gravitational Space Propulsion Laboratory (EGSPL), Bhubaneswar, Odisha 751030, India;
  Formerly, University of Calcutta, Senate House, 87/1 College Street, Kolkata 700073, West
  Bengal, India (corresponding). 2. Ushashi Bhattacharya — ORCID 0009-0002-2254-3914 —
  rai.bhattacharya.in@gmail.com — Formerly, National Taiwan University, Taipei 106319, Taiwan.
- Subjects: primary math-ph; secondary hep-th, gr-qc.
- MSC 2020: primary 81T15, 83C45; secondary 81T18, 60G55, 81T08, 81U20.
- Keywords: nonlocal quantum gravity; ultraviolet finiteness; brane gas; entire form factors;
  Poisson point processes; perturbative unitarity.
- Abstract: the first page of the PDF (plain-text copy below).
- License: CC BY 4.0.
- Related work: earlier preprint by the same contributors, "Ghosts in Poisson Brane Gases",
  hexagon:2610.00342v1 (cited on the title page); companion manuscript "Brane cluster gases and
  ghost-free form factors for quantum gravity" (the archive of Ref. [13]); code and figure
  scripts at https://github.com/creelie/brane-cluster-gases, release 3.2.0,
  https://github.com/creelie/brane-cluster-gases/releases/tag/v3.2.0.
- AI field: the text of `AI-PROVENANCE.md`.
- Journal field (if asked): submitted to the Journal of High Energy Physics.

## Resubmission notes (the "changes in this version" field)

Version 2 is the manuscript as submitted to the Journal of High Energy Physics, in its
one-column layout. Relative to
version 1 of this deposit:

1. The theorem, its proof and every result of the paper are unchanged: no formula, bound,
   figure or table changed. The layout is now one column (the JHEP class) instead of the
   two-column REVTeX layout of version 1.
2. The paper names release 3.2.0 of the archive of its code, figures and audit, which also
   holds the compiled manuscript and its source as `paper/paper1/` and `paper/paper2/`, and cites
   it by its release link, https://github.com/creelie/brane-cluster-gases/releases/tag/v3.2.0.
3. All eighty-eight registered references of the bibliography were re-checked against their
   Crossref or Zenodo records on 21 October 2026; the entries are unchanged.
4. The scope of the result is stated in the same words in the title, the abstract, Sec. VIII D
   ("In what sense the theory is ultraviolet complete"), the status table and the discussion:
   finite and unitary at every order of perturbation theory for n >= 4; the free Euclidean field
   is a Gaussian measure that is not reflection positive; no construction beyond perturbation
   theory is claimed, and the three obstacles to one (the failure of reflection positivity for
   every entire non-constant form factor, the Euclidean conformal mode, the strong coupling above
   an energy between 1.4M and 1.9M) are named.

## Abstract (plain text)

A frozen Poisson gas of branes with no preferred size, each brane a layer of independent cells
that shift the proper time of a crossing graviton, multiplies the graviton kinetic operator by an
entire function without zeros, exp[(n/2) Ein(Box^2/M^4)] times a random factor that tends to a
constant at high momentum, in almost every realization. We prove that for n >= 4 the theory of
gravity built on it has no ultraviolet divergence at any order of perturbation theory: every
graph with two or more loops converges, the one-loop poles are those of a local theory and are
cancelled by three operators quartic in curvature with finite coefficients, Newton's constant and
the cosmological constant do not run, the zero-point energy of the graviton is 3 n M^4/(64 pi^2)
for every regulator, and the physical S matrix is unitary at every loop order. The propagator
keeps the graviton pole alone, and the potential of a point mass is finite at the origin. A
one-loop graph in closed form confirms the cutting rule and grows faster than any exponential of
the energy, so the loop expansion is an expansion in a small parameter only below an energy
between 1.4M and 1.9M. The free Euclidean field is a Gaussian measure with finite curvature that
is not reflection positive, which states what a construction beyond perturbation theory would
have to supply.
