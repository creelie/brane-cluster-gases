# JHEP submission — files, form fields and cover letter

Files in this folder for JHEP:

- `uv-finite-brane-gas-jhep-tex.zip` — the single tex.zip in the JHEP class:
  `uv-finite-brane-gas-jhep.tex` (bibliography inside, no `.bib`), `jheppub.sty` and the figure
  panels in `figs/`. Three `pdflatex` passes compile it on its own: 83 pages, no bad box, no
  undefined reference. The same two files are the Hexagon version 2 upload. The content is that
  of the two-column REVTeX source `uv-finite/uv-finite-brane-gas.tex` of the repository; only the class, the front matter (`\author`, `\affiliation`, `\emailAdd`, `\abstract`,
  `\keywords`), single-column floats and table rules differ.
- `uv-finite-brane-gas-jhep.pdf` — the compiled JHEP-layout paper, to upload as the PDF.
- `COVER-LETTER-JHEP.md` — the cover letter.
- `AI-PROVENANCE.md` — the provenance statement (JHEP has no separate field; the paper's "Use of
  AI tools" section is the disclosure).

## Submission (https://jhep.sissa.it, "Submit a paper")

- Journal: JHEP. Article type: regular article, hep-th.
- Title: Cancellation of all ultraviolet divergences of quantum gravity by a scale-invariant gas
  of branes
- Authors: 1. Deep Bhattacharjee — ORCID 0000-0003-0466-750X — itsdeep@live.com (second address
  d.bhattacharjee@erl-forschung.de) — Formerly, Electro-Gravitational Space Propulsion Laboratory
  (EGSPL), Bhubaneswar, Odisha 751030, India; Formerly, University of Calcutta, Senate House, 87/1
  College Street, Kolkata 700073, West Bengal, India. Corresponding author. 2. Ushashi
  Bhattacharya — ORCID 0009-0002-2254-3914 — rai.bhattacharya.in@gmail.com — Formerly, National
  Taiwan University, Taipei 106319, Taiwan.
- Abstract: the first page of the PDF.
- Keywords (JHEP list, set in the paper): Models of Quantum Gravity; Renormalization and
  Regularization; Scattering Amplitudes.
- MSC 2020: primary 81T15, 83C45; secondary 81T18, 60G55, 81T08, 81U20. (JHEP does not ask for it; it is given for indexing services.)
- arXiv number: JHEP asks for one and prefers an arXiv posting. The paper is on Hexagon
  (version 2 of the deposit); say so in the cover letter and leave the field empty, which is
  allowed. If it is posted on arXiv first, enter the number and add `\arxivnumber{...}` before
  `\maketitle` in the source.
- Open access: fully open access under SCOAP3; no charge to the authors and nothing to select.
- Previous submission: yes, Physical Review D, under the title "Finite quantum gravity from a
  scale-invariant gas of branes", rejected; revised throughout since. Not under consideration
  elsewhere.
- Funding and competing interests: none (stated in the paper).
- Data: https://github.com/creelie/brane-cluster-gases/releases/tag/v3.2.0 (release 3.2.0).
