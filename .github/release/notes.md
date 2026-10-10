# Brane cluster gases and ghost-free form factors: corrected manuscript (v2.2.0)

This release corrects eight errors in "Brane cluster gases and ghost-free form factors for quantum gravity" (`paper/`) found in review, and attaches that manuscript's PDF, LaTeX source and arXiv package to the release for the first time. The second manuscript, "Finite quantum gravity from a scale-invariant gas of branes" (`uv-finite/`), is unchanged from 2.1.1.

- The spin-zero term of the quadratic action is −(D−2)p²b P^s, the inverse of the propagator written below it. The suite checks it against the linearized Einstein–Hilbert operator in D = 3, 4, 5 and 6.
- Lee et al. (2020) bound the range of a gravitational-strength Yukawa force by 38.6 μm; they do not report 1% agreement at 52 μm. The 14.3 μm bound is replaced by their statement and a rough estimate 1/M < 33 μm, with the caveat that a proper fit is needed.
- The terms of the product expansion are subsets of branes, which are strata only when every subset meets (caption of Fig. 1, the cluster sum, the Poisson theorem, the linked-cluster remark).
- The ghost-free corollary concerns the mean form factor of a gas and counts only branes of positive weight; the quenched paragraph and Fig. 8 leave out realizations with no such brane.
- In the composition lemma the constant depends on N and on a bound ρ < 1 for the reflection amplitudes, the reflection amplitude from the right is −r̄ t/t̄ in general, and the dressing of the two-point function is part of Hypothesis 1.
- The counterterm corollary includes the Einstein–Hilbert term.
- The text after Table I includes Einstein gravity.
- Strata are indexed by nonempty subsets, the nerve complex is not augmented, and the census counts 2^D − 1 strata besides the ambient torus.
- The abstract is cut to its main results, the experimental caveat on the Lee et al. bound moves to a footnote, and the manuscript states how AI tools were used, as the second manuscript already does.

`code/checks_clusters.py` runs 176 checks, all of which pass.

Assets: for each manuscript the PDF, the LaTeX source (tex.zip) and the arXiv package (tar.gz) with its code in `anc/`, and the figures of the second manuscript (figures.zip).
