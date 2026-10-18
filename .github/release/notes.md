# Ghosts in Poisson brane gases: third manuscript (v3.0.0)

This release adds a third manuscript, "Ghosts in Poisson Brane Gases" by Deep Bhattacharjee and Ushashi Bhattacharya (`ghosts/`). It computes the graviton propagator averaged over a Poisson gas of branes exactly, E[1/(zF(z))] = exp(−φ(z))/z with φ(z) = ∫ z/(z+t) ν(dt), and shows that this average differs from 1/(z E F(z)), the expression used in the two earlier manuscripts.

- The averaged propagator has no positive Källén–Lehmann representation; its massive states carry total weight e^{−|ν|} − 1 < 0.
- For an intensity ν(dt) = λ(t) dt/t the averaged massive spectrum consists of ghosts alone exactly when λ ≤ 1; for continuous λ with sup λ > 1 both signs occur.
- Any average with a finite four-dimensional tadpole has states of both signs, so ultraviolet decay and positivity pull against each other.
- Frozen configurations keep alternating residues; for the scale-free gas the residues decay like a Gaussian in their index.

All results are proved by hand in the paper. `ghosts/code` runs 19 Python, 5 C and 9 Julia checks, and `ghosts/lean/BraneGhosts.lean` proves thirteen Lean 4 theorems, each on the standard axioms only. The first two manuscripts are unchanged from 2.2.0.

Assets: for each manuscript the PDF, the LaTeX source (tex.zip) and the arXiv package (tar.gz) with its code in `anc/`, and the figures of the second manuscript (figures.zip).
