# Audit of the ultraviolet-finite theory

Five scripts that re-derive results of "Cancellation of all ultraviolet
divergences of quantum gravity by a scale-invariant gas of branes" by methods
different from those of the paper and of `code/verify.py`. Each prints one
`PASS` or `FAIL` line per check and exits nonzero on any failure;
`python3 run_all.py` runs all five. They need `numpy`, `scipy`, `sympy` and
`mpmath`.

| script | what it re-derives | method | checks |
|---|---|---|---|
| `check_oneloop_matrix.py` | the one-loop pole matrix of the three quartic operators, its determinant and the tuning (Sec. "Three quartic operators") | exact rational arithmetic in sympy, from the angular averages | 30 |
| `check_powercounting.py` | the bound on the superficial degree of an L-loop subgraph, its threshold n >= 4 and both blocks of the power-counting table | exhaustive enumeration of every vertex-degree vector, no merging inequality assumed | 12 |
| `check_vacuum.py` | the zero-point energy 3 n M^4/(64 pi^2) and the cosmological constant 3 n G M^4/(8 pi) | the Mellin identity in its strip at complex s, the continuation to s = 1 by two routes that do not use Gamma(s)/s, and three smooth regulators | 14 |
| `check_positivity.py` | the free Euclidean graviton is not reflection positive | Hankel matrices and sign changes of the time-slice covariance F_q(t), with general relativity as the control | 8 |
| `check_bubble.py` | every entry of the one-loop bubble table and the energy at which the graph overtakes the tree amplitude | adaptive quadrature in polar coordinates about the graviton pole, tail integrated to infinity; two Pauli-Villars masses | 10 |

## Findings

- All 74 checks pass. No formula of the paper had to be changed.
- The bubble table: `Re A` and the ratio `s a(-s) Re A` agree with the paper
  to better than 0.5 % at every energy, `Im A = -pi^3` exactly, and the result
  is independent of the Pauli-Villars mass to nine digits once the tail of the
  real section (which falls only like the fourth inverse power of the loop
  radius for a local subtraction) is integrated to infinity; a finite cutoff
  on that radius is the one way to get this integral wrong by a few tenths of
  a percent.
- The "bound" column of that table is the leading term of the lower bound
  maximised over `0 < delta < 1/2`, the range for which the inequality is
  stated; the entries at `E/M <= 1.5` are those of this maximisation.
- Reflection positivity: for the skeleton with `n = 4, 6, 8` the smallest
  Hankel eigenvalue of `F_q` is negative at `-6 %` to `-9 %` of the largest,
  at 20 and at 30 digits and on two grids, and `F_q(t)` itself turns negative
  near `t = 12/M` for `q = M`. For general relativity the same test gives
  `e^{-qt}/(2q)` and positive semidefinite matrices to `1e-13`.
- The Mellin transform of `Ein` equals `Gamma(s)/s` in its strip to `1e-12`
  at six complex points, and the continued value at `s = 1` is `1` whether it
  is reached through the convergent integral of `E_1`, through the cutoffs
  `exp[-(w/W)^p]` with `p = 1, 2, 3`, or through the sharp cutoff: the
  zero-point energy does not depend on the regulator.
- The power counting: over every degree vector with `L <= 8` loops and up to
  `2L + 2` vertices, the sum of the vertex excesses never exceeds
  `(2L - k)_+`, and the bound is attained by the star in which one vertex
  carries every extra line.
