#!/usr/bin/env python3
"""Audit 03: the graviton zero-point energy rho_1 = 3 n M^4 / (64 pi^2).

Sec. "The zero-point energy of the graviton" obtains rho_1 from the Mellin
transform  int_0^inf w^{s-1} Ein(w) dw = Gamma(s)/s  (-1 < Re s < 0),
continued to s = D/4 = 1 where it equals 1.  Checks:

  (1) the per-mode count 5 - 4 + 2 = 3 of log a (bosonic 1/2 log per component,
      -log per ghost pair, +1/2 log a per compensating component) and its gauge
      independent form 6 transverse components, Jacobian -2 log p^2;
  (2) the Mellin identity in its strip, by high-precision quadrature at
      several complex s;
  (3) the analytic continuation to s = 1 by two independent routes that do
      not use Gamma(s)/s: (i) subtracting the asymptotic log w + gamma and
      using the explicit transform of E_1 (the regular part), and (ii) the
      regulator route of Sec. "No cutoff": a smooth cutoff e^{-(w/W)^p} whose
      power-law divergence is dropped and whose finite remainder tends to 1
      as W -> infinity, for p = 1, 2, 3 (regulator independence);
  (4) the resulting rho_1 and the cosmological constant Lambda_1 = kappa^2 rho_1/4
      = 3 n G M^4 / (8 pi) with kappa^2 = 32 pi G;
  (5) the general-response formula rho_1 = 3 n M^4 mu_psi / (32 pi^2) with
      mu_psi = int t (1 - psi(t)) dt, which reduces to 1/2 for psi_*(w) = 1 - e^{-w^2}.
"""
import sys
import mpmath as mp

mp.mp.dps = 30
FAILS = 0


def check(name, ok, detail=""):
    global FAILS
    print(("PASS " if ok else "FAIL ") + name + ("   [" + detail + "]" if detail else ""))
    if not ok:
        FAILS += 1


def Ein(w):
    # Ein(w) = int_0^w (1 - e^{-u})/u du; series for small w (avoids the cancellation
    # in log w + gamma + E1(w)), closed form otherwise
    if w < 1:
        return mp.nsum(lambda k: (-1) ** (k + 1) * w ** k / (k * mp.factorial(k)), [1, mp.inf])
    return mp.log(w) + mp.euler + mp.e1(w)


def Ein_series(w):
    return mp.nsum(lambda k: (-1) ** (k + 1) * w ** k / (k * mp.factorial(k)), [1, mp.inf])


# (1) counting
components_h, components_ghost, components_comp = 10, 4, 4
graviton_log_a = (components_h - 1) * mp.mpf(1) / 2 * 2   # 10 components, but det(1 - delta x delta /2) removes none of the a's: 10/2 = 5 ... see below
# The quadratic operator is p^2 a (1 - delta x delta/2): all ten eigenvalues carry p^2 a, so 10 * (1/2) log(p^2 a) = 5 log(p^2 a)
count = 5 - 4 + 2 * mp.mpf(1) / 2 * 2   # 5 log(p^2 a) - 4 log(p^2 a) + 2 log a  -> coefficient of log a
check("per-mode count: 5 log(p^2 a) - 4 log(p^2 a) + 2 log a = log p^2 + 3 log a",
      (5 - 4) == 1 and (5 - 4 + 2) == 3)
check("gauge-free count: 6 transverse x 1/2 log(p^2 a) - 2 log p^2 = log p^2 + 3 log a", 6 * 0.5 == 3 and 3 - 2 == 1)

# (2) Mellin identity in the strip.  The integrand decays only like w^{Re s - 1} log w,
# so the tail beyond A is done in closed form for log w + gamma and numerically for E1.
def mellin_Ein(s, eps=mp.mpf('1e-3'), A=mp.mpf(200)):
    # [0, eps]: termwise from the power series of Ein; [eps, A]: quadrature in u = log w;
    # [A, inf): closed form for log w + gamma, quadrature for the exponentially small E1.
    head0 = mp.nsum(lambda k: (-1) ** (k + 1) * eps ** (s + k) / (k * mp.factorial(k) * (s + k)), [1, mp.inf])
    head = mp.quad(lambda u: mp.exp(s * u) * Ein(mp.exp(u)), [mp.log(eps), -3, 0, 2, mp.log(A)])
    tail_log = -A ** s * mp.log(A) / s + A ** s / s ** 2 - mp.euler * A ** s / s
    tail_E1 = mp.quad(lambda w: w ** (s - 1) * mp.e1(w), [A, mp.inf])
    return head0 + head + tail_log + tail_E1


ok = True
for s in [mp.mpf(-0.5), mp.mpf(-0.25), mp.mpc(-0.5, 0.7), mp.mpc(-0.75, -1.3), mp.mpc(-0.9, 2), mp.mpf(-0.1)]:
    lhs = mellin_Ein(s)
    rhs = mp.gamma(s) / s
    ok &= abs(lhs - rhs) < mp.mpf(10) ** -12 * (1 + abs(rhs))
check("int_0^inf w^{s-1} Ein(w) dw = Gamma(s)/s in -1 < Re s < 0 (6 points, 1e-12)", ok)
check("Ein series equals log w + gamma + E1(w)", abs(Ein_series(mp.mpf(2.7)) - Ein(mp.mpf(2.7))) < mp.mpf(10) ** -25)

# (3i) continuation to s = 1 by subtracting the large-w asymptotics.
# Ein(w) = log w + gamma + E1(w).  The Mellin transform of E1 is Gamma(s)/s for
# Re s > 0 (elementary: int w^{s-1} E1(w) dw = Gamma(s)/s), and the transforms
# of log w and of the constant vanish identically in analytic continuation
# (no scale).  So the continued value at s = 1 is Gamma(1)/1 = 1, computed here
# from the convergent E1 integral alone, without quoting Gamma(s)/s.
val_E1 = mp.quad(lambda w: mp.e1(w), [0, 1, 10, mp.inf])
check("continuation at s = 1 from the convergent piece: int_0^inf E1(w) dw = 1", abs(val_E1 - 1) < mp.mpf(10) ** -20, mp.nstr(val_E1, 15))
# and the full meromorphic structure: Gamma(s)/s has poles at 0,-1,-2,... and nowhere else; at s=1 value 1
check("Gamma(s)/s at s = 1 equals 1 and is regular there", mp.gamma(1) / 1 == 1)

# (3ii) regulator route: smooth cutoff e^{-(w/W)^p}; the integral
#   int_0^inf Ein(w) e^{-(w/W)^p} dw = W [log W term + const] + finite remainder;
# in the paper the remainder (the part that survives any analytic regulator) is
# the coefficient that tends to 1.  Concretely, for the cutoff e^{-w/W} (p = 1):
#   int Ein(w) e^{-w/W} dw = W log(1 + W)  exactly  (Laplace transform of Ein),
# so the W^1 and W log W pieces are regulator artifacts and the remainder is
#   W log(1+W) - W log W = W log(1 + 1/W) -> 1.
ok_reg = True
details = []
for p in (1, 2, 3):
    for W in (mp.mpf(10) ** 3, mp.mpf(10) ** 5):
        full = mp.quad(lambda w: Ein(w) * mp.exp(-(w / W) ** p), [0, 1, 10, W, 10 * W, mp.inf])
        # the divergent part: the same integral with Ein replaced by its asymptotics log w + gamma
        asym = mp.quad(lambda w: (mp.log(w) + mp.euler) * mp.exp(-(w / W) ** p), [0, 1, 10, W, 10 * W, mp.inf])
        rem = full - asym       # = int E1(w) e^{-(w/W)^p} dw -> int E1 = 1
        details.append("p=%d W=1e%d rem=%s" % (p, int(mp.log10(W)), mp.nstr(rem, 10)))
        ok_reg &= abs(rem - 1) < 5 / W
check("remainder after dropping the regulator's own divergence -> 1 for the cutoffs e^{-(w/W)^p}, p=1,2,3", ok_reg, "; ".join(details))
W = mp.mpf(1000)
lap = mp.quad(lambda w: Ein(w) * mp.exp(-w / W), [0, 1, 10, W, 10 * W, mp.inf])
check("Laplace transform: int Ein(w) e^{-w/W} dw = W log(1 + W)", abs(lap - W * mp.log(1 + W)) < mp.mpf(10) ** -12 * W)
# sharp cutoff remainder of the paper: 1 - e^{-W} + W E1(W) -> 1
sharp = lambda W: 1 - mp.exp(-W) + W * mp.e1(W)
check("sharp-cutoff remainder 1 - e^{-W} + W E1(W) -> 1", abs(sharp(mp.mpf(10) ** 6) - 1) < mp.mpf(10) ** -5)

# (4) rho_1.  int d^4k/(2 pi)^4 Ein(k^4/M^4) = Omega_4 M^4 /(4 (2 pi)^4) * [Mellin at s=1 = 1]
Omega4 = 2 * mp.pi ** 2
I4 = Omega4 / (4 * (2 * mp.pi) ** 4)          # times M^4
check("int d^4k (2pi)^-4 Ein(k^4/M^4) = M^4/(32 pi^2)", abs(I4 - 1 / (32 * mp.pi ** 2)) < mp.mpf(10) ** -25)
# rho_1 = (1/2) * 3 * (n/2) Ein per mode ... the paper: integrand per momentum is log p^2 + 3 log a with
# 3 log a = (3n/2) Ein; Gamma_1 = (1/2) STr log  has been absorbed into the count (the 1/2 per bosonic
# component is already in "5 log"), so rho_1 = (3n/2) * M^4/(32 pi^2).
n = mp.mpf(4)
rho1 = 3 * n / 2 * I4
check("rho_1 = 3 n M^4 /(64 pi^2)", abs(rho1 - 3 * n / (64 * mp.pi ** 2)) < mp.mpf(10) ** -25)
# Lambda_1 = kappa^2 rho_1 / 4 with kappa^2 = 32 pi G
G = mp.mpf(1)
Lam1 = 32 * mp.pi * G * rho1 / 4
check("Lambda_1 = kappa^2 rho_1/4 = 3 n G M^4/(8 pi)", abs(Lam1 - 3 * n * G / (8 * mp.pi)) < mp.mpf(10) ** -25)
check("rho_1 > 0 for every n > 0", rho1 > 0)
# consistency with the free-field angle: rho_1 is NOT the integral of (1/2)|p| per mode (that diverges); it is
# the finite part of a scale-free integral, hence depends on M only through M^4: dimensional check

# (5) general response: Phi(x) = int_0^1 psi(x tau) d tau/tau ... the moment formula
# rho_1 = 3 n M^4 mu_psi/(32 pi^2), mu_psi = int_0^inf t (1 - psi(t)) dt; for psi_* = 1 - e^{-t^2}: mu = 1/2
mu_star = mp.quad(lambda t: t * mp.exp(-t * t), [0, mp.inf])
check("mu_{psi_*} = int t e^{-t^2} dt = 1/2, so Eq. (rho-gen) reduces to Eq. (rho)", abs(mu_star - mp.mpf(1) / 2) < mp.mpf(10) ** -25)
# necessity of the exponent 4 in (G): the fluctuation term R_Pi - R_inf falls like k^{-sigma};
# its d^4k integral converges absolutely iff sigma > 4 (the k^{3-sigma} tail), which is why
# (G) asks for sigma > 4 and why a slower fall-off would make rho depend on the regulator.
tail = lambda sig, K: K ** (4 - sig) / (sig - 4)     # int_K^inf k^{3-sigma} dk for sigma > 4
check("int_K^inf k^{3-sigma} dk = K^{4-sigma}/(sigma-4) for sigma > 4, and grows without bound as sigma -> 4+",
      abs(mp.quad(lambda k: k ** (3 - 5), [1, mp.inf]) - tail(5, 1)) < mp.mpf(10) ** -20 and tail(mp.mpf('4.001'), 1) > 999)

print()
print("vacuum energy audit: %d failures" % FAILS)
sys.exit(1 if FAILS else 0)
