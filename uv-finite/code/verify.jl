# Independent checks in Julia for "Finite quantum gravity from a scale-invariant gas of
# branes" (D. Bhattacharjee).
#
# Every statement checked here is derived analytically in the paper.  This file re-evaluates
# the identities, bounds and closed forms with its own special functions (power series,
# continued fractions, the arithmetic-geometric mean) and its own double-exponential
# quadrature in 256-bit arithmetic, checks the angular averages of Appendix B exactly in
# rational arithmetic, and samples the gas by Monte Carlo.  It uses only Julia's standard
# library and shares no code with verify.py or formfactor_check.c.
#
# Run:  julia verify.jl        (Julia 1.10 or later)

using Printf, Random, LinearAlgebra

setprecision(BigFloat, 256)
const B = BigFloat
const γE = B(Base.MathConstants.eulergamma)
const PIB = B(pi)

const NPASS = Ref(0)
const NFAIL = Ref(0)
function check(name, ok)
    if ok
        NPASS[] += 1; println("[PASS] ", name)
    else
        NFAIL[] += 1; println("[FAIL] ", name)
    end
end
relerr(a, b) = abs(a - b) / max(abs(b), 1e-300)

# ---------------------------------------------------------------------------------------
# Special functions
# ---------------------------------------------------------------------------------------

"Ein(w) = sum_{k>=1} (-1)^(k+1) w^k / (k k!), by its power series (real or complex)."
const TOL = eps(B) / 4
function Ein(w)
    iszero(w) && return zero(w)
    s = zero(w); term = w; k = 1
    while true
        s += (isodd(k) ? term : -term) / k
        term *= w / (k + 1)
        k += 1
        if abs(term) <= abs(s) * TOL && k > 2 * abs(w) + 5
            return s
        end
    end
end

"E1(y) for real y >= 1 by its continued fraction (modified Lentz)."
function E1cf(y::B)
    tiny = B(10)^-200
    b = y + 1; c = 1 / tiny; d = 1 / b; h = d
    for i in 1:100000
        an = -B(i)^2
        b += 2
        d = 1 / (an * d + b); c = b + an / c
        del = c * d; h *= del
        abs(del - 1) < eps(B) && break
    end
    return h * exp(-y)
end

"Ein for large real arguments through Ein = log y + gamma + E1 (used only after it is checked)."
Ein_big(y::B) = y < 40 ? Ein(y) : log(y) + γE + E1cf(y)

"Lower incomplete gamma by its series: gamma(a, x) = x^a e^{-x} sum x^k / (a (a+1) ... (a+k))."
function lowgamma(a::B, x::B)
    term = 1 / a; s = term; k = 0
    while true
        k += 1
        term *= x / (a + k)
        s += term
        term < s * eps(B) && break
    end
    return x^a * exp(-x) * s
end

"Gamma(1/4) from the arithmetic-geometric mean: Gamma(1/4)^2 = (2 pi)^{3/2} / AGM(1, sqrt 2)."
function agm(a::B, b::B)
    for _ in 1:200
        a, b = (a + b) / 2, sqrt(a * b)
        abs(a - b) < eps(B) * a && break
    end
    return a
end
const G14 = sqrt((2 * PIB)^(B(3) / 2) / agm(B(1), sqrt(B(2))))
const G34 = PIB * sqrt(B(2)) / G14                 # reflection formula
"Gamma at quarter- and half-integers by the recurrence."
function gammaq(x::Rational)
    den = denominator(x)
    base = den == 1 ? B(1) : den == 2 ? sqrt(PIB) : (x - floor(x) == 1 // 4 ? G14 : G34)
    x0 = den == 1 ? 1 // 1 : x - floor(x)
    g = base; y = x0
    while y < x
        g *= B(y); y += 1
    end
    return g
end

# ---------------------------------------------------------------------------------------
# Double-exponential quadrature in BigFloat
# ---------------------------------------------------------------------------------------

"tanh-sinh on [a, b]."
function quad(f, a::B, b::B; h = B(1) / 64, T = 6.5)
    s = zero(B)
    for k in -Int(round(T / h)):Int(round(T / h))
        t = k * h
        u = PIB * sinh(t)
        e = exp(-u)
        x = a + (b - a) / (1 + e)
        w = (b - a) * PIB * cosh(t) * e / (1 + e)^2
        (x <= a || x >= b) && continue
        s += w * f(x)
    end
    return s * h
end

"exp-sinh on [a, infinity)."
function quadinf(f, a::B; h = B(1) / 128, T1 = 6.5, T2 = 4.0)
    s = zero(B)
    for k in -Int(round(T1 / h)):Int(round(T2 / h))
        t = k * h
        e = exp(PIB / 2 * sinh(t))
        s += PIB / 2 * cosh(t) * e * f(a + e)
    end
    return s * h
end

# ---------------------------------------------------------------------------------------
println("== 1. The entire exponential integral (Sec. II E)")
# ---------------------------------------------------------------------------------------
for y in (B(1), B(2), B(7), B(20), B(35))
    lhs = Ein(y)
    rhs = log(y) + γE + E1cf(y)
    check(@sprintf("Ein(y) = log y + gamma_E + E1(y) at y = %g (continued fraction)", Float64(y)),
          relerr(lhs, rhs) < B(10)^-60)
end
for y in (B(1) / 2, B(3), B(12))
    e1 = quadinf(t -> exp(-t) / t, y)
    check(@sprintf("E1(%g) by quadrature agrees with Ein - log - gamma, and 0 < E1 <= e^-y/y", Float64(y)),
          relerr(e1, Ein(y) - log(y) - γE) < B(10)^-40 && 0 < e1 <= exp(-y) / y)
end
# g(y) = Ein(y) - log(1+y) is nondecreasing from 0 to gamma_E
let ys = B.(0:0.25:30), g = [Ein(y) - log1p(y) for y in ys]
    # g' = (1 - e^-y)/y - 1/(1+y) >= 0, and g -> gamma_E like gamma_E - 1/y
    check("g(y) = Ein(y) - log(1+y) nondecreasing on [0, 30], g(0) = 0, gamma_E - 1/30 < g(30) < gamma_E",
          all(diff(g) .>= 0) && g[1] == 0 && γE - 1 / B(30) < g[end] < γE)
end
# Ein series coefficients and the infrared expansion a = 1 + n x^2/2 + n(n-1) x^4/8 + O(x^6)
for n in (4, 7)
    x = B(1) / 1000
    a = exp(n / B(2) * Ein(x^2))
    check("infrared expansion of the skeleton, n = $n", abs(a - (1 + n * x^2 / 2 + n * (n - 1) * x^4 / 8)) < 10 * x^6 * n^3)
end

# ---------------------------------------------------------------------------------------
println("== 2. Skeleton bounds on the real axis and in the cones (Eqs. 21, 26)")
# ---------------------------------------------------------------------------------------
for n in (4, 5, 8)
    chat = exp(n * γE / 2)
    ok = true
    for x in B.(vcat(0:0.05:2, 2.5:0.5:7))
        a = exp(n / B(2) * Ein(x^2))
        ok &= (1 + x^2)^(B(n) / 2) <= a * (1 + B(10)^-60) && a <= chat * (1 + x^2)^(B(n) / 2)
    end
    check("two-sided bound (1+x^2)^(n/2) <= a <= c (1+x^2)^(n/2) on 0 <= x <= 7, n = $n", ok)
end
for n in (4, 6), δ in (0.1, 0.3), r in (1.5, 2.5, 4.0)
    ok = true
    for φ in range(-(π / 4 - δ), π / 4 - δ, length = 9)
        z = Complex{B}(B(r) * cos(B(φ)), B(r) * sin(B(φ)))
        a = exp(n / B(2) * Ein(z^2))
        ε = a / (exp(n * γE / 2) * z^n) - 1
        bound = exp(n * exp(-B(r)^2 * sin(2 * B(δ))) / (2 * B(r)^2)) - 1
        ok &= abs(ε) <= bound
    end
    check(@sprintf("cone estimate |eps(z)| <= exp(n e^{-|z|^2 sin 2d}/(2|z|^2)) - 1, n=%d, d=%.1f, |z|=%.1f", n, δ, r), ok)
end

# ---------------------------------------------------------------------------------------
println("== 3. Fluctuations of a realization for the power law (Eqs. 41-44, Table II)")
# ---------------------------------------------------------------------------------------
const θ0 = log(B(2))
for (n, ζ) in ((4, 9), (6, 13))
    for x in (B(1) / 2, B(1), B(2))
        direct = n * θ0 * quad(τ -> τ^(ζ - 1) * (1 - exp(-x^2 * τ^2))^2, B(0), B(1))
        closed = n * θ0 * (1 / B(ζ) - lowgamma(B(ζ) / 2, x^2) / x^ζ + lowgamma(B(ζ) / 2, 2x^2) / (2 * (2x^2)^(B(ζ) / 2)))
        check(@sprintf("E R(x)^2 closed form = quadrature, n=%d zeta=%d x=%g", n, ζ, Float64(x)), relerr(closed, direct) < B(10)^-40)
        direct2 = n * θ0 * quad(τ -> τ^(ζ - 1) * exp(-2 * x^2 * τ^2), B(0), B(1))
        closed2 = n * θ0 / 2 * lowgamma(B(ζ) / 2, 2x^2) / (2x^2)^(B(ζ) / 2)
        check(@sprintf("E|R(x) - R_inf|^2 closed form = quadrature, n=%d zeta=%d x=%g", n, ζ, Float64(x)), relerr(closed2, direct2) < B(10)^-40)
    end
end
# E e^{R_inf} = exp[(n/zeta) int_0^theta0 (e^t - 1 - t)/t^2 dt]; the integral as a series
EeR(n, ζ) = exp(n / B(ζ) * sum(θ0^(k - 1) / (factorial(big(k)) * (k - 1)) for k in 2:80))
let I = quad(t -> t < B(10)^-25 ? 1 / B(2) + t / 6 : (expm1(t) - t) / t^2, B(0), θ0), S = sum(θ0^(k - 1) / (factorial(big(k)) * (k - 1)) for k in 2:80)
    check("int_0^theta0 (e^t-1-t)/t^2 dt: series = quadrature", relerr(S, I) < B(10)^-50)
end
table = [(4, 9, 0.555, 1.190, 0.257, 4.47, 0.372), (5, 11, 0.561, 1.195, 0.253, 3.74, 0.281),
         (6, 13, 0.566, 1.198, 0.250, 3.35, 0.227), (7, 15, 0.569, 1.201, 0.249, 3.11, 0.190),
         (8, 17, 0.571, 1.202, 0.247, 2.97, 0.163)]
for (n, ζ, sd, ee, lb, xs, rel) in table
    v1 = n * θ0 / ζ
    c_sd = sqrt(v1)
    c_ee = EeR(n, ζ)
    c_lb = exp(-sqrt(2 * v1 * log(B(20))))
    rms(x) = sqrt(n * θ0 / 2 * lowgamma(B(ζ) / 2, 2x^2) / (2x^2)^(B(ζ) / 2))
    lo, hi = B(1), B(30)
    for _ in 1:200
        mid = (lo + hi) / 2
        rms(mid) > B(10)^-3 ? (lo = mid) : (hi = mid)
    end
    c_rel = 2 * sqrt(θ0 / (n * (ζ - 4)))
    ok = all(abs.(Float64.([c_sd, c_ee, c_lb, lo, c_rel]) .- [sd, ee, lb, xs, rel]) .<= [5e-4, 5e-4, 5e-4, 5e-3, 5e-4])
    check(@sprintf("Table II row n=%d: %.3f %.3f %.3f %.2f %.3f", n, c_sd, c_ee, c_lb, lo, c_rel), ok)
end
# identical branes: Psi_*(x) = Ein(x^2) - Ein(2x^2)/2 = int_0^1 psi_*(x tau)^2 dtau/tau
for x in (B(1) / 2, B(2), B(4))
    direct = quad(τ -> (1 - exp(-x^2 * τ^2))^2 / τ, B(0), B(1))
    check(@sprintf("Psi_*(%g) = Ein(x^2) - Ein(2x^2)/2", Float64(x)), relerr(Ein(x^2) - Ein(2x^2) / 2, direct) < B(10)^-40)
end
let x = B(6)
    check("Psi_*(x) - log x - (gamma_E - log 2)/2 = O(e^{-x^2}) at x = 6",
          abs(Ein(x^2) - Ein(2x^2) / 2 - log(x) - (γE - log(B(2))) / 2) < 2 * exp(-x^2))
end

# ---------------------------------------------------------------------------------------
println("== 4. Monte Carlo of the leading coefficient (Eqs. 31, 44, 45)")
# ---------------------------------------------------------------------------------------
"Poisson variate of mean lam: sum of Knuth variates of mean at most 20."
function poisson(rng, lam)
    k = 0
    while lam > 0
        l = min(lam, 20.0); lam -= l
        L = exp(-l); p = 1.0
        while true
            p *= rand(rng)
            p <= L && break
            k += 1
        end
    end
    return k
end
let rng = MersenneTwister(20261007), n = 4, ζ = 9, t0 = log(2.0), ε = 1e-4, N = 40000
    # strengths t = theta0 tau^zeta have intensity (n/zeta) dt/t^2 on (0, t0]
    c = n / ζ
    Λ = c * (1 / ε - 1 / t0)
    comp = c * log(t0 / ε)
    R = Vector{Float64}(undef, N)
    for i in 1:N
        m = poisson(rng, Λ)
        s = 0.0
        for _ in 1:m
            s += 1 / (1 / ε - rand(rng) * (1 / ε - 1 / t0))
        end
        R[i] = s - comp + sqrt(c * ε) * randn(rng)        # strengths below eps: Gaussian limit
    end
    v = sum(abs2, R .- sum(R) / N) / (N - 1)
    vth = n * t0 / ζ
    check(@sprintf("Var R_inf: Monte Carlo %.4f vs n theta0/zeta = %.4f (N = %d)", v, vth, N), abs(v - vth) < 4 * vth * sqrt(2 / N) * 2)
    m = sum(exp.(R)) / N
    mth = Float64(EeR(n, ζ))
    se = sqrt(sum(abs2, exp.(R) .- m) / (N - 1) / N)
    check(@sprintf("E e^{R_inf}: Monte Carlo %.4f vs closed form %.4f", m, mth), abs(m - mth) < 4 * se)
    check(@sprintf("mean of R_inf: Monte Carlo %.4f vs 0", sum(R) / N), abs(sum(R) / N) < 4 * sqrt(vth / N))
    y = 1.0
    check("lower tail P(R_inf < -1) below exp(-1/(2 n V(1)))", count(<(-y), R) / N <= exp(-y^2 / (2 * vth)))
end

# ---------------------------------------------------------------------------------------
println("== 5. Vacuum energy (Eqs. 29, 30, 77-80)")
# ---------------------------------------------------------------------------------------
let μ = quadinf(t -> t * exp(-t^2), B(0))
    check("mu_psi* = int_0^inf t e^{-t^2} dt = 1/2", abs(μ - B(1) / 2) < B(10)^-50)
end
for τ in (B(1), B(1) / 3)
    # int d^4k/(2pi)^4 [psi(tau k^2) - 1] = -mu/(16 pi^2 tau^2), radial measure k^3 dk/(8 pi^2)
    lhs = quadinf(k -> k^3 / (8 * PIB^2) * (-exp(-τ^2 * k^4)), B(0))
    check(@sprintf("single-brane moment, Eq. (30), tau = %.3f", Float64(τ)), relerr(lhs, -(B(1) / 2) / (16 * PIB^2 * τ^2)) < B(10)^-40)
end
for (n, ζ) in ((4, 9), (6, 13))
    # relative spread: (2/n) sqrt(n theta0 int tau^{zeta-5} dtau) = 2 sqrt(theta0/(n (zeta-4)))
    var = n * θ0 * quad(τ -> τ^(ζ - 5), B(0), B(1))
    check("relative spread of the zero-point energy, n = $n, zeta = $ζ",
          relerr(2 / B(n) * sqrt(var), 2 * sqrt(θ0 / (n * (ζ - 4)))) < B(10)^-40)
end

# ---------------------------------------------------------------------------------------
println("== 6. Newtonian potential: depth of the well (Eqs. 86-90)")
# ---------------------------------------------------------------------------------------
let g = BigFloat()       # MPFR's own Gamma, an independent implementation
    ccall((:mpfr_gamma, Base.MPFR.libmpfr), Int32, (Ref{BigFloat}, Ref{BigFloat}, Base.MPFR.MPFRRoundingMode),
          g, B(1) / 4, Base.MPFR.ROUNDING_MODE[])
    check("Gamma(1/4) from the AGM agrees with MPFR's Gamma to 70 digits", abs(G14 - g) < B(10)^-70)
end
for n in 4:8
    J = gammaq(1 // 4) * gammaq(n // 2 - 1 // 4) / gammaq(n // 2) / 4
    Jq = quad(t -> t^(-B(3) / 4) / (1 + t)^(B(n) / 2), B(0), B(1)) / 4 + quadinf(t -> t^(-B(3) / 4) / (1 + t)^(B(n) / 2), B(1)) / 4
    check("J_$n: Beta-function closed form = quadrature", relerr(J, Jq) < B(10)^-30)
    if iseven(n)
        m = n ÷ 2
        check("J_$n = (pi sqrt 2 / 4) prod_{j<m} (1 - 1/(4j))", relerr(J, PIB * sqrt(B(2)) / 4 * prod(B(1) - 1 / (4 * B(j)) for j in 1:m-1; init = B(1))) < B(10)^-60)
    end
    K = G14 / 4 * (2 / B(n))^(B(1) / 4)
    check("J_$n / K_$n <= 2n/(2n-1) (Wendel)", J / K <= B(2n) / (2n - 1))
end
for (n, lo, val, hi) in ((4, 0.4852, 0.5100, 0.5303), (6, 0.4385, 0.4519, 0.4640))
    I = quadinf(k -> exp(-n / B(2) * Ein_big(k^4)), B(0))
    J = gammaq(1 // 4) * gammaq(n // 2 - 1 // 4) / gammaq(n // 2) / 4
    K = G14 / 4 * (2 / B(n))^(B(1) / 4)
    f(v) = Float64(2 / PIB * v)
    check(@sprintf("|Phi_N(0)|/(G m M) for n=%d: %.4f <= %.4f <= %.4f", n, f(K), f(I), f(J)),
          abs(f(K) - lo) < 5e-5 && abs(f(I) - val) < 5e-5 && abs(f(J) - hi) < 5e-5 && K <= I <= J)
end

# ---------------------------------------------------------------------------------------
println("== 7. Angular averages and the one-loop pole matrix, exactly (App. B, Eqs. 71-74)")
# ---------------------------------------------------------------------------------------
# The 24 vertices of the 24-cell, (+-1, +-1, 0, 0) and permutations over sqrt 2, form a
# spherical 5-design on S^3, so the mean over them of any polynomial of degree <= 5 in the
# unit vector is its average over the sphere.  Every quantity below is even in khat and
# depends on it through omega_ab = khat_a khat_b = v_a v_b / 2, which is rational.
const Q = Rational{BigInt}
const VERTS = let vs = Vector{Vector{Int}}()
    for i in 1:4, j in i+1:4, si in (-1, 1), sj in (-1, 1)
        v = zeros(Int, 4); v[i] = si; v[j] = sj; push!(vs, v)
    end
    vs
end
check("24-cell vertex count", length(VERTS) == 24)
δ(a, b) = a == b ? Q(1) : Q(0)
ωof(v) = [Q(v[a] * v[b], 2) for a in 1:4, b in 1:4]
avg(f) = sum(f(ωof(v)) for v in VERTS) / 24
check("<khat_a khat_b> = delta_ab / 4 on the 24-cell",
      all(avg(ω -> ω[a, b]) == δ(a, b) / 4 for a in 1:4, b in 1:4))
check("<khat_a khat_b khat_c khat_d> = (dd + dd + dd)/24 on the 24-cell",
      all(avg(ω -> ω[a, b] * ω[c, d]) == (δ(a, b) * δ(c, d) + δ(a, c) * δ(b, d) + δ(a, d) * δ(b, c)) / 24
          for a in 1:4, b in 1:4, c in 1:4, d in 1:4))

"Kulkarni-Nomizu product h o k: an algebraic curvature tensor."
kn(h, k) = [h[a, c] * k[b, d] + h[b, d] * k[a, c] - h[a, d] * k[b, c] - h[b, c] * k[a, d] for a in 1:4, b in 1:4, c in 1:4, d in 1:4]
function randsym(rng)
    m = Q.(rand(rng, -5:5, 4, 4)); (m + transpose(m))
end
rng = MersenneTwister(7)
ok1 = true; ok2 = true; ok3 = true
for trial in 1:6
    Rm = kn(randsym(rng), randsym(rng)) + kn(randsym(rng), randsym(rng)) + kn(randsym(rng), randsym(rng))
    global ok3 &= all(Rm[a, b, c, d] == -Rm[b, a, c, d] && Rm[a, b, c, d] == Rm[c, d, a, b] &&
                      Rm[a, b, c, d] + Rm[a, c, d, b] + Rm[a, d, b, c] == 0 for a in 1:4, b in 1:4, c in 1:4, d in 1:4)
    Ric = [sum(Rm[m, b, m, d] for m in 1:4) for b in 1:4, d in 1:4]
    Rs = sum(Ric[a, a] for a in 1:4)
    Ric2 = sum(Ric .^ 2); Riem2 = sum(Rm .^ 2)
    # Eq. (71): Rbar^{mn} Rbar^{rs} <Pt - (3/2) T x T>_{mnrs} = (Ric^2 - R^2)/2
    lhs1 = avg(function (ω)
        θ = [δ(a, b) - ω[a, b] for a in 1:4, b in 1:4]
        T = θ / 3 + ω
        Pt = sum(Ric[m, n] * Ric[r, s] * ((θ[m, r] * θ[n, s] + θ[m, s] * θ[n, r]) / 2 - θ[m, n] * θ[r, s] / 3)
                 for m in 1:4, n in 1:4, r in 1:4, s in 1:4)
        TT = sum(Ric[m, n] * T[m, n] for m in 1:4, n in 1:4)^2
        Pt - Q(3, 2) * TT
    end)
    global ok1 &= lhs1 == (Ric2 - Rs^2) / 2
    # Eq. (72): <tr W^2 - (tr W)^2/2> = (3 Riem^2 - R^2)/48, W^{ns} = R^{mnrs} khat_m khat_r
    lhs2 = avg(function (ω)
        W = [sum(Rm[m, n, r, s] * ω[m, r] for m in 1:4, r in 1:4) for n in 1:4, s in 1:4]
        sum(W .* transpose(W)) - (sum(W[a, a] for a in 1:4))^2 / 2
    end)
    global ok2 &= lhs2 == (3 * Riem2 - Rs^2) / 48
end
check("random curvature tensors (Kulkarni-Nomizu sums) have the symmetries of Riemann", ok3)
check("Eq. (71) exactly, six random curvature tensors", ok1)
check("Eq. (72) exactly, six random curvature tensors", ok2)
# fixed khat = e4: T.T = D/(D-1) = 4/3, T.delta = 2, theta P0 theta = D - 1 = 3
let ω = [δ(a, 4) * δ(b, 4) for a in 1:4, b in 1:4], θ = [δ(a, b) for a in 1:4, b in 1:4] - ω, T = θ / 3 + ω
    check("T.T = 4/3, T.delta = 2, theta.theta = 3 in D = 4",
          sum(T .* T) == Q(4, 3) && sum(T[a, a] for a in 1:4) == 2 && sum(θ .* θ) == 3)
end
# the three pole lines and the matrix of Eq. (73)
c_s1 = 8 * Q(-3, 2)                      # R^2
c_s2 = 8 * Q(1, 4) * Q(1, 2)             # (Ric^2 - R^2)
c_s3R = 8 * 4 * Q(3, 48); c_s3S = 8 * 4 * Q(-1, 48)   # Riem^2 and R^2
check("pole coefficients -12, 1, (2, -2/3)", c_s1 == -12 && c_s2 == 1 && c_s3R == 2 && c_s3S == Q(-2, 3))
# Riem^2 = E + 4 Ric^2 - R^2
A = Q[c_s1 -c_s2 (c_s3S - c_s3R); 0 c_s2 4*c_s3R; 0 0 c_s3R]
check("matrix of Eq. (73) is [[-12,-1,-8/3],[0,1,8],[0,0,2]]", A == Q[-12 -1 Q(-8, 3); 0 1 8; 0 0 2])
check("det = -24 and upper-left block det = -12", det(A) == -24 && det(A[1:2, 1:2]) == -12)
let b = Q[Q(7, 5), Q(-3, 11), Q(13, 17)]          # arbitrary beta^(0)
    t = [ (b[1] + b[2] - Q(8, 3) * b[3]) / 12, -(b[2] - 4 * b[3]), -b[3] / 2 ]
    check("the tuning of Eq. (74) cancels the three poles", A * t + b == zeros(Q, 3))
end

# ---------------------------------------------------------------------------------------
println("== 8. Power counting (Sec. IV D)")
# ---------------------------------------------------------------------------------------
ω̄(n, L) = 4L - (2n + 2) * (L - 1) + 2 * max(L - n - 1, 0)
check("omegaBar(L) < 0 for all 2 <= L <= 200 exactly when n >= 4 (n = 1..30)",
      all((all(ω̄(n, L) < 0 for L in 2:200)) == (n >= 4) for n in 1:30))
check("Table V", [ω̄(n, L) for n in 3:8, L in 1:5] == [4 0 -4 -8 -10; 4 -2 -8 -14 -20; 4 -4 -12 -20 -28;
                                                       4 -6 -16 -26 -36; 4 -8 -20 -32 -44; 4 -10 -24 -38 -52])
check("merging inequality, exhaustive for 2 <= a, b, k <= 40",
      all(max(a - k, 0) + max(b - k, 0) <= max(a + b - 2 - k, 0) for a in 2:40, b in 2:40, k in 2:40))
let rng = MersenneTwister(11), worst = 0.0
    for r in 1:4, _ in 1:50000
        q = randn(rng, 4) .* 10.0^(4 * rand(rng) - 2)
        ls = [randn(rng, 4) .* 10.0^(4 * rand(rng) - 2) for _ in 1:r]
        rand(rng) < 0.2 && (ls[end] = -q - sum(ls[1:end-1]; init = zeros(4)))      # q' = 0
        s = sum(norm, ls); qp = q + sum(ls)
        lhs = (norm(q) + s)^2
        rhs = (1 + 2r + 2r^2) * (1 + norm(q)) * (1 + norm(qp)) * prod(1 + norm(l) for l in ls)
        worst = max(worst, lhs / rhs)
    end
    check(@sprintf("insertion lemma (|q|+s)^2 <= C_r (1+|q|)(1+|q'|) prod(1+|l|), r <= 4, worst ratio %.3f", worst), worst <= 1)
end

# ---------------------------------------------------------------------------------------
println("== 9. Gravitational scattering of matter at tree level (Sec. VII C, Eq. 82)")
# ---------------------------------------------------------------------------------------
# Two distinct massless scalars in the center-of-mass frame, metric (+,-,-,-).
# T^{mn} = p^m p'^n + p'^m p^n - eta^{mn} p.p' for each scalar line.
let η = Diagonal(B[1, -1, -1, -1]), rng = MersenneTwister(5), ok = true, okt = true
    dot4(a, b) = a' * η * b
    for _ in 1:20
        E = B(rand(rng)) * 10 + 1; θ = B(rand(rng)) * PIB; φ = B(rand(rng)) * 2PIB
        p1 = E * B[1, 0, 0, 1]; p2 = E * B[1, 0, 0, -1]
        n̂ = B[sin(θ) * cos(φ), sin(θ) * sin(φ), cos(θ)]
        p3 = E * vcat(B(1), n̂); p4 = E * vcat(B(1), -n̂)
        T(p, q) = p * q' + q * p' - inv(η) * dot4(p, q)          # upper indices
        T1 = T(p1, p3); T2 = T(p2, p4)
        T1T2 = sum((η * T1 * η) .* T2)                             # T1_{mn} T2^{mn}
        tr1 = sum(η .* T1); tr2 = sum(η .* T2)
        s = dot4(p1 + p2, p1 + p2); t = dot4(p1 - p3, p1 - p3); u = dot4(p1 - p4, p1 - p4)
        ok &= abs(T1T2 - tr1 * tr2 / 2 + s * u) < B(10)^-60 * s^2
        okt &= abs(-t - s * sin(θ / 2)^2) < B(10)^-60 * s && abs(-u - s * cos(θ / 2)^2) < B(10)^-60 * s
    end
    check("T1.T2 - T1 T2/2 = -s u for two distinct massless scalars (20 random configurations)", ok)
    check("-t = s sin^2(theta/2), -u = s cos^2(theta/2)", okt)
end
for n in 4:8
    kn = B(n - 1)^(B(n - 1) / 2) / B(n)^(B(n) / 2)
    lo, hi = B(0), B(1)
    f(x) = x / (1 + x^2)^(B(n) / 2)
    for _ in 1:300            # golden-section search for the maximum
        m1 = lo + (hi - lo) * (3 - sqrt(B(5))) / 2; m2 = lo + (hi - lo) * (sqrt(B(5)) - 1) / 2
        f(m1) < f(m2) ? (lo = m1) : (hi = m2)
    end
    check(@sprintf("max_x x/(1+x^2)^(n/2) = (n-1)^((n-1)/2)/n^(n/2) = %.4f at x = (n-1)^(-1/2), n = %d", Float64(kn), n),
          abs(f(lo) - kn) < B(10)^-50 && abs(lo - 1 / sqrt(B(n - 1))) < B(10)^-20 && (n != 4 || abs(Float64(kn) - 0.325) < 5e-4))
end
let ok = true
    for n in (4, 6), θ in (B(1) / 2, B(1), B(2), B(3)), x in (B(1) / 10, B(1) / 2, B(1), B(3), B(10), B(50))
        # skeleton: |A| = 8 pi G s cot^2(theta/2) / ahat(x),  s = x M^2 / sin^2(theta/2); units G = M = 1
        s = x / sin(θ / 2)^2
        A = 8 * PIB * s * cot(θ / 2)^2 / exp(n / B(2) * Ein_big(x^2))
        bound = 8 * PIB * cos(θ / 2)^2 / sin(θ / 2)^4 * x / (1 + x^2)^(B(n) / 2)
        ok &= A <= bound * (1 + B(10)^-60)
    end
    check("skeleton matter amplitude below the bound of Sec. VII C (C = 1) on a grid of angles and momenta", ok)
end

# ---------------------------------------------------------------------------------------
println("== 10. The free Euclidean graviton: curvature at a point (Sec. VII D, Eq. 83)")
# ---------------------------------------------------------------------------------------
# Linearized Riemann tensor of a Fourier mode h e^{ipx} (Euclidean, overall factors drop out
# of C.C / (h P2 h)): R_{abcd} = (p_b p_c h_ad + p_a p_d h_bc - p_a p_c h_bd - p_b p_d h_ac)/2.
let rng = MersenneTwister(13), ok = true, okP = true
    for _ in 1:20
        p = randn(rng, 4); p2 = sum(abs2, p)
        h = randn(rng, 4, 4); h = (h + h') / 2
        Rm = [(p[b] * p[c] * h[a, d] + p[a] * p[d] * h[b, c] - p[a] * p[c] * h[b, d] - p[b] * p[d] * h[a, c]) / 2
              for a in 1:4, b in 1:4, c in 1:4, d in 1:4]
        Ric = [sum(Rm[m, a, m, b] for m in 1:4) for a in 1:4, b in 1:4]
        Rs = sum(Ric[a, a] for a in 1:4)
        δ(a, b) = a == b ? 1.0 : 0.0
        g(a, b, c, d) = δ(a, c) * δ(b, d) - δ(a, d) * δ(b, c)
        C = [Rm[a, b, c, d] - (δ(a, c) * Ric[b, d] - δ(a, d) * Ric[b, c] - δ(b, c) * Ric[a, d] + δ(b, d) * Ric[a, c]) / 2 +
             Rs * g(a, b, c, d) / 6 for a in 1:4, b in 1:4, c in 1:4, d in 1:4]
        θ = [δ(a, b) - p[a] * p[b] / p2 for a in 1:4, b in 1:4]
        P2(a, b, c, d) = (θ[a, c] * θ[b, d] + θ[a, d] * θ[b, c]) / 2 - θ[a, b] * θ[c, d] / 3
        hP2h = sum(h[a, b] * P2(a, b, c, d) * h[c, d] for a in 1:4, b in 1:4, c in 1:4, d in 1:4)
        ok &= abs(sum(C .^ 2) - p2^2 / 2 * hP2h) < 1e-10 * (1 + p2^2 * sum(abs2, h))
        okP &= abs(sum(P2(a, b, a, b) for a in 1:4, b in 1:4) - 5) < 1e-12
    end
    check("Weyl tensor of a Fourier mode: C.C = (p^4/2) h P2 h (20 random modes)", ok)
    check("tr P2 = 5 in four dimensions", okP)
end
for n in 4:8
    closed = sqrt(PIB) / 4 * gammaq((n - 3) // 2) / gammaq(n // 2)
    direct = quad(x -> x^2 / (1 + x^2)^(B(n) / 2), B(0), B(1)) + quadinf(x -> x^2 / (1 + x^2)^(B(n) / 2), B(1); T2 = 5.5)   # slow x^(2-n) tail
    check("int_0^inf x^2 (1+x^2)^(-n/2) dx = (sqrt(pi)/4) Gamma((n-3)/2)/Gamma(n/2), n = $n", relerr(closed, direct) < B(10)^-30)
end
let n = 4, chat = exp(4 * γE / 2)
    I = quad(x -> x^2 * exp(-n / B(2) * Ein_big(x^2)), B(0), B(1)) + quadinf(x -> x^2 * exp(-n / B(2) * Ein_big(x^2)), B(1))
    check(@sprintf("skeleton, n = 4: int x^2/ahat = %.4f, between pi/(4 chat) = %.4f and pi/4", Float64(I), Float64(PIB / (4 * chat))),
          PIB / (4 * chat) <= I <= PIB / 4 && abs(Float64(I) - 0.4237) < 5e-5)
    check(@sprintf("rms Weyl curvature sqrt(5 I/pi) = %.3f M^3/M_P", Float64(sqrt(5 * I / PIB))), abs(Float64(sqrt(5 * I / PIB)) - 0.82) < 5e-3)
end
let # n = 3: the partial integrals grow like log X, so the variance diverges logarithmically
    part(X) = quad(x -> x^2 / (1 + x^2)^(B(3) / 2), B(0), X; T = 7.0)
    d1 = part(B(10)^4) - part(B(10)^3); d2 = part(B(10)^5) - part(B(10)^4)
    check("n = 3: int_0^X x^2/(1+x^2)^(3/2) grows like log X (increments per decade -> log 10)",
          abs(d1 - log(B(10))) < 1e-5 && abs(d2 - log(B(10))) < 1e-7)
end

@printf("\n%d checks passed, %d failed\n", NPASS[], NFAIL[])
exit(NFAIL[] == 0 ? 0 : 1)
