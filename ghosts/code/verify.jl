# Independent checks in Julia for "Ghosts in Poisson brane gases".
# Run with  julia verify.jl .  Uses only the standard library.

using Random

passed = Ref(0); failed = Ref(0)
function check(name, ok, detail = "")
    println((ok ? "PASS " : "FAIL ") * name * (isempty(detail) ? "" : "  " * detail))
    ok ? (passed[] += 1) : (failed[] += 1)
end

# Gas with nu = c dt/t on [a, b]: phi(z) = c (log(b/a) - log((z+b)/(z+a))).
phi_box(z, c, a, b) = c * (log(b / a) - log((z + b) / (z + a)))

function sample(rng, c, a, b)
    n = 0
    m = c * log(b / a)
    # Poisson(m) by inversion
    u = rand(rng); p = exp(-m); s = p
    while u > s
        n += 1; p *= m / n; s += p
    end
    return [a * exp(rand(rng) * log(b / a)) for _ in 1:n]
end

rng = MersenneTwister(181026)
c, a, b = 2.0, 1.0, 3.0
M = 400_000
confs = [sample(rng, c, a, b) for _ in 1:M]
for z in (0.7 + 0.0im, 4.0 + 3.0im, -0.5 + 0.2im, 25.0im)
    vals = [1 / prod(1 .+ z ./ t; init = 1.0 + 0im) for t in confs]
    mean = sum(vals) / M
    sd = sqrt(sum(abs2.(vals .- mean)) / M)
    exact = exp(-phi_box(z, c, a, b))
    err = abs(mean - exact)
    check("E[1/F($z)] = exp(-phi) for c=2 on [1,3]", err < 5 * sd / sqrt(M),
          "err/se = $(round(err / (sd / sqrt(M)), digits = 2))")
end

# x Gbar(x) = exp(-phi(x)) decreases from 1 to exp(-|nu|): no positive
# Kallen-Lehmann representation.
xs = exp.(range(-6, 8, length = 200))
vals = exp.(-phi_box.(xs, c, a, b))
check("x Gbar(x) strictly decreasing", all(diff(vals) .< 0))
check("limit exp(-|nu|) = 3^-2", abs(vals[end] - 3.0^(-2)) < 1e-3)

# Scale-free gas with 1 < theta < 2: density -(1/pi) Im Gbar(-s+i0) > 0.
for theta in (1.25, 1.5, 1.75)
    ok = true
    for s in exp.(range(0.01, 5, length = 100))
        z = -s + 1e-12im
        g = (1 + z)^(-theta) / z
        rho = -imag(g) / pi
        exact = -sin(pi * theta) / pi * (s - 1)^(-theta) / s
        ok &= rho > 0 && abs(rho - exact) < 1e-6 * abs(exact)
    end
    check("theta = $theta: averaged density positive above t0", ok)
end

println("$(passed[]) checks passed, $(failed[]) failed")
exit(failed[] == 0 ? 0 : 1)
