/*
 * formfactor_check.c
 *
 * Independent checks, in C99 with no external libraries, of the closed-form
 * statements about the skeleton a(z) = exp[(n/2) Ein(z^2/M^4)] and about the
 * fluctuations of one realization of the gas, used in
 * "Finite quantum gravity from a scale-invariant gas of branes".
 *
 * Every statement checked here is derived analytically in the paper.  The
 * special functions are implemented from scratch (power series, continued
 * fraction, adaptive Simpson quadrature), so this program does not share code
 * with the Python checks.  Units: M = 1.
 *
 * References such as Eq. (omega) or Sec. degree name the LaTeX labels of the
 * paper (\label{eq:omega}, \label{sec:degree}).
 *
 * Build and run:   cc -O2 -std=c99 -o formfactor_check formfactor_check.c -lm
 *                  ./formfactor_check
 */
#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static const double EG = 0.57721566490153286061;   /* Euler's constant */
static const double PI = 3.14159265358979323846;
static int npass = 0, nfail = 0;

static void check(const char *name, int ok)
{
    printf("[%s] %s\n", ok ? "PASS" : "FAIL", name);
    if (ok) npass++; else nfail++;
}

/* ---------- special functions ---------- */

/* Ein(y) = sum_{k>=1} (-1)^{k+1} y^k / (k k!), used for |y| <= 1 */
static double ein_series(double y)
{
    double term = 1.0, sum = 0.0;
    for (int k = 1; k < 60; k++) {
        term *= y / k;                         /* y^k / k! */
        double t = ((k % 2) ? 1.0 : -1.0) * term / k;
        sum += t;
        if (fabs(t) < 1e-18 * fabs(sum)) break;
    }
    return sum;
}

/* E1(z) for Re z > 0, |z| >= 1: modified Lentz continued fraction
   E1(z) = e^{-z} / (z + 1/(1 + 1/(z + 2/(1 + 2/(z + ...)))))           */
static double complex e1_cf(double complex z)
{
    const double tiny = 1e-300;
    double complex b = z + 1.0, c = 1.0 / tiny, d = 1.0 / b, h = d;
    for (int i = 1; i < 10000; i++) {
        double an = -(double)i * i;
        b += 2.0;
        d = 1.0 / (an * d + b);
        c = b + an / c;
        double complex del = c * d;
        h *= del;
        if (cabs(del - 1.0) < 1e-16) break;
    }
    return h * cexp(-z);
}

/* real Ein(y), y >= 0 */
static double ein(double y)
{
    if (y <= 1.0) return ein_series(y);
    return log(y) + EG + creal(e1_cf(y));
}

/* adaptive Simpson */
typedef double (*fn)(double, void *);
static double simpson_rec(fn f, void *p, double a, double b, double fa, double fm,
                          double fb, double whole, double eps, int depth)
{
    double m = 0.5 * (a + b), lm = 0.5 * (a + m), rm = 0.5 * (m + b);
    double flm = f(lm, p), frm = f(rm, p);
    double left = (m - a) / 6 * (fa + 4 * flm + fm);
    double right = (b - m) / 6 * (fm + 4 * frm + fb);
    if (depth <= 0 || fabs(left + right - whole) <= 15 * eps)
        return left + right + (left + right - whole) / 15;
    return simpson_rec(f, p, a, m, fa, flm, fm, left, eps / 2, depth - 1) +
           simpson_rec(f, p, m, b, fm, frm, fb, right, eps / 2, depth - 1);
}
static double integrate(fn f, void *p, double a, double b, double eps)
{
    double fa = f(a, p), fb = f(b, p), fm = f(0.5 * (a + b), p);
    return simpson_rec(f, p, a, b, fa, fm, fb, (b - a) / 6 * (fa + 4 * fm + fb), eps, 50);
}

static double ein_integrand(double t, void *p)
{
    (void)p;
    if (t < 1e-8) return 1.0 - 0.5 * t;
    return -expm1(-t) / t;
}

/* lower incomplete gamma function gamma(b, y) by its power series, y <= 200 */
static double lowgam(double b, double y)
{
    double term = 1.0 / b, sum = term;
    for (int k = 1; k < 2000; k++) {
        term *= y / (b + k);
        sum += term;
        if (term < 1e-17 * sum) break;
    }
    return exp(b * log(y) - y) * sum;
}

/* xorshift64* uniform on (0, 1) and Box-Muller normal */
static double urand(unsigned long long *s)
{
    *s ^= *s >> 12; *s ^= *s << 25; *s ^= *s >> 27;
    return ((*s * 2685821657736338717ULL) >> 11) * (1.0 / 9007199254740992.0) + 1e-17;
}
static double nrand(unsigned long long *s)
{
    return sqrt(-2 * log(urand(s))) * cos(2 * PI * urand(s));
}

/* a(x) for real x with M = 1 */
static double a_of(double x, double n) { return exp(0.5 * n * ein(x * x)); }

/* integrands for J_n and for int dk / a(k^2); substitution k = u/(1-u) on [0,1) */
struct par { double n; };
static double jn_integrand(double u, void *p)
{
    double n = ((struct par *)p)->n;
    if (u >= 1.0) return 0.0;
    double k = u / (1 - u), dk = 1.0 / ((1 - u) * (1 - u));
    return pow(1 + k * k * k * k, -0.5 * n) * dk;
}
static double inva_integrand(double u, void *p)
{
    double n = ((struct par *)p)->n;
    if (u >= 1.0) return 0.0;
    double k = u / (1 - u), dk = 1.0 / ((1 - u) * (1 - u));
    return dk / a_of(k * k, n);          /* a(z) at z = k^2, i.e. x = k^2 */
}

/* integrands for the zero-point energy checks (Sec. vacuum) */
struct wpar { double W, s; };
static double ein_of_w(double w, void *p) { (void)p; return ein(w); }
static double laplace_integrand(double u, void *p)
{
    double W = ((struct wpar *)p)->W;
    if (u >= 1.0) return 0.0;
    double w = W * u / (1 - u), dw = W / ((1 - u) * (1 - u));
    return ein(w) * exp(-w / W) * dw;
}
static double mellin_integrand(double t, void *p)
{
    double s = ((struct wpar *)p)->s;
    return exp(s * t) * ein(exp(t));
}

/* exact rational arithmetic for the killer matrix */
typedef struct { long long p, q; } rat;
static long long gcdll(long long a, long long b) { a = llabs(a); b = llabs(b); while (b) { long long t = a % b; a = b; b = t; } return a ? a : 1; }
static rat mk(long long p, long long q) { if (q < 0) { p = -p; q = -q; } long long g = gcdll(p, q); rat r = {p / g, q / g}; return r; }
static rat add(rat a, rat b) { return mk(a.p * b.q + b.p * a.q, a.q * b.q); }
static rat mul(rat a, rat b) { return mk(a.p * b.p, a.q * b.q); }
static rat neg(rat a) { return mk(-a.p, a.q); }
static int is0(rat a) { return a.p == 0; }

/* integrands for the cutoff variance of Eq. (cutvar): v^(zeta-5) (1 - e^{-v^2})^2 and its Gaussian part */
static double cutvar_integrand(double v, void *p)
{
    double zeta = *(double *)p, x = v * v;
    double r = x < 1e-8 ? 1.0 - 0.5 * x : -expm1(-x) / x;
    return pow(v, zeta - 1.0) * r * r;
}
static double cutvar_tail(double v, void *p)
{
    double zeta = *(double *)p;
    return pow(v, zeta - 5.0) * (2.0 * exp(-v * v) - exp(-2.0 * v * v));
}
static int omega_m2(int n, int L, int j) /* omega_bar with 2 m_sigma = j */
{
    int e = 2 * L - 2 * n - 2 + j;
    return 4 * L - (2 * n + 2) * (L - 1) + (e > 0 ? e : 0);
}

int main(void)
{
    char buf[256];

    /* 1. Ein: series / continued fraction against direct quadrature */
    double ys[] = {0.05, 0.5, 1.0, 1.5, 3.0, 8.0, 20.0, 45.0};
    for (int i = 0; i < 8; i++) {
        double q = integrate(ein_integrand, NULL, 0.0, ys[i], 1e-14);
        snprintf(buf, sizeof buf, "Ein(%.2f): closed form = quadrature (rel. err. < 1e-10)", ys[i]);
        check(buf, fabs(ein(ys[i]) - q) < 1e-10 * fabs(q));
    }

    /* 2. g(y) = Ein(y) - log(1+y) is nondecreasing, 0 <= g <= gamma */
    {
        int ok = 1; double prev = 0.0;
        for (int i = 1; i <= 40000; i++) {
            double y = 1e-3 * i * (1 + 1e-3 * i);      /* y up to ~1.6e3 */
            double g = ein(y) - log1p(y);
            if (g < prev - 1e-13 || g < -1e-15 || g > EG + 1e-13) ok = 0;
            prev = g;
        }
        check("g(y) = Ein(y) - log(1+y) nondecreasing with 0 <= g <= gamma_E", ok);
        check("g(y) -> gamma_E as y -> infinity (|g(1e3) - gamma_E| < 1e-3)", fabs(ein(1e3) - log1p(1e3) - EG) < 1e-3);
    }

    /* 3. two-sided bound (1+x^2)^{n/2} <= a(x) <= e^{n gamma/2} (1+x^2)^{n/2} */
    for (int n = 1; n <= 12; n++) {
        int ok = 1;
        for (int i = -4000; i <= 4000; i++) {
            double x = sinh(i * 1e-3 * 2.0) ;          /* |x| up to ~ 1.5e3 */
            double a = a_of(x, n), lo = pow(1 + x * x, 0.5 * n);
            if (a < lo * (1 - 1e-12) || a > exp(0.5 * n * EG) * lo * (1 + 1e-12)) ok = 0;
        }
        snprintf(buf, sizeof buf, "n=%d: two-sided bound on the real axis", n);
        check(buf, ok);
    }

    /* 4. low-momentum expansion a = 1 + (n/2) x^2 + n(n-1)/8 x^4 + O(x^6) */
    for (int n = 2; n <= 10; n += 4) {
        double x = 2e-3, a = a_of(x, n);
        double c4 = (a - 1 - 0.5 * n * x * x) / pow(x, 4);
        snprintf(buf, sizeof buf, "n=%d: x^4 coefficient of a equals n(n-1)/8", n);
        check(buf, fabs(c4 - n * (n - 1) / 8.0) < 1e-3);
    }

    /* 5. J_n = (1/4) Gamma(1/4) Gamma(n/2 - 1/4) / Gamma(n/2), and the bounds on
          int_0^inf dk / a(k^2)  (potential at the origin)                       */
    for (int n = 3; n <= 12; n++) {
        struct par p = {n};
        double J = 0.25 * exp(lgamma(0.25) + lgamma(0.5 * n - 0.25) - lgamma(0.5 * n));
        double Jq = integrate(jn_integrand, &p, 0.0, 1.0, 1e-13);
        snprintf(buf, sizeof buf, "n=%d: J_n Beta-function formula = quadrature", n);
        check(buf, fabs(J - Jq) < 1e-9);
        if (n >= 4) {
            double Iq = integrate(inva_integrand, &p, 0.0, 1.0, 1e-12);
            snprintf(buf, sizeof buf, "n=%d: e^{-n gamma/2} J_n <= int dk/a <= J_n  (%.5f <= %.5f <= %.5f)",
                     n, exp(-0.5 * n * EG) * J, Iq, J);
            check(buf, exp(-0.5 * n * EG) * J <= Iq && Iq <= J);
            /* sharper bracket: Ein(w) <= w gives K_n <= int dk/a; Wendel gives J_n <= 2n/(2n-1) K_n */
            double K = 0.25 * tgamma(0.25) * pow(2.0 / n, 0.25);
            snprintf(buf, sizeof buf, "n=%d: K_n <= int dk/a <= J_n <= 2n/(2n-1) K_n  (%.5f <= %.5f <= %.5f <= %.5f)",
                     n, K, Iq, J, 2.0 * n / (2.0 * n - 1.0) * K);
            check(buf, K <= Iq && Iq <= J && J <= 2.0 * n / (2.0 * n - 1.0) * K);
        }
    }

    /* 6. complex sector: |E1(w)| <= e^{-Re w}/|w| for Re w > 0, and
          |a(z)/(c z^n) - 1| <= exp[(n/2) e^{-Re z^2}/|z|^2] - 1 for |arg z| < pi/4 */
    {
        int ok1 = 1, ok2 = 1; double n = 6;
        for (int ir = 0; ir < 60; ir++) {
            double r = 1.0 + 0.1 * ir;
            for (int it = 0; it < 40; it++) {
                double th = (PI / 4 - 0.02) * it / 39.0;
                double complex z = r * cexp(I * th), w = z * z;
                double complex e1 = e1_cf(w);
                if (cabs(e1) > exp(-creal(w)) / cabs(w) * (1 + 1e-12)) ok1 = 0;
                /* a(z) = c z^n exp[(n/2) E1(z^2)] from Ein(w) = log w + gamma + E1(w) */
                double complex ratio = cexp(0.5 * n * e1);
                double bound = expm1(0.5 * n * exp(-creal(w)) / cabs(w));
                if (cabs(ratio - 1) > bound * (1 + 1e-12) + 1e-14) ok2 = 0;   /* 1e-14: rounding */
            }
        }
        check("|E1(w)| <= e^{-Re w}/|w| on the sector |arg w| < pi/2", ok1);
        check("a(z) = c (z/M^2)^n [1 + exponentially small] in |arg z| < pi/4", ok2);
        /* cross-check Ein(w) = log w + gamma + E1(w) at a complex point by series */
        double complex w = 0.8 + 0.5 * I, term = 1, s = 0;
        for (int k = 1; k < 80; k++) { term *= w / k; s += ((k % 2) ? 1.0 : -1.0) * term / k; }
        /* E1 at |w| < 1 from the series E1 = -gamma - log w + Ein(w): consistency
           with the continued fraction at |w| ~ 1 on the boundary */
        double complex w2 = 1.2 + 0.4 * I, t2 = 1, s2 = 0;
        for (int k = 1; k < 120; k++) { t2 *= w2 / k; s2 += ((k % 2) ? 1.0 : -1.0) * t2 / k; }
        check("Ein(w) = log w + gamma + E1(w) at complex w (series vs continued fraction)",
              cabs(s2 - (clog(w2) + EG + e1_cf(w2))) < 1e-10);
        double complex w3 = 0.3 + 2.5 * I, t3 = 1, s3 = 0;
        for (int k = 1; k < 200; k++) { t3 *= w3 / k; s3 += ((k % 2) ? 1.0 : -1.0) * t3 / k; }
        check("same identity near the imaginary axis, w = 0.3 + 2.5i",
              cabs(s3 - (clog(w3) + EG + e1_cf(w3))) < 1e-10);
        (void)s;
    }

    /* 7. resonances of 2 - e^{-s^2 z^2}: s^2 z^2 = -log 2 - 2 pi i k lie in pi/4 < |arg z| < 3pi/4 */
    {
        int ok = 1;
        for (int k = -200; k <= 200; k++) {
            double complex w = -log(2.0) - 2 * PI * I * k, z = csqrt(w);
            double complex zz[2] = {z, -z};
            for (int j = 0; j < 2; j++) {
                double th = fabs(carg(zz[j]));
                if (!(th > PI / 4 && th < 3 * PI / 4)) ok = 0;
                if (cabs(1 + (1 - cexp(-zz[j] * zz[j]))) > 1e-9 * (1 + cabs(zz[j]))) ok = 0;
            }
        }
        check("resonances of 2 - exp(-s^2 z^2) lie strictly inside pi/4 < |arg z| < 3pi/4", ok);
    }

    /* 8. power counting from the vertex bound:
          omega(L) = 4L - (2n+2)(L-1) + 2(L-n-1)_+ < 0 for all L >= 2  <=>  n >= 4 */
    {
        int ok = 1;
        for (long n = 1; n <= 200; n++) {
            int allneg = 1;
            for (long L = 2; L <= 1000; L++) {
                long w = 4 * L - (2 * n + 2) * (L - 1) + 2 * (L - n - 1 > 0 ? L - n - 1 : 0);
                if (w >= 0) allneg = 0;
            }
            if (allneg != (n >= 4)) ok = 0;
        }
        check("superficial degree negative for every L >= 2 exactly when n >= 4", ok);
        long w32 = 4 * 2 - (2 * 3 + 2) * 1 + 0;
        check("n = 3 is marginal: omega(2) = 0", w32 == 0);
        /* the table: n = 3, L = 5 is the first entry that differs from 2n+2+(2-2n)L */
        long w35 = 4 * 5 - 8 * 4 + 2 * 1, std35 = 8 + (2 - 6) * 5;
        check("n = 3, L = 5: omega = -10, standard estimate -12", w35 == -10 && std35 == -12);
    }

    /* 8b. merging inequality: sum_v (h_v - k)_+ <= (sum_v h_v - 2(V-1) - k)_+ for h_v >= 2, k >= 2,
           exhaustively for V <= 4, 2 <= h_v <= 12, 2 <= k <= 14 */
    {
        int ok = 1;
        for (int V = 1; V <= 4; V++)
            for (int k = 2; k <= 14; k++) {
                int h[4] = {2, 2, 2, 2};
                for (;;) {
                    int lhs = 0, sum = 0;
                    for (int v = 0; v < V; v++) { lhs += h[v] > k ? h[v] - k : 0; sum += h[v]; }
                    int rhs = sum - 2 * (V - 1) - k; if (rhs < 0) rhs = 0;
                    if (lhs > rhs) ok = 0;
                    int v = 0;
                    while (v < V && ++h[v] > 12) { h[v] = 2; v++; }
                    if (v == V) break;
                }
            }
        check("merging inequality, exhaustive for V <= 4", ok);
    }

    /* 8c. insertion lemma: (|q| + s)^2 <= (1+2r+2r^2)(1+|q|)(1+|q'|) prod_a (1+|l_a|),
           s = sum |l_a|, q' = q + sum l_a; random four-vectors over seven decades */
    {
        unsigned long long st = 88172645463325252ULL;
        #define RNG() (st ^= st << 13, st ^= st >> 7, st ^= st << 17, (double)(st >> 11) / 9007199254740992.0)
        int ok = 1;
        double worst = 0;
        for (long t = 0; t < 400000; t++) {
            int r = 1 + (int)(5 * RNG());
            double q[4], qp[4], l[5][4], s = 0, prodl = 1;
            double sc = pow(10.0, -2 + 7 * RNG());
            for (int m = 0; m < 4; m++) q[m] = (2 * RNG() - 1) * sc;
            for (int a = 0; a < r; a++) {
                double sa = pow(10.0, -2 + 7 * RNG());
                for (int m = 0; m < 4; m++) l[a][m] = (2 * RNG() - 1) * sa;
            }
            double u = RNG();
            if (u < 0.3) { /* make q' small: the last leg nearly cancels everything */
                for (int m = 0; m < 4; m++) {
                    double acc = q[m];
                    for (int a = 0; a < r - 1; a++) acc += l[a][m];
                    l[r - 1][m] = -acc + (2 * RNG() - 1);
                }
            } else if (u < 0.45) for (int m = 0; m < 4; m++) q[m] = 1e-3 * (2 * RNG() - 1);
            for (int m = 0; m < 4; m++) { qp[m] = q[m]; for (int a = 0; a < r; a++) qp[m] += l[a][m]; }
            double nq = 0, nqp = 0;
            for (int m = 0; m < 4; m++) { nq += q[m] * q[m]; nqp += qp[m] * qp[m]; }
            nq = sqrt(nq); nqp = sqrt(nqp);
            for (int a = 0; a < r; a++) {
                double na = 0;
                for (int m = 0; m < 4; m++) na += l[a][m] * l[a][m];
                na = sqrt(na); s += na; prodl *= 1 + na;
            }
            double ratio = (nq + s) * (nq + s) / ((1 + 2.0 * r + 2.0 * r * r) * (1 + nq) * (1 + nqp) * prodl);
            if (ratio > worst) worst = ratio;
            if (ratio > 1) ok = 0;
        }
        snprintf(buf, sizeof buf, "insertion lemma with C_r = 1+2r+2r^2, 4e5 random configurations (worst ratio %.3f)", worst);
        check(buf, ok);
        #undef RNG
    }

    /* 8d. admissible class: C_+ and the moment int t(1-phi) dt for the three responses */
    {
        struct { const char *name; int kind; double Cp, mom; } R[3] = {
            {"1-e^{-w^2}", 0, 0.5 * EG, 0.5},
            {"1-(1+w^2)e^{-w^2}", 1, 0.5 * (EG - 1), 1.0},
            {"1-e^{-w^4}", 2, 0.25 * EG, 0.25 * sqrt(PI)}};
        for (int i = 0; i < 3; i++) {
            int K = R[i].kind;
            /* int_0^1 phi(t)/t dt - int_1^inf (1-phi(t))/t dt - by composite Simpson on fine grids */
            double A = 0, B = 0, Mo = 0;
            int Ng = 200000;
            for (int j = 0; j <= Ng; j++) {
                double t = (double)j / Ng, w = (j == 0 || j == Ng) ? 1 : (j % 2 ? 4 : 2);
                double ph = t == 0 ? 0 : (K == 0 ? -expm1(-t * t) : K == 1 ? 1 - (1 + t * t) * exp(-t * t) : -expm1(-t * t * t * t));
                A += w * (t == 0 ? 0 : ph / t);
            }
            A /= 3.0 * Ng;
            double T = 12.0;
            for (int j = 0; j <= Ng; j++) {
                double t = 1 + (T - 1) * j / Ng, w = (j == 0 || j == Ng) ? 1 : (j % 2 ? 4 : 2);
                double om = K == 0 ? exp(-t * t) : K == 1 ? (1 + t * t) * exp(-t * t) : exp(-t * t * t * t);
                B += w * om / t;
            }
            B *= (T - 1) / (3.0 * Ng);
            for (int j = 0; j <= Ng; j++) {
                double t = T * j / Ng, w = (j == 0 || j == Ng) ? 1 : (j % 2 ? 4 : 2);
                double om = K == 0 ? exp(-t * t) : K == 1 ? (1 + t * t) * exp(-t * t) : exp(-t * t * t * t);
                Mo += w * t * om;
            }
            Mo *= T / (3.0 * Ng);
            snprintf(buf, sizeof buf, "%s: C_+ = %.6f (closed form %.6f), moment %.6f (closed form %.6f)", R[i].name, A - B, R[i].Cp, Mo, R[i].mom);
            check(buf, fabs(A - B - R[i].Cp) < 1e-8 && fabs(Mo - R[i].mom) < 1e-8);
        }
    }

    /* 8e. identical branes: Psi_*(x) = Ein(x^2) - Ein(2x^2)/2 against direct quadrature */
    {
        int ok = 1;
        double xs[3] = {0.4, 1.1, 2.3};
        for (int i = 0; i < 3; i++) {
            double x = xs[i], acc = 0;
            int Ng = 200000;
            for (int j = 0; j <= Ng; j++) {
                double t = (double)j / Ng, w = (j == 0 || j == Ng) ? 1 : (j % 2 ? 4 : 2);
                double ph = -expm1(-x * x * t * t);
                acc += w * (t == 0 ? 0 : ph * ph / t);
            }
            acc /= 3.0 * Ng;
            if (fabs(acc - (ein(x * x) - 0.5 * ein(2 * x * x))) > 1e-9) ok = 0;
        }
        check("Psi_*(x) = Ein(x^2) - Ein(2x^2)/2 at x = 0.4, 1.1, 2.3", ok);
    }

    /* 8f. one realization of the fine-grained gas, strength th0 tau^zeta, th0 = log 2:
           incomplete-gamma variances, Table of fluctuations, Eq. (moment1), and a Monte Carlo
           of R_inf from its Levy measure (n/zeta) dt/t^2 on (0, th0] with a private generator. */
    {
        double th0 = log(2.0);
        int ok = 1;
        double zs[3] = {5.0, 9.0, 13.0}, xs[3] = {0.5, 1.7, 4.0};
        for (int a = 0; a < 3; a++)
            for (int b = 0; b < 3; b++) {
                double z = zs[a], x = xs[b], vR = 0, vf = 0;
                int Ng = 200000;
                for (int j = 0; j <= Ng; j++) {
                    double t = (double)j / Ng, w = (j == 0 || j == Ng) ? 1 : (j % 2 ? 4 : 2);
                    double ph = -expm1(-x * x * t * t), tz = pow(t, z - 1);
                    vR += w * tz * ph * ph;
                    vf += w * tz * exp(-2 * x * x * t * t);
                }
                vR *= 4 * th0 / (3.0 * Ng);
                vf *= 4 * th0 / (3.0 * Ng);
                double cR = 4 * th0 * (1 / z - lowgam(z / 2, x * x) / pow(x, z) + 0.5 * lowgam(z / 2, 2 * x * x) / pow(2 * x * x, z / 2));
                double cf = 2 * th0 * lowgam(z / 2, 2 * x * x) / pow(2 * x * x, z / 2);
                if (fabs(vR / cR - 1) > 1e-8 || fabs(vf / cf - 1) > 1e-8) ok = 0;
            }
        check("Var R_Pi(x) and Var(R_Pi - R_inf): incomplete-gamma closed forms = quadrature", ok);

        double tab[5][5] = {{0.555, 1.190, 0.257, 4.47, 0.372}, {0.561, 1.195, 0.253, 3.74, 0.281},
                            {0.566, 1.198, 0.250, 3.35, 0.227}, {0.569, 1.201, 0.249, 3.11, 0.190},
                            {0.571, 1.202, 0.247, 2.97, 0.163}};
        ok = 1;
        for (int n = 4; n <= 8; n++) {
            double z = 2 * n + 1, got[5];
            got[0] = sqrt(n * th0 / z);
            /* int_0^th0 (e^t - 1 - t)/t^2 dt by Simpson */
            double acc = 0;
            int Ng = 20000;
            for (int j = 0; j <= Ng; j++) {
                double t = th0 * j / Ng, w = (j == 0 || j == Ng) ? 1 : (j % 2 ? 4 : 2);
                acc += w * (t < 1e-6 ? 0.5 + t / 6 : (expm1(t) - t) / (t * t));
            }
            acc *= th0 / (3.0 * Ng);
            got[1] = exp(n / z * acc);
            got[2] = exp(-sqrt(2 * n * th0 / z * log(20.0)));
            double lo = 0.5, hi = 50.0;           /* bisection for rms remainder = 1e-3 */
            for (int it = 0; it < 200; it++) {
                double mid = 0.5 * (lo + hi), v = 0.5 * n * th0 * lowgam(z / 2, 2 * mid * mid) / pow(2 * mid * mid, z / 2);
                if (sqrt(v) > 1e-3) lo = mid; else hi = mid;
            }
            got[3] = 0.5 * (lo + hi);
            got[4] = 2 * sqrt(th0 / (n * (z - 4)));
            for (int c = 0; c < 5; c++)
                if (fabs(got[c] - tab[n - 4][c]) > 0.0005 + 0.0011 * tab[n - 4][c]) ok = 0;
        }
        check("Table of fluctuations, n = 4..8: sd, E e^R_inf, 95% bound, x_*, delta rho/rho_1", ok);

        ok = 1;
        double taus[3] = {0.3, 1.0, 2.2};
        for (int i = 0; i < 3; i++) {
            double tau = taus[i], acc = 0, K = 8.0 / sqrt(tau);
            int Ng = 200000;
            for (int j = 0; j <= Ng; j++) {
                double k = K * j / Ng, w = (j == 0 || j == Ng) ? 1 : (j % 2 ? 4 : 2);
                acc += w * k * k * k * -exp(-tau * tau * k * k * k * k);
            }
            acc *= K / (3.0 * Ng);
            double lhs = 2 * PI * PI * acc / pow(2 * PI, 4);
            if (fabs(lhs + 0.5 / (16 * PI * PI * tau * tau)) > 1e-11) ok = 0;
        }
        check("int d^4k/(2pi)^4 [psi_*(tau k^2) - 1] = -(1/2)/(16 pi^2 tau^2)", ok);

        /* Monte Carlo of R_inf, n = 4, zeta = 9: jumps t in [e, th0] with density prop. to t^-2,
           Poisson number with mean (n/zeta)(1/e - 1/th0); jumps below e replaced by a Gaussian */
        {
            unsigned long long s = 0x9E3779B97F4A7C15ULL;
            int n = 4, Msamp = 100000;
            double z = 9.0, k = n / z, e = 1e-3, lam = k * (1 / e - 1 / th0), m1 = 0, m2 = 0, me = 0, low = 0;
            double b95 = exp(-sqrt(2 * n * th0 / z * log(20.0)));
            for (int i = 0; i < Msamp; i++) {
                /* Poisson(lam) as a sum of Poisson(<= 50) draws, each by Knuth's product method */
                double rem = lam, sum = 0;
                int N = 0;
                while (rem > 0) {
                    double step = rem > 50 ? 50 : rem, L = exp(-step), p = 1;
                    rem -= step;
                    for (;;) { p *= urand(&s); if (p <= L) break; N++; }
                }
                for (int j = 0; j < N; j++) sum += 1 / (1 / e - urand(&s) * (1 / e - 1 / th0));
                double R = sum - k * log(th0 / e) + sqrt(k * e) * nrand(&s);
                m1 += R; m2 += R * R; me += exp(R); if (exp(R) < b95) low += 1;
            }
            m1 /= Msamp; m2 = m2 / Msamp - m1 * m1; me /= Msamp; low /= Msamp;
            snprintf(buf, sizeof buf, "MC of R_inf (n = 4, zeta = 9): var %.4f (closed %.4f), E e^R %.4f (closed 1.1902), P(e^R < %.3f) = %.5f <= 0.05",
                     m2, n * th0 / z, me, b95, low);
            check(buf, fabs(m2 / (n * th0 / z) - 1) < 0.03 && fabs(me / 1.1902 - 1) < 0.01 && low <= 0.05);
        }
    }

    /* 9. one-loop poles of the three quartic operators, exact rationals.
          (dbeta1, dbeta2, dbetaE) = (1/c) A (s1, s2, s3) with
          A = [[-12, -1, -8/3], [0, 1, 8], [0, 0, 2]].
          Columns from the correlators: s1 -> 8*(-3/2) R^2,
          s2 -> 8*(1/8)(Ric^2 - R^2), s3 -> 8*(1/12)(3 Riem^2 - R^2),
          then Riem^2 = E + 4 Ric^2 - R^2.                                       */
    {
        rat c1R = mul(mk(8, 1), mk(-3, 2));
        rat c2Ric = mul(mk(8, 1), mk(1, 8)), c2R = neg(c2Ric);
        rat c3Riem = mul(mk(8, 1), mk(3, 12)), c3R = mul(mk(8, 1), mk(-1, 12));
        /* express s3 column in basis (R^2, Ric^2, E) */
        rat c3E = c3Riem, c3Ric = mul(c3Riem, mk(4, 1)), c3Rtot = add(c3R, neg(c3Riem));
        rat A[3][3] = {{c1R, c2R, c3Rtot}, {mk(0, 1), c2Ric, c3Ric}, {mk(0, 1), mk(0, 1), c3E}};
        int okcols = A[0][0].p == -12 && A[0][0].q == 1 && A[0][1].p == -1 && A[0][1].q == 1 &&
                     A[0][2].p == -8 && A[0][2].q == 3 && A[1][1].p == 1 && A[1][1].q == 1 &&
                     A[1][2].p == 8 && A[1][2].q == 1 && A[2][2].p == 2 && A[2][2].q == 1;
        check("killer matrix columns: (-12,0,0), (-1,1,0), (-8/3,8,2)", okcols);
        rat det = mul(mul(A[0][0], A[1][1]), A[2][2]);
        check("killer matrix determinant = -24 (times c^-3), never zero", det.p == -24 && det.q == 1);
        /* angular-average rationals */
        rat x = add(mk(5, 9), neg(mul(mk(3, 2), mk(1, 27))));
        rat y = add(mk(-5, 36), neg(mul(mk(3, 2), mk(13, 54))));
        check("5/9 - (3/2)(1/27) = 1/2 and -5/36 - (3/2)(13/54) = -1/2", x.p == 1 && x.q == 2 && y.p == -1 && y.q == 2);
        rat r1 = add(mk(1, 24), neg(mk(2, 48)));  /* Ric^2 coefficient */
        rat r2 = mk(3, 48);                        /* Riem^2: (3/2)/24  */
        rat r3 = neg(mk(1, 48));                   /* R^2 */
        check("<tr W^2> - <(tr W)^2>/2 = (3 Riem^2 - R^2)/48", is0(r1) && r2.p == 1 && r2.q == 16 && r3.p == -1 && r3.q == 48);
        /* tuning: for arbitrary rational beta^(0), the closed-form s cancel all three */
        rat b1 = mk(7, 5), b2 = mk(-11, 3), bE = mk(13, 7);
        rat s3 = mul(mk(-1, 2), bE);
        rat s2 = add(neg(b2), mul(mk(4, 1), bE));
        rat s1 = mul(mk(1, 12), add(add(b1, b2), mul(mk(-8, 3), bE)));
        rat e1 = add(b1, add(mul(A[0][0], s1), add(mul(A[0][1], s2), mul(A[0][2], s3))));
        rat e2 = add(b2, add(mul(A[1][1], s2), mul(A[1][2], s3)));
        rat e3 = add(bE, mul(A[2][2], s3));
        check("closed-form tuning (s1,s2,s3) cancels beta1, beta2, betaE exactly", is0(e1) && is0(e2) && is0(e3));
    }

    /* 10. zero-point energy: int_0^W Ein = W Ein(W) - W + 1 - e^{-W};
           int_0^inf Ein(w) e^{-w/W} dw = W log(1+W); Mellin transform Gamma(s)/s;
           rho_1 = (3n/2) * 2 pi^2 * Gamma(1)/4 / (2 pi)^4 = 3n/(64 pi^2).            */
    {
        int ok = 1;
        double Ws[3] = {2.0, 9.0, 35.0};
        for (int i = 0; i < 3; i++) {
            double W = Ws[i];
            double lhs = integrate(ein_of_w, NULL, 0.0, W, 1e-12);
            double rhs = W * ein(W) - W + 1 - exp(-W);
            if (fabs(lhs - rhs) > 1e-9 * W * W) ok = 0;
        }
        check("hard cutoff: int_0^W Ein(w) dw = W Ein(W) - W + 1 - e^-W", ok);
        ok = 1;
        double Wl[3] = {0.5, 4.0, 60.0};
        for (int i = 0; i < 3; i++) {
            struct wpar q = {Wl[i], 0.0};
            double lhs = integrate(laplace_integrand, &q, 0.0, 1.0 - 1e-12, 1e-12);
            if (fabs(lhs - Wl[i] * log(1 + Wl[i])) > 1e-7 * Wl[i] * Wl[i]) ok = 0;
        }
        check("smooth cutoff: int_0^inf Ein(w) e^(-w/W) dw = W log(1+W)", ok);
        ok = 1;
        double sv[2] = {-0.3, -0.5};
        for (int i = 0; i < 2; i++) {
            struct wpar q = {0.0, sv[i]};
            double lhs = integrate(mellin_integrand, &q, -150.0, 300.0, 1e-12);
            if (fabs(lhs - tgamma(sv[i]) / sv[i]) > 1e-8) ok = 0;
        }
        check("Mellin transform of Ein: Gamma(s)/s at s = -0.3, -0.5", ok);
        double n = 6.0, rho = 1.5 * n * 2 * PI * PI * tgamma(1.0) / 4 / pow(2 * PI, 4);
        check("rho_1 = 3n/(64 pi^2) for n = 6", fabs(rho - 3 * n / (64 * PI * PI)) < 1e-15);
    }

    /* 12. The derived premises (Secs. cells, vacuum) and the weakest fine graining (Sec. degree) */
    {
        double th = 1.2, sc = 0.7, zs3[3] = {0.4, 1.1, 2.5};
        int ok = 1;
        for (int i = 0; i < 3; i++) {
            double sum = 0.0, term = exp(-th), q = exp(-sc * sc * zs3[i] * zs3[i]);
            for (int N = 0; N < 120; N++) { sum += term; term *= th * q / (N + 1); }
            if (fabs(sum - exp(-th * (1.0 - q))) > 1e-15) ok = 0;
        }
        check("Eq. (cellt): Poisson sum over the number of cells = exp[-theta(1 - e^{-s^2 z^2})]", ok);
        ok = 1;
        double zetas[4] = {1.0, 2.5, 3.0, 3.5};
        for (int i = 0; i < 4; i++) {
            double z = zetas[i];
            double q = integrate(cutvar_integrand, &z, 0.0, 1.0, 1e-13) + 1.0 / (4.0 - z) - integrate(cutvar_tail, &z, 1.0, 12.0, 1e-13);
            double cz = 0.5 * tgamma(0.5 * z - 2.0) * (pow(2.0, 2.0 - 0.5 * z) - 2.0);
            if (fabs(q / cz - 1.0) > 1e-9) ok = 0;
        }
        check("Eq. (cutvar): C_zeta = Gamma(zeta/2-2)(2^(2-zeta/2)-2)/2 against Simpson, zeta = 1, 2.5, 3, 3.5", ok);
        int lower[6][5] = {{4, 0, -4, -6, -8}, {4, -2, -8, -12, -16}, {4, -4, -12, -18, -24},
                           {4, -6, -16, -24, -32}, {4, -8, -20, -30, -40}, {4, -10, -24, -36, -48}};
        ok = 1;
        for (int n = 3; n <= 8; n++)
            for (int L = 1; L <= 5; L++)
                if (omega_m2(n, L, 2 * n - 4) != lower[n - 3][L - 1]) ok = 0;
        check("Table (power), lower block: m_sigma = n - 2", ok);
        ok = 1;
        for (int n = 1; n <= 30; n++) {
            int allneg = 1;
            for (int L = 2; L <= 200; L++)
                for (int j = 0; j <= (2 * n - 4 > 0 ? 2 * n - 4 : 0); j++)
                    if (omega_m2(n, L, j) >= 0) allneg = 0;
            if (allneg != (n >= 4)) ok = 0;
        }
        check("omega_bar < 0 for 2 <= L <= 200 and all 0 <= 2 m_sigma <= 2n - 4 exactly when n >= 4", ok);
    }

    printf("\n%d checks passed, %d failed\n", npass, nfail);
    return nfail != 0;
}
