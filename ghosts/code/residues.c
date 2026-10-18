/* Residues of frozen brane gases, in long double arithmetic.
 *
 * Build and run:  cc -O2 -o residues residues.c -lm && ./residues
 *
 * For random configurations of the scale-free gas (masses t0 e^{u_k}, the u_k
 * the arrival times of a rate-theta Poisson process) the program checks that
 * the residues r_k = -1/prod_{j != k}(1 - t_k/t_j) alternate in sign starting
 * with a ghost, that they sum to -1, that |r_k| grows when heavier masses are
 * added, and that log|r_k| = -S_k + R_k with 0 <= R_k.  It also tests the
 * half-angle inequality used to dominate the products in the average.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static unsigned long long state = 0x2026101800000001ULL;

static double uniform01(void)
{
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return ((state >> 11) + 0.5) * (1.0 / 9007199254740992.0);
}

static int passed = 0, failed = 0;

static void check(const char *name, int ok)
{
    printf("%s %s\n", ok ? "PASS" : "FAIL", name);
    if (ok) passed++; else failed++;
}

#define NMAX 40

int main(void)
{
    const double theta = 1.0;
    int signs_ok = 1, sum_ok = 1, mono_ok = 1, split_ok = 1;
    for (int trial = 0; trial < 200; trial++) {
        long double u[NMAX], t[NMAX];
        long double acc = 0.0L;
        for (int i = 0; i < NMAX; i++) {
            acc += -logl((long double)uniform01()) / theta;
            u[i] = acc;
            t[i] = expl(u[i]);
        }
        for (int N = 20; N <= 24; N += 4) {
            long double sum = 0.0L;
            for (int k = 0; k < N; k++) {
                long double p = 1.0L;
                for (int j = 0; j < N; j++)
                    if (j != k) p *= 1.0L - t[k] / t[j];
                long double r = -1.0L / p;
                if ((r < 0) != (k % 2 == 0)) signs_ok = 0;
                sum += r;
                if (N == 24 && k < 20) {
                    long double q = 1.0L;
                    for (int j = 0; j < 20; j++)
                        if (j != k) q *= 1.0L - t[k] / t[j];
                    if (fabsl(r) < fabsl(1.0L / q)) mono_ok = 0;
                }
                if (N == 24) {
                    long double S = 0.0L, R = 0.0L;
                    for (int j = 0; j < k; j++) S += u[k] - u[j];
                    for (int j = 0; j < N; j++)
                        if (j != k) R -= log1pl(-expl(-fabsl(u[j] - u[k])));
                    if (R < 0 || fabsl(logl(fabsl(r)) - (-S + R)) > 1e-9L) split_ok = 0;
                }
            }
            if (fabsl(sum + 1.0L) > 1e-12L) sum_ok = 0;
        }
    }
    check("residues alternate, lightest a ghost (200 configurations)", signs_ok);
    check("residues sum to -1", sum_ok);
    check("|r_k| grows when masses are added", mono_ok);
    check("log|r_k| = -S_k + R_k with R_k >= 0", split_ok);

    /* The half-angle bound |t + z| >= cos(arg z / 2) (t + |z|), t > 0. */
    int ha_ok = 1;
    for (int i = 0; i < 1000000; i++) {
        double t = exp(20.0 * uniform01() - 10.0);
        double rad = exp(20.0 * uniform01() - 10.0);
        double ang = M_PI * (2.0 * uniform01() - 1.0) * 0.999999;
        double x = rad * cos(ang), y = rad * sin(ang);
        double lhs = hypot(t + x, y), rhs = cos(ang / 2.0) * (t + rad);
        if (lhs < rhs * (1.0 - 1e-12)) ha_ok = 0;
    }
    check("|t+z| >= cos(arg z/2)(t+|z|) on 10^6 random pairs", ha_ok);

    printf("%d checks passed, %d failed\n", passed, failed);
    return failed ? 1 : 0;
}
