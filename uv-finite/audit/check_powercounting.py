#!/usr/bin/env python3
"""Audit 02: power counting, by exhaustive enumeration instead of the merging lemma.

Sec. "Superficial degree" bounds the degree of an L-loop subgraph by

    omega_bar(L) = 4L - (2n+2)(L-1) + 2 (L - n - 1 + m_sigma)_+ ,

using the merging inequality (a-k)_+ + (b-k)_+ <= (a+b-2-k)_+ to maximise
sum_v (h_v - k)_+ over the vertex degrees h_v >= 2 with sum_v h_v = 2I,
I = V + L - 1.  Here the maximum is found by brute force over every degree
vector of every (V, L) in a range, with no inequality assumed, and compared
with (2L - k)_+.  Then the table of the paper, the threshold n >= 4, the
marginal n = 3, the growth with m_sigma, and the standard local-theory degree
are checked from the formula.  This repeats PowerCounting.lean's integer
statements by a different method (search instead of proof).
"""
import sys
from functools import lru_cache

FAILS = 0


def check(name, ok):
    global FAILS
    print(("PASS " if ok else "FAIL ") + name)
    if not ok:
        FAILS += 1


def pos(x):
    return x if x > 0 else 0


@lru_cache(maxsize=None)
def max_excess(V, total, k):
    """max of sum_{v=1}^V (h_v - k)_+ over h_v >= 2 with sum h_v = total."""
    if V == 1:
        return pos(total - k) if total >= 2 else None
    best = None
    for h in range(2, total - 2 * (V - 1) + 1):
        rest = max_excess(V - 1, total - h, k)
        if rest is None:
            continue
        cand = pos(h - k) + rest
        if best is None or cand > best:
            best = cand
    return best


ok_merge = True
worst = 0
for L in range(1, 9):
    for V in range(1, 2 * L + 3):
        I = V + L - 1
        total = 2 * I
        if total < 2 * V:
            continue
        for k in range(3, 32):
            m = max_excess(V, total, k)
            if m is None:
                continue
            # the paper's claim: max over degree vectors <= (2L - k)_+, and it is
            # attained when all extra lines meet at one vertex (V arbitrary).
            if m > pos(2 * L - k):
                ok_merge = False
                worst = max(worst, m - pos(2 * L - k))
check("sum_v (h_v - k)_+ <= (2L - k)_+ for all degree vectors, L<=8, V<=2L+2, 3<=k<=31 (exhaustive)", ok_merge)

# attained: a star with one vertex of degree 2L and V-1 vertices of degree 2
ok_att = True
for L in range(1, 9):
    for V in range(1, 2 * L + 3):
        total = 2 * (V + L - 1)
        for k in range(3, 32):
            m = max_excess(V, total, k)
            if m is not None and V >= 1 and total - 2 * (V - 1) >= 2:
                ok_att &= (m == pos(total - 2 * (V - 1) - k))
check("the bound is attained by the vertex that carries all extra lines", ok_att)


def omega_bar(L, n, m):
    return 4 * L - (2 * n + 2) * (L - 1) + 2 * pos(L - n - 1 + m)


# table tab:power, upper block (m_sigma = 0) and lower block (m_sigma -> n-2)
table_upper = {3: [4, 0, -4, -8, -10], 4: [4, -2, -8, -14, -20], 5: [4, -4, -12, -20, -28],
               6: [4, -6, -16, -26, -36], 7: [4, -8, -20, -32, -44], 8: [4, -10, -24, -38, -52]}
table_lower = {3: [4, 0, -4, -6, -8], 4: [4, -2, -8, -12, -16], 5: [4, -4, -12, -18, -24],
               6: [4, -6, -16, -24, -32], 7: [4, -8, -20, -30, -40], 8: [4, -10, -24, -36, -48]}
check("Table tab:power, upper block (m_sigma = 0)",
      all(omega_bar(L, n, 0) == table_upper[n][L - 1] for n in table_upper for L in range(1, 6)))
check("Table tab:power, lower block (m_sigma = n - 2)",
      all(omega_bar(L, n, n - 2) == table_lower[n][L - 1] for n in table_lower for L in range(1, 6)))
check("one loop: omega_bar(1) = 4 for every n, m", all(omega_bar(1, n, m) == 4 for n in range(1, 30) for m in range(0, n)))
check("two loops: omega_bar(2) = 6 - 2n whenever m_sigma < n - 2 (m integer -> m <= n-3)",
      all(omega_bar(2, n, m) == 6 - 2 * n for n in range(3, 30) for m in range(0, max(0, n - 2))))
check("local theory: omega_bar = 2n+2 + (2-2n)L for m=0, L <= n+1",
      all(omega_bar(L, n, 0) == 2 * n + 2 + (2 - 2 * n) * L for n in range(1, 20) for L in range(1, n + 2)))
check("local theory: omega_bar = (4-2n)L for m=0, L > n+1",
      all(omega_bar(L, n, 0) == (4 - 2 * n) * L for n in range(1, 20) for L in range(n + 2, n + 30)))
check("m_sigma -> n-2 limit: omega_bar = -(2n-4)(L-1) for L >= 3",
      all(omega_bar(L, n, n - 2) == -(2 * n - 4) * (L - 1) for n in range(3, 20) for L in range(3, 40)))
# the threshold
neg_all = lambda n: all(omega_bar(L, n, m) < 0 for L in range(2, 60) for m in range(0, max(1, n - 1)))
check("omega_bar(L) < 0 for all L >= 2 and all 0 <= m_sigma <= n-2 exactly when n >= 4 (n = 1..20)",
      all(neg_all(n) == (n >= 4) for n in range(1, 21)))
check("n = 3 is marginal: omega_bar(2) = 0, omega_bar(L) < 0 for L >= 3",
      omega_bar(2, 3, 0) == 0 and omega_bar(2, 3, 1) == 0 and all(omega_bar(L, 3, m) < 0 for L in range(3, 60) for m in (0, 1)))
check("omega_bar is nondecreasing in m_sigma", all(omega_bar(L, n, m) <= omega_bar(L, n, m + 1)
                                                    for n in range(2, 20) for L in range(1, 40) for m in range(0, n - 2)))
print()
print("power counting audit: %d failures" % FAILS)
sys.exit(1 if FAILS else 0)
