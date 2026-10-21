#!/usr/bin/env python3
"""Run the five audit scripts in turn and exit nonzero if any of them fails.

Each script is independent of the others and of code/verify.py; each prints one
PASS or FAIL line per check and a summary line.  check_positivity.py and
check_bubble.py take several minutes each (oscillatory and two-dimensional
quadrature at high precision).
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = ["check_oneloop_matrix.py", "check_powercounting.py", "check_vacuum.py",
           "check_positivity.py", "check_bubble.py"]

failed = []
for name in SCRIPTS:
    print("=" * 78)
    print(name)
    print("=" * 78)
    rc = subprocess.call([sys.executable, os.path.join(HERE, name)])
    if rc:
        failed.append(name)
print()
if failed:
    print("audit: FAILED in " + ", ".join(failed))
    sys.exit(1)
print("audit: all %d scripts passed" % len(SCRIPTS))
