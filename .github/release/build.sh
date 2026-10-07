#!/usr/bin/env bash
# Build the release assets of "Finite quantum gravity from a scale-invariant gas of branes":
# the PDF, the LaTeX source (tex.zip), the arXiv package with the code in anc/ (tar.gz) and
# every figure as PDF and PNG with its caption (figures.zip).
#
# Usage: build.sh <version> <output directory>
# Needs pdflatex with revtex4-2 and cm-super, pdfinfo and pdftoppm (poppler), zip and python3.
set -euo pipefail

version=$1
mkdir -p "$2"
out=$(cd "$2" && pwd)
root=$(cd "$(dirname "$0")/../.." && pwd)
src=$root/uv-finite
name=uv-finite-brane-gas-v$version
stamp=$(git -C "$root" log -1 --format=%cI)
work=$(mktemp -d)

mkdir -p "$work/tex/figs" "$work/arxiv/anc"
cp "$src/uv-finite-brane-gas.tex" "$work/tex/"
cp "$src"/figs/*.png "$work/tex/figs/"
cp -r "$work/tex/." "$work/arxiv/"
cp "$src"/code/{verify.py,formfactor_check.c,verify.jl,PowerCounting.lean,make_figures.py} "$work/arxiv/anc/"

# The source has to compile from these files alone, to the same number of pages as the
# committed PDF, with no undefined reference and no overfull box.
cp -r "$work/tex" "$work/build"
(
  cd "$work/build"
  for pass in 1 2 3; do
    pdflatex -interaction=nonstopmode -halt-on-error uv-finite-brane-gas.tex > /dev/null
  done
  pages() { pdfinfo "$1" | awk '/^Pages:/ {print $2}'; }
  test "$(pages uv-finite-brane-gas.pdf)" = "$(pages "$src/uv-finite-brane-gas.pdf")"
  if grep -iE 'undefined|overfull' uv-finite-brane-gas.log; then exit 1; fi
)

python3 "$root/.github/release/figures.py" "$work/build" "$src/figs" "$version" "$work/figures"

find "$work/tex" "$work/arxiv" "$work/figures" -exec touch -h -d "$stamp" {} +
cp "$src/uv-finite-brane-gas.pdf" "$out/$name.pdf"
(cd "$work/tex" && find . -type f | LC_ALL=C sort | sed 's|^\./||' | TZ=UTC zip -X -q -9 "$out/$name-tex.zip" -@)
(cd "$work/arxiv" && tar --sort=name --mtime="$stamp" --owner=0 --group=0 --numeric-owner -cf - \
  uv-finite-brane-gas.tex figs anc | gzip -n -9 > "$out/$name-arxiv.tar.gz")
(cd "$work" && find figures -type f | LC_ALL=C sort | TZ=UTC zip -X -q -9 "$out/$name-figures.zip" -@)
ls -l "$out"
