#!/bin/sh
# Build every figure: pdflatex on the standalone sources, then PNG at 300 dpi.
set -e
cd "$(dirname "$0")"
for f in fig_*.tex; do
  b=${f%.tex}
  pdflatex -interaction=nonstopmode -halt-on-error "$f" > /dev/null
  pdftoppm -r 300 -png -singlefile "$b.pdf" "../figures/$b"
done
rm -f *.aux *.log
