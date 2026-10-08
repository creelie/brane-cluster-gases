"""Render every figure of the UV-finite paper on its own page, as a PDF and a 400 dpi PNG.

Usage: python3 figures.py <build dir> <panels dir> <version> <output dir>

<build dir> holds uv-finite-brane-gas.tex compiled once, so that its .aux gives the figure,
equation and section numbers. Each figure is set by pdflatex in a standalone document with the
paper's preamble and its text and column widths, without the caption. The output directory gets
FigNN-<label>.pdf and .png for every figure, the plotted panels in panels/, and a README.md with
all captions.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

build, panels, version, out = sys.argv[1:5]
src = open(f"{build}/uv-finite-brane-gas.tex", encoding="utf-8").read()
aux = open(f"{build}/uv-finite-brane-gas.aux", encoding="utf-8").read()
num = dict(re.findall(r"\\newlabel\{([^}]*)\}\{\{([^}]*)\}", aux))

TW, CW = "510.0pt", "246.0pt"  # revtex4-2 with aps, prd, 10pt, twocolumn
KIND = {"logic": "TikZ", "cells": "TikZ", "finegrain": "TikZ", "diagrams": "TikZ",
        "chain": "TikZ and plot", "unitarity": "plot and TikZ", "bubble": "plots and TikZ"}
MACROS = {r"\bE": r"\mathbb{E}", r"\bC": r"\mathbb{C}", r"\bR": r"\mathbb{R}",
          r"\Ein": r"\operatorname{Ein}", r"\sgn": r"\operatorname{sgn}",
          r"\gE": r"\gamma_{\mathrm E}", r"\dd": r"\mathrm{d}", r"\Tr": r"\operatorname{Tr}"}

# Preamble of the paper, without what only revtex needs.
pre = src[src.index("\n", src.index("\\documentclass")) + 1 : src.index("\\begin{document}")]
pre = pre.replace("\\usepackage{orcidlink}\n", "")
pre = pre.replace("\\hypersetup{", "\\usepackage{hyperref}\n\\hypersetup{", 1)
pre = re.sub(r"\\newcolumntype\{L\}.*\n", "", pre)
labels = "\n".join(l for l in aux.splitlines() if l.startswith("\\newlabel{"))

figs = []
for m in re.finditer(r"\\begin\{(figure\*?)\}(\[[^\]]*\])?(.*?)\\end\{\1\}", src, re.S):
    env, body = m.group(1), m.group(3)
    start = body.index("\\caption{") + len("\\caption{")
    i, depth = start, 1
    while depth:
        depth += {"{": 1, "}": -1}.get(body[i], 0)
        i += 1
    figs.append({"wide": env == "figure*",
                 "label": re.search(r"\\label\{fig:([^}]*)\}", body).group(1),
                 "content": body[: start - len("\\caption{")].replace("\\centering", "", 1).strip(),
                 "caption": body[start : i - 1]})


def markdown(caption):
    """The caption with the paper's numbers filled in, for GitHub Markdown with math."""
    parts = caption.split("$")
    for i, p in enumerate(parts):
        if i % 2:
            for k, v in MACROS.items():
                p = re.sub(re.escape(k) + r"(?![A-Za-z])", lambda m, v=v: v, p)
        else:
            p = re.sub(r"\\eqref\{([^}]*)\}", lambda m: f"({num[m.group(1)]})", p)
            p = re.sub(r"\\ref\{([^}]*)\}", lambda m: num[m.group(1)], p)
            p = p.replace("\\,", " ").replace("~", " ").replace("--", "\u2013")
        parts[i] = p
    return "$".join(parts)


os.makedirs(f"{out}/panels", exist_ok=True)
work = tempfile.mkdtemp()
shutil.copytree(panels, f"{work}/figs")
open(f"{work}/labels.tex", "w", encoding="utf-8").write(
    "\\makeatletter\n" + labels + "\n\\makeatother\n")
rows, caps = [], []
for k, f in enumerate(figs, 1):
    name = f"Fig{k:02d}-{f['label']}"
    width = TW if f["wide"] else CW
    doc = "\n".join([
        "\\documentclass[border=6pt]{standalone}", pre.rstrip(), "\\input{labels.tex}",
        "\\begin{document}",
        f"\\setlength{{\\textwidth}}{{{TW}}}\\setlength{{\\columnwidth}}{{{CW}}}\\setlength{{\\linewidth}}{{{width}}}%",
        f"\\begin{{minipage}}{{{width}}}\\centering", f["content"], "\\end{minipage}",
        "\\end{document}", ""])
    open(f"{work}/{name}.tex", "w", encoding="utf-8").write(doc)
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f"{name}.tex"],
                   cwd=work, check=True, stdout=subprocess.DEVNULL)
    shutil.copy2(f"{work}/{name}.pdf", f"{out}/{name}.pdf")
    subprocess.run(["pdftoppm", "-r", "400", "-png", "-singlefile", f"{out}/{name}.pdf",
                    f"{out}/{name}"], check=True)
    rows.append(f"| {k} | `{name}.pdf`, `{name}.png` | {KIND.get(f['label'], 'plot')} |")
    caps.append(f"### Fig. {k}\n\n{markdown(f['caption'])}\n")

for p in sorted(os.listdir(panels)):
    shutil.copy2(f"{panels}/{p}", f"{out}/panels/{p}")

open(f"{out}/README.md", "w", encoding="utf-8").write(f"""\
# Figures: Finite quantum gravity from a scale-invariant gas of branes

Deep Bhattacharjee. Release {version} of
[creelie/brane-cluster-gases](https://github.com/creelie/brane-cluster-gases).

Each of the {len(figs)} figures is given as it appears in the paper, without its caption: a
vector PDF and a PNG at 400 dpi. Equation, section and table numbers inside the figures are
those of the paper. The folder `panels/` holds the {len(os.listdir(panels))} plotted panels that the LaTeX source
includes; `uv-finite/code/make_figures.py` in the repository regenerates them.

| Fig. | Files | Kind |
|---|---|---|
""" + "\n".join(rows) + "\n\n## Captions\n\n" + "\n".join(caps))
json.dump([f["label"] for f in figs], sys.stdout)
print()
