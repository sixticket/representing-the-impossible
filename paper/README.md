# Paper

This directory contains a self-contained Springer Nature / Nature Portfolio
LaTeX draft. The template files are from the official December 2024 Springer
Nature authoring package.

Compile with:

```bash
tectonic main.tex --keep-logs --keep-intermediates
```

or:

```bash
latexmk -pdf main.tex
```

or:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The manuscript figures are copied into this directory so that the source bundle
can be uploaded without depending on paths outside `paper/`.
