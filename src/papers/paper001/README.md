# paper001

## Install

Clone the repository and enter this directory:

```
git clone https://github.com/eoinmurray/pinglab.git
cd pinglab/src/papers/paper001
```

`latexmk` ships with every TeX distribution — install one and you have it. The
manuscript uses the eLife article class, vendored here as `elife.cls` (with
`vancouver-elife.bst`), which pulls a number of LaTeX packages; a **full** TeX
Live / MacTeX has them all. Compile with `pdflatex` (the default below).

**macOS** — full TeX Live (~4 GB, has everything; simplest):

```
brew install --cask mactex
```

After a cask install, open a new shell so `/Library/TeX/texbin` is on `PATH`.

**Any platform** — install [TeX Live](https://tug.org/texlive/), which bundles `latexmk`.

Verify:

```
latexmk --version
```

## Build

```
latexmk -pdf main.tex      # build main.pdf
latexmk -C                 # clean all build artifacts
```

## Watch

```
latexmk -pvc -view=none -pdf main.tex
```

## Figures

Figures are copied from the notebook outputs so the build is self-contained:

```
sh copy-figures.sh
```
