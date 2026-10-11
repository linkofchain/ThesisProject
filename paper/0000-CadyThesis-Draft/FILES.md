# LIST OF FILES

## Build dependencies
Compiles with xelatex (required for fontspec/IPA fonts) + biber + makeglossaries.
On Ubuntu/Debian, a fresh machine needs:
```
sudo apt-get install -y texlive-xetex texlive-fonts-extra texlive-bibtex-extra biber latexmk
```
(`texlive-latex-recommended`/`texlive-latex-extra` cover most other packages used, e.g. tikz, pgfplots, glossaries, hyperref.)
Also requires the Doulos SIL font on the system (used for IPA transcriptions via `\ipa{}`), e.g. `fonts-sil-doulos` or equivalent.

## main.tex
The start of a new thesis.

## main.bib
The bibliography for it.

## palthesis.tex
A short example thesis, which approximately the same contents as example palindrome2.tex elsewhere – see https://www.overleaf.com/read/hfqxgcjssvxx .
Not need for your thesis.

## pal.bib, pal.py
Used by palthesis.tex; not needed for your thesis.

## cluu.cls
The definition of the CLUU class.

## cluu.bbx, cluu.cbx
Needed for the CLUU style for bibliographies.

## UU_*.pdf
Images needed for the cover of the theses.
