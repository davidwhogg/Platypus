LATEX       = pdflatex
CHECK_RERUN = grep Rerun $*.log

all: clusters.pdf melissa_atlas.pdf

melissa_atlas.pdf: melissa_atlas.tex
	${LATEX} $<

%.pdf: %.tex
	${LATEX} $<
	${LATEX} $<
	${LATEX} $<
