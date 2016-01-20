LATEX       = pdflatex
CHECK_RERUN = grep Rerun $*.log

all: clusters.pdf

%.pdf: %.tex
	${LATEX} $<
	${LATEX} $<
	${LATEX} $<
