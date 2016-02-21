LATEX       = pdflatex
CHECK_RERUN = grep Rerun $*.log

all: gradients.pdf

%.pdf: %.tex
	${LATEX} $<
	${LATEX} $<
	${LATEX} $<
