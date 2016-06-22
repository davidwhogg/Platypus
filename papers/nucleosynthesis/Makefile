LATEX       = pdflatex
CHECK_RERUN = grep Rerun $*.log

all: nuke.pdf diversity.pdf

%.pdf: %.tex
	${LATEX} $<
	${LATEX} $<
	${LATEX} $<
