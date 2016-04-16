LATEX       = pdflatex
CHECK_RERUN = grep "Rerun to get" $*.log

all: kmeans.pdf

melissa_atlas.pdf: melissa_atlas.tex
	${LATEX} $<

%.pdf: %.tex
	${LATEX} $<
	( ${CHECK_RERUN} && ${LATEX} $< ) || echo "Done."
	( ${CHECK_RERUN} && ${LATEX} $< ) || echo "Done."
