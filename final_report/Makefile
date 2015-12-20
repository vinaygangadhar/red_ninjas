PAPER = main
TEX = $(wildcard *.tex)
BIB = main.bib
FIGS = $(wildcard figs/*.pdf figs/*.png graphs/*.pdf graphs/*.png, tables/*.pdf tables/*.png tables/*.tex)

.PHONY: all clean

$(PAPER).pdf: $(TEX) $(BIB) $(FIGS) jpaper.cls 
	echo $(FIGS)
	pdflatex $(PAPER)
	bibtex $(PAPER)
	pdflatex $(PAPER)
	pdflatex $(PAPER)

clean:
	rm -f *.aux *.bbl *.blg *.log *.out $(PAPER).pdf

