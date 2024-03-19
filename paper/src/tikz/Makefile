all: linux

linux: pdf 
	$(MAKE) convert_with_pdftocairo 
	$(MAKE) clean

mac: pdf 
	$(MAKE) convert_with_sips 
	$(MAKE) clean

MYDIR = .
pdf: $(MYDIR)/*.tex
	for file in $^ ; do \
		xelatex -shell-escape $${file} ; \
	done

# this is converting pdfs to pngs using sips (mac version)
convert_with_sips: $(MYDIR)/*.pdf
	for file in $^ ; do \
		sips --setProperty format png --resampleHeightWidthMax 2000 $${file} --out $${file%.*}.png ; \
	done

# this is converting pdfs to pngs using pdftocairo (linux version)
convert_with_pdftocairo: $(MYDIR)/*.pdf
	for file in $^ ; do \
		pdftocairo -png -r 500 -transp -singlefile $${file} $${file%.*} ; \
	done

clean:
	rm -f *.aux
	rm -f *.fdb_latexmk
	rm -f *.fls
	rm -f *.log
	rm -f *.xdv
	rm -f *.pdf

empty: clean
	rm -f *.png
	rm -f *.svg