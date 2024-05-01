all: train_networks;
	$(MAKE) pdf_without_training

all_mac: train_networks;
	$(MAKE) pdf_without_training_mac

arxiv_version_without_training: arxiv_version_without_training_and_compilation
	$(MAKE) compile_with_pdflatex;
	$(MAKE) compile_with_pdflatex;
	rm build/*.log;
	rm build/*.out;
	rm build/*.aux;
	rm build/*.pdf;
	rm -r build/_minted-*;
	rm build/.DS*;
	rm build/*.jl;
	sed -i'' -e 's/\\usepackage{minted}/\\usepackage[frozencache=true,cachedir=minted-cache]{minted}/g' documenter_custom.sty;
	zip -r arxiv_submission.zip build

arxiv_version_without_training_and_compilation: tikz_images tex_without_training_and_images
	cp src/assets/preamble_arxiv.tex build/preamble.tex;
	sed -i'' -e 's/\\begin{figure}/\\begin{figure}\\centering/g' build/V*.tex;
	sed -i'' -e 's/\\begin{tabulary}/\\footnotesize\\begin{tabulary}/g' build/V*.tex;
	mv -v build/tikz/*.png build/;
	mv -v build/simulations/vpt_Float32/*.png build/;
	sed -i'' -e 's/tikz\///g' build/V*.tex;
	sed -i'' -e 's/simulations\/vpt_Float32\///g' build/V*.tex;
	rm -r build/tikz;
	rm -r build/simulations;
	rm build/edpsmath.cls;
	rm build/edpsproc.clo;
	rm -r build/assets;
	rm -r build/*.tex-e;
	rm -r build/*.bib

pdf_without_training: tikz_images pdf_without_training_and_images;

pdf_without_training_and_images: tex_without_training_and_images;
	$(MAKE) compile_tex;
	$(MAKE) compile_tex

tex_without_training_and_images: generate_tex;
	$(MAKE) copy_edps_files chapter_to_section introduction_gets_no_label use_acknowledgements_environment make_tex_references give_equations_labels do_correct_quotation_marks do_correct_footnotes do_correct_references_to_section do_correct_remarks

tikz_images:
	$(MAKE) linux -C src/tikz

copy_edps_files:
	cp src/assets/edpsmath.cls build/edpsmath.cls;
	cp src/assets/edpsproc.clo build/edpsproc.clo;
	cp src/assets/documenter.sty build/documenter_custom.sty

generate_tex:
	julia --project --threads=8 make.jl no_pdf

compile_tex: 
	cd build; \
	xelatex -shell-escape V*.tex 

compile_with_pdflatex:
	cd build; \
	pdflatex -shell-escape V*.tex

train_networks:
	julia --project --threads=8 src/volume_preserving_transformer.jl

chapter_to_section:
	sed -i'' -e 's/\\section{/\\subsection{/g' build/V*.tex;
	sed -i'' -e 's/\\chapter{/\\section{/g' build/V*.tex;
	sed -i'' -e '/\\part{/d' build/V*.tex

make_tex_references:
	sed -i'' -e 's/m\[\([^ ]*\)\]m(@latex)/\\cref{\1}/g' build/V*.tex;
	sed -i'' -e 's/M\[\([^ ]*\)\]m(@latex)/\\Cref{\1}/g' build/V*.tex

introduction_gets_no_label:
	sed -i'' -e 's/\\section{Introduction}/\\section*{Introduction}/g' build/V*.tex

use_acknowledgements_environment:
	sed -i'' -e 's/\\section{Acknowledgements}/\\begin{acknowledgement}/g' build/V*.tex;
	sed -i'' -e 's/\\appendix/\\end{acknowledgement}\\appendix/g' build/V*.tex

give_equations_labels:
	sed -i'' -e 's/equation\*/equation/g' build/V*.tex;
	sed -i'' -e 's/\\label{\(.*\)}\\end{split}/\\end{split}\\label{\1}/g' build/V*.tex

do_correct_quotation_marks:
	sed -i'' -e 's/{\\textquotedbl}/"/g' build/V*.tex;
	sed -i'' -e 's/ "/ ``/g' build/V*.tex

do_correct_footnotes:
	sed -i'' -e 's/\\footnotemark\[[0-9]*\]/\\footnotemark/g' build/V*.tex;
	sed -i'' -e 's/\\footnotetext\[[0-9]*\]/\\footnotetext/g' build/V*.tex;
	sed -i'' -e 's/\\footnotemark /\\footnotemark{} /g' build/V*.tex

do_correct_references_to_section:
	sed -i'' -e 's/\\hyperlinkref{\([0-9][0-9]*\)}{[^0-9][^0-9.]*}/\\Cref{\1}/g' build/V*.tex

do_correct_remarks:
	sed -i'' -e 's/REMARK::\(.*\)::/\\begin{rmrk}\1\\end{rmrk}/g' build/V*.tex;
	sed -i'' -e 's/DEFINITION::\(.*\)::/\\begin{dfntn}\1\\end{dfntn}/g' build/V*.tex

clear: 
	$(MAKE) empty -C src/tikz;
	rm -r -f build;
	rm arxiv_submission.zip;