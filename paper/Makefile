all: train_networks;
	$(MAKE) pdf_without_training

all_mac: train_networks;
	$(MAKE) pdf_without_training_mac

pdf_without_training: linux_images pdf_without_training_and_images;
	$(MAKE) compile_tex;
	$(MAKE) compile_tex

pdf_without_training_mac: mac_images pdf_without_training_and_images;
	$(MAKE) compile_tex;
	$(MAKE) compile_tex

pdf_without_training_and_images: generate_tex;
	$(MAKE) copy_edps_files chapter_to_section introduction_gets_no_label use_acknowledgements_environment make_tex_references give_equations_labels do_correct_quotation_marks do_correct_footnotes do_correct_references_to_section do_correct_remarks

linux_images:
	$(MAKE) linux -C src/tikz

mac_images:
	$(MAKE) mac -C src/tikz

copy_edps_files:
	cp src/assets/edpsmath.cls build/edpsmath.cls;
	cp src/assets/edpsproc.clo build/edpsproc.clo;
	cp src/assets/documenter.sty build/documenter_custom.sty

generate_tex:
	julia --project --threads=8 make.jl no_pdf

compile_tex: 
	cd build; \
	xelatex -shell-escape V*.tex 

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
	sed -i'' -e 's/\\section{References}/\\end{acknowledgement}\\section{References}/g' build/V*.tex

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
	sed -i'' -e 's/\\hyperlinkref{\([0-9][0-9]*\)}{[^0-9][^0-9]*}/\\cref{\1}/g' build/V*.tex

do_correct_remarks:
	sed -i'' -e 's/REMARK::\(.*\)::/\\begin{rmrk}\1\\end{rmrk}/g' build/V*.tex;
	sed -i'' -e 's/DEFINITION::\(.*\)::/\\begin{dfntn}\1\\end{dfntn}/g' build/V*.tex

clear: 
	$(MAKE) empty -C src/tikz;
	rm -r -f build;