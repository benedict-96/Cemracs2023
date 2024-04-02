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
	$(MAKE) copy_edps_files chapter_to_section introduction_gets_no_label make_tex_references give_equations_labels

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

give_equations_labels:
	sed -i'' -e 's/equation\*/equation/g' build/V*.tex;
	sed -i'' -e 's/\\label{\(.*\)}\\end{split}/\\end{split}\\label{\1}/g' build/V*.tex

clear: 
	$(MAKE) empty -C src/tikz;
	rm -r -f build;