all: linux_images train_networks generate_tex copy_edps_files chapter_to_section introduction_gets_no_label compile_tex

linux_images:
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

train_networks:
	julia --project --threads=8 src/volume_preserving_feed_forward.jl;
	julia --project --threads=8 src/volume_preserving_transformer.jl

chapter_to_section:
	sed -i 's/\\chapter{/\\section{/g' build/V*.tex;
	sed -i '/\\part{/d' build/*.tex

introduction_gets_no_label:
	sed -i 's/\\section{Introduction}/\\section*{Introduction}/g' build/V*.tex

clear: 
	$(MAKE) empty -C src/tikz;
	rm -r -f build;