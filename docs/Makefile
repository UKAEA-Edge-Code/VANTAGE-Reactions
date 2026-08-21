all:
	rm -rf ./build
	mkdir ./build
	$(MAKE) -C Doxygen docs_doxygen
	cd sphinx && make html
