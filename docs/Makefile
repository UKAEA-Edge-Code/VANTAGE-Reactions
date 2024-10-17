all:
	rm -rf ./build
	mkdir ./build	
	bash get_example_sources.sh
	cd Doxygen && make 
	cd sphinx && make html
