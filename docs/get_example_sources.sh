EXAMPLE_SOURCES="../test/unit/example_sources/"
EXAMPLE_OUTPUTS="sphinx/source/example_sources"

mkdir -p $EXAMPLE_OUTPUTS

for fx in $EXAMPLE_SOURCES/*.hpp; do 
    echo $fx
    echo $(basename -- $fx)
    echo $EXAMPLE_OUTPUTS/$(basename -- $fx)
    cat $fx > $EXAMPLE_OUTPUTS/$(basename -- $fx)
done
