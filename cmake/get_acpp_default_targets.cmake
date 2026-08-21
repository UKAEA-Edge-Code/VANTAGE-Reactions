# This just gets the targets for AdaptiveCpp (eg. cuda-nvcxx)
function(get_acpp_default_targets ACPP_COMPILER ACPP_DEFAULT_TARGETS)
    # Get one directory up
    get_filename_component(PARENT_DIR "${ACPP_COMPILER}" PATH)

    # Get one more directory up (removing last two components)
    get_filename_component(GRANDPARENT_DIR "${PARENT_DIR}" PATH)

    message(STATUS "AdaptiveCpp Root directory: ${GRANDPARENT_DIR}")

    file(READ "${GRANDPARENT_DIR}/etc/AdaptiveCpp/acpp-core.json" acpp_core_json)

    string(JSON acpp_default_targets GET "${acpp_core_json}" "default-targets")

    set(${ACPP_DEFAULT_TARGETS} ${acpp_default_targets} PARENT_SCOPE)
endfunction()