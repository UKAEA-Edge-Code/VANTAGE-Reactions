include(GoogleTest)

set(TEST_CHARGE_EXCHANGE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/ChargeExchange)
set(TEST_CROSS_SECTION_DIR ${CMAKE_CURRENT_SOURCE_DIR}/CrossSections)
set(TEST_DATA_CALCULATOR_DIR ${CMAKE_CURRENT_SOURCE_DIR}/DataCalculator)
set(TEST_EXAMPLES_DIR ${CMAKE_CURRENT_SOURCE_DIR}/Examples)
set(TEST_IONISATION_REACTION_DIR ${CMAKE_CURRENT_SOURCE_DIR}/IonisationReaction)
set(TEST_LINEAR_REACTION_BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/LinearReactionBase)
set(TEST_MERGE_TRANSFORMATION_STRATEGY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/MergeTransformationStrategy)
set(TEST_PARTICLE_SPEC_BUILDER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/ParticleSpecBuilder)
set(TEST_PROPERTIES_DIR ${CMAKE_CURRENT_SOURCE_DIR}/Properties)
set(TEST_REACTION_CONTROLLER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/ReactionController)
set(TEST_REACTION_DATA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/ReactionData)
set(TEST_RECOMBINATION_DIR ${CMAKE_CURRENT_SOURCE_DIR}/Recombination)
set(TEST_SPECIES_DIR ${CMAKE_CURRENT_SOURCE_DIR}/Species)
set(TEST_TRANSFORMATION_WRAPPER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/TransformationWrapper)
set(TEST_VELOCITY_SAMPLING_DIR ${CMAKE_CURRENT_SOURCE_DIR}/VelocitySampling)

set(TEST_DIRS
    ${TEST_CHARGE_EXCHANGE_DIR}
    ${TEST_CROSS_SECTION_DIR}
    ${TEST_DATA_CALCULATOR_DIR}
    ${TEST_EXAMPLES_DIR}
    ${TEST_IONISATION_REACTION_DIR}
    ${TEST_LINEAR_REACTION_BASE_DIR}
    ${TEST_MERGE_TRANSFORMATION_STRATEGY_DIR}
    ${TEST_PARTICLE_SPEC_BUILDER_DIR}
    ${TEST_PROPERTIES_DIR}
    ${TEST_REACTION_CONTROLLER_DIR}
    ${TEST_REACTION_DATA_DIR}
    ${TEST_RECOMBINATION_DIR}
    ${TEST_SPECIES_DIR}
    ${TEST_TRANSFORMATION_WRAPPER_DIR}
    ${TEST_VELOCITY_SAMPLING_DIR}
)

set(EXECUTABLE testReactions)
set(TEST_MAIN ${CMAKE_CURRENT_SOURCE_DIR}/Main.cpp)

foreach(TEST_DIR ${TEST_DIRS})
    file(GLOB_RECURSE src_unit "${TEST_DIR}/*.cpp")
    get_filename_component(TEST_NAME ${TEST_DIR} NAME)
    message(STATUS "Found test - ${TEST_NAME}")
    
    set(TEST_NAME_LIST ${TEST_NAME_LIST} ${TEST_NAME})
    set(TEST_LIST ${TEST_LIST} ${src_unit})
    set(TEST_SOURCES ${TEST_MAIN} ${src_unit})

    if($ENV{SINGLE_TEST_NAME} STREQUAL ${TEST_NAME} OR $ENV{SINGLE_TEST_NAME} STREQUAL "FULL") 
        message("Building ${TEST_NAME}...")

        add_executable(${TEST_NAME} ${TEST_SOURCES})

        target_link_libraries(${TEST_NAME} PRIVATE VANTAGE::Reactions NESO-Particles::NESO-Particles GTest::gtest)
        target_compile_definitions(${TEST_NAME} PRIVATE GPU_SELECTOR=0)
        target_compile_options(${TEST_NAME} PRIVATE ${BUILD_TYPE_COMPILE_FLAGS} ${TEST_COVERAGE_FLAGS})
        SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${TEST_COVERAGE_FLAGS}")
    
        add_sycl_to_target(TARGET ${TEST_NAME} SOURCES ${src_unit})
        gtest_add_tests(TARGET ${TEST_NAME} SOURCES ${src_unit})

        install(TARGETS ${TEST_NAME} DESTINATION bin)
    endif()
    unset(src_unit)
endforeach()

if($ENV{SINGLE_TEST_NAME} STREQUAL "ALL" OR $ENV{SINGLE_TEST_NAME} STREQUAL "FULL")
    message("Building all tests...")
    add_executable(${EXECUTABLE} ${TEST_MAIN} ${TEST_LIST})

target_link_libraries(${EXECUTABLE} PRIVATE VANTAGE::Reactions NESO-Particles::NESO-Particles GTest::gtest)
target_compile_definitions(${EXECUTABLE} PRIVATE GPU_SELECTOR=0)
target_compile_options(${EXECUTABLE} PRIVATE ${BUILD_TYPE_COMPILE_FLAGS})
SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
add_sycl_to_target(TARGET ${EXECUTABLE} SOURCES ${src_unit})
gtest_add_tests(TARGET ${EXECUTABLE} SOURCES ${src_unit})

    install(TARGETS ${EXECUTABLE} DESTINATION bin)
else()
    set(VALID_SINGLE_TEST FALSE)
    foreach(ITEST_NAME ${TEST_NAME_LIST})
        if($ENV{SINGLE_TEST_NAME} STREQUAL ${ITEST_NAME})
            unset(VALID_SINGLE_TEST)
            set(VALID_SINGLE_TEST TRUE)
        endif()
    endforeach()
    if(NOT ${VALID_SINGLE_TEST})
        message(FATAL_ERROR "Please pass a valid value for SINGLE_TEST_NAME (either ALL or a name of a test binary).")
    endif()
endif()
