Find_package(Threads REQUIRED)
find_package(GTest)

if(NOT GTest_FOUND)
  include(FetchContent)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/054a986a8513149e8374fc669a5fe40117ca6b41.zip
        DOWNLOAD_EXTRACT_TIMESTAMP
        FALSE)
  # For Windows: Prevent overriding the parent project's compiler/linker
  # settings
  set(gtest_force_shared_crt
      ON
      CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
endif()

# Set up to generate coverage reports
if (NOT REACTIONS_DEVICE_TYPE STREQUAL GPU)
  find_package(PythonInterp REQUIRED)
  find_program(GCOVR_PATH gcovr PATHS ${CMAKE_SOURCE_DIR}/scripts/test)

  if(BUILD_TYPE STREQUAL "TEST")
    add_custom_target(gcovr
      # Create folder
      COMMAND ${CMAKE_COMMAND} -E make_directory ${PROJECT_BINARY_DIR}/gcovr

      # Running gcovr
      COMMAND ${Python_EXECUTABLE} ${GCOVR_PATH} --html --html-details -s
              -r ${PROJECT_SOURCE_DIR} --object-directory=${PROJECT_BINARY_DIR}
              -o gcovr/index.html
      WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
      COMMENT "Running gcovr to produce HTML code coverage report."
    )
    # Show info where to find the report
    add_custom_command(TARGET gcovr POST_BUILD
      COMMAND ;
      COMMENT "Open ./gcovr/index.html in your browser to view the full coverage report."
    )
  endif()
endif()

add_subdirectory(unit)

