# import of package / builder classes
import os
import shutil

# import Package API
from spack.package import *
from spack_repo.builtin.build_systems.cmake import CMakePackage


class Vantagereactions(CMakePackage):
    """Vantagereactions"""

    git = "git@github.com:UKAEA-Edge-Code/VANTAGE-Reactions.git"

    version("main", branch="main")
    version("working", branch="reactions-base", preferred=True)

    variant("enable_tests", default=False, description="Enable tests")
    variant(
        "header_only",
        default=False,
        description=(
            "Build as a header-only INTERFACE library instead of a compiled "
            "SHARED/STATIC library (maps to "
            "-DVANTAGE_REACTIONS_HEADER_ONLY=ON). The compiled build is the "
            "default; opt into this only if you need the legacy header-only "
            "behaviour."
        ),
    )
    variant(
        "tests_split",
        default=False,
        description=(
            "Build one executable per test_*.cpp instead of the single "
            "unit_tests monolith (maps to -DREACTIONS_TESTS_SPLIT=ON). "
            "Off by default; the monolith is what CI and run_tests.sh use. "
            "Only meaningful with +enable_tests."
        ),
    )
    variant(
        "test_filter",
        default="all",
        values=any,
        multi=True,
        description=(
            "When +tests_split, build only these test_*.cpp stems. 'all' "
            "(the default) builds every test_*.cpp. Otherwise one or more "
            "stems, e.g. test_filter=test_reaction_controller or "
            "test_filter=test_properties,test_species. Maps to "
            "-DREACTIONS_TEST_FILTER. An unknown stem fails at configure time."
        ),
    )

    depends_on("c")
    depends_on("cxx")
    depends_on("mpi", type=("build", "link", "run"))
    depends_on("neso-particles", type=("build", "link", "run"))
    depends_on("sycl", type=("build", "link", "run"))
    depends_on("googletest", type=("build", "link", "run"))

    # tests_split is only meaningful when tests are built.
    conflicts("+tests_split", when="~enable_tests")

    def cmake_args(self):
        args = []
        args.append(self.define_from_variant("REACTIONS_ENABLE_TESTS", "enable_tests"))
        args.append(self.define_from_variant("VANTAGE_REACTIONS_HEADER_ONLY", "header_only"))
        args.append(self.define_from_variant("REACTIONS_TESTS_SPLIT", "tests_split"))
        # test_filter is a multi-valued variant of test_*.cpp stems; 'all'
        # (the default) means no filter (build every test_*.cpp). CMake wants a
        # semicolon list, so join the spack variant values.
        test_filter = self.spec.variants["test_filter"].value
        if test_filter and "all" not in test_filter:
            args.append(self.define("REACTIONS_TEST_FILTER", ";".join(test_filter)))
        return args
