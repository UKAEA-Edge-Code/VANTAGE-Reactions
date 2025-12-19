from spack import *
import os
import shutil


class Reactions(CMakePackage):
    """Reactions"""

    git = "git@github.com:UKAEA-Edge-Code/Reactions.git"

    version("main", branch="main")
    version("working", branch="reactions-base", preferred=True)

    variant("nvcxx", default=False, description="Builds with CUDA CMake flags.")
    variant("enable_tests", default=False, description="Enable tests")
    variant("enable_coverage", default=False, description="Enable coverage analysis")

    depends_on("mpi", type=("build", "link", "run"))
    depends_on("neso-particles", type=("build", "link", "run"))
    depends_on("sycl", type=("build", "link", "run"))
    depends_on("googletest", type=("build", "link", "run"))

    conflicts("+nvcxx", when="%oneapi", msg="Nvidia compilation can only be used with gcc or clang compilers.")

    def cmake_args(self):
        args = []
        args.append(self.define_from_variant("REACTIONS_ENABLE_TESTS", "enable_tests"))
        args.append(self.define_from_variant("REACTIONS_ENABLE_COVERAGE", "enable_coverage"))

        return args
