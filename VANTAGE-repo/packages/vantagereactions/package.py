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

    depends_on("c")
    depends_on("cxx")
    depends_on("mpi", type=("build", "link", "run"))
    depends_on("neso-particles", type=("build", "link", "run"))
    depends_on("sycl", type=("build", "link", "run"))
    depends_on("googletest", type=("build", "link", "run"))

    def cmake_args(self):
        args = []
        args.append(self.define_from_variant("REACTIONS_ENABLE_TESTS", "enable_tests"))
        args.append(self.define_from_variant("VANTAGE_REACTIONS_HEADER_ONLY", "header_only"))

        return args
