# import of package / builder classes
import os
import shutil

# import Package API
from spack.package import *
from spack_repo.builtin.build_systems.cmake import CMakePackage


class Reactions(CMakePackage):
    """Reactions"""

    git = "git@github.com:UKAEA-Edge-Code/Reactions.git"

    version("main", branch="main")
    version("working", branch="reactions-base", preferred=True)

    variant("enable_tests", default=False, description="Enable tests")

    depends_on("mpi", type=("build", "link", "run"))
    depends_on("neso-particles", type=("build", "link", "run"))
    depends_on("sycl", type=("build", "link", "run"))
    depends_on("googletest", type=("build", "link", "run"))

    # requires(
    #     "%gcc", "%clang",
    #     policy="one_of",
    #     msg="Reactions builds with only gcc or clang."
    # )

    def cmake_args(self):
        args = []
        args.append(self.define_from_variant("REACTIONS_ENABLE_TESTS", "enable_tests"))

        return args
