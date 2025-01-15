from spack import *
import os
import shutil


class Reactions(CMakePackage):
    """Reactions"""

    git = "git@github.com:UKAEA-Edge-Code/Reactions.git"

    version("main", branch="main")
    version("working", branch="reactions-base", preferred=True)

    variant("nvcxx", default=False, description="Builds with CUDA CMake flags.")
    variant("build_option", default="RELEASE", description="CMake build type", values=("DEBUG", "RELEASE", "TEST"))

    depends_on("mpi", type=("build", "link", "run"))
    depends_on("neso-particles", type=("build", "link", "run"))
    depends_on("sycl", type=("build", "link", "run"))
    depends_on("googletest", type=("build", "link", "run"))

    def cmake_args(self):
        args = []
        args.append(self.define("BUILD_TYPE", self.spec.variants["build_option"].value))
        if "+nvcxx" in self.spec:
            args.append("-DNESO_PARTICLES_DEVICE_TYPE=GPU")
            args.append("-DHIPSYCL_TARGETS=cuda-nvcxx")
        elif "~nvcxx" in self.spec:
            args.append("-DNESO_PARTICLES_DEVICE_TYPE=CPU")
            args.append("-DHIPSYCL_TARGETS=omp")

        return args


