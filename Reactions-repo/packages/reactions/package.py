from spack import *
import os
import shutil


class Reactions(CMakePackage):
    """Reactions"""

    git = "git@github.com:UKAEA-Edge-Code/Reactions.git"

    version("main", branch="main")
    version("working", branch="reactions-base", preferred=True)

    variant("nvcxx", default=False, description="Builds with CUDA CMake flags.")
    variant("test", default=False, description="Enable tests.")

    depends_on("mpi", type=("build", "link", "run"))
    depends_on("neso-particles", type=("build", "link", "run"))
    depends_on("sycl", type=("build", "link", "run"))
    depends_on("googletest", type=("build", "link", "run"))

    def setup_build_environment(self, env):
        if "+test" in self.spec:
            env.set("BUILD_TYPE", "TEST")
        else:
            env.set("BUILD_TYPE", "RELEASE")

    def cmake_args(self):
        args = []
        if "+nvcxx" in self.spec:
            args.append("-DNESO_PARTICLES_DEVICE_TYPE=GPU")
            args.append("-DHIPSYCL_TARGETS=cuda-nvcxx")
        elif "~nvcxx" in self.spec:
            args.append("-DNESO_PARTICLES_DEVICE_TYPE=CPU")
            args.append("-DHIPSYCL_TARGETS=omp")
        
        return args


