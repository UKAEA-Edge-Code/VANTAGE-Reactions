from spack import *
import os
import shutil


class NesoParticles(CMakePackage):
    """NESO-Particles"""

    homepage = "https://excalibur-neptune.github.io/NESO-Particles/"

    git = "https://github.com/ExCALIBUR-NEPTUNE/NESO-Particles.git"

    version("0.5.0", commit="ba6750d429fe15bbec9b9c507b795cd3117b79b4", preferred=True)
    version("0.4.0", commit="c615974661e0f4c8d9db709d65d27cc2927bbfaf")
    version("0.3.1", commit="9c6b4626645f6aaaca478e4798f2fdee5dd2675b")
    version("0.3.0", commit="cb55184bb7196c210d8c3f4397f4a29808acd038")
    version("0.2.0", commit="bae6ee91fe6558a0d7eba040dd93db34eda348a8")
    version("0.1.0", commit="2e1b1aac6f4f9c22b31787cf300fcc2b4914cbe1")
    version("working", branch="main")
    version("main", branch="main")

    variant("build_tests", default=True, description="Builds the NESO-Particles tests.")
    variant("nvcxx", default=False, description="Builds with CUDA CMake flags.")

    depends_on("mpi", type=("build", "link", "run"))

    # Non-SYCL dependencies
    depends_on("hdf5 +mpi +hl", type=("build", "link", "run"))
    depends_on("cmake@3.21:", type="build")
    depends_on("cmake@3.24:", type="build", when="@main")
    depends_on("cmake@3.24:", type="build", when="@working")
    depends_on("googletest@1.10.0: +gmock", type=("build", "link", "run"))

    # Depend on a sycl implementation - with workarounds for intel packaging.
    depends_on("sycl", type=("build", "link", "run"))
    depends_on("intel-oneapi-dpl", when="^dpcpp", type="link")
    conflicts("%dpcpp", msg="Use oneapi compilers instead of dpcpp driver.")
    conflicts("^dpcpp", when="%gcc", msg="DPC++ can only be used with Intel oneAPI compilers.")
    conflicts("+nvcxx", when="%oneapi", msg="Nvidia compilation option can only be used with gcc compilers")

    def cmake_args(self):
        args = []
        if not "+build_tests" in self.spec:
            args.append("-DNESO_PARTICLES_ENABLE_TESTS=off")
        if "+nvcxx" in self.spec:
            args.append("-DNESO_PARTICLES_DEVICE_TYPE=GPU")
            args.append("-DHIPSYCL_TARGETS=cuda-nvcxx")

        return args
