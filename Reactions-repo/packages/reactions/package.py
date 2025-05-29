from spack import *
import os
import shutil
import glob


class Reactions(CMakePackage):
    """Reactions"""

    git = "git@github.com:UKAEA-Edge-Code/Reactions.git"

    version("main", branch="main")
    version("working", branch="reactions-base", preferred=True)

    variant("nvcxx", default=False, description="Builds with CUDA CMake flags.")
    variant("enable_tests", default=False, description="Enable tests")

    depends_on("mpi", type=("build", "link", "run"))
    depends_on("neso-particles", type=("build", "link", "run"))
    depends_on("sycl", type=("build", "link", "run"))
    depends_on("googletest", type=("build", "link", "run"))

    conflicts("+nvcxx", when="%oneapi", msg="Nvidia compilation can only be used with gcc or clang compilers.")

    requires(
        "%gcc", "%clang",
        policy="one_of",
        msg="Reactions builds with only gcc or clang."
    )

    def cmake_args(self):
        args = []
        args.append(self.define_from_variant("REACTIONS_ENABLE_TESTS", "enable_tests"))
        if "+nvcxx" in self.spec:
            args.append("-DREACTIONS_DEVICE_TYPE=GPU")

        return args

    @run_after("install")
    def clean(self):
        test_bin_dir = os.path.join(self.stage.source_path, "tests_binaries")

        try:
            os.mkdir(test_bin_dir)
        except FileExistsError:
            pass

        build_bin_dir = os.path.join(self.prefix, "bin")
        stage_build_dir = os.path.join(self.stage.path, "*")

        shutil.copytree(build_bin_dir, test_bin_dir, dirs_exist_ok=True)

        for item_path in glob.glob(stage_build_dir):
            if os.path.isdir(item_path):
                print(item_path)
                shutil.rmtree(item_path)
            elif os.path.isfile(item_path):
                continue
