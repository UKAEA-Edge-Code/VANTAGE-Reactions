# Copyright 2013-2021 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

import json
from os import path
from glob import glob

from llnl.util import filesystem

from spack import *

"""
Install nvc++ version with something like

    spack install neso.adaptivecpp +nvcxx

# openmp only
    spack install neso.adaptivecpp
to avoid llvm build
"""


class Adaptivecpp(CMakePackage):
    """AdaptiveCPP is an implementation of the SYCL standard programming model
    over NVIDIA CUDA/AMD HIP"""

    homepage = "https://github.com/AdaptiveCpp/AdaptiveCpp"
    git = "https://github.com/AdaptiveCpp/AdaptiveCpp.git"

    provides("sycl")

    version(
        "24.10.0",
#        commit="7677cf6eefd8ab46d66168cd07ab042109448124",
        sha256="3bcd94eee41adea3ccc58390498ec9fd30e1548af5330a319be8ce3e034a6a0b",
        url="https://github.com/AdaptiveCpp/AdaptiveCpp/archive/refs/tags/v24.10.0.tar.gz"
#        submodules=True,
    )
    version(
        "24.06.0",
        commit="fc51dae9006d6858fc9c33148cc5f935bb56b075",
        submodules=True,
    )
    version(
        "24.02.0",
        commit="974adc33ea5a35dd8b5be68c7a744b37482b8b64",
        submodules=True,
    )
    version(
        "23.10.0",
        commit="3952b468c9da89edad9dff953cdcab0a3c3bf78c",
        submodules=True,
    )

    variant(
        "cuda",
        default=False,
        description="Enable CUDA backend for SYCL kernels using llvm+cuda",
    )
    variant(
        "nvcxx",
        default=False,
        description="Enable CUDA backend for SYCL kernels using nvcxx",
    )
    variant(
        "omp_llvm",
        default=False,
        description="Enable accelerated OMP backend for SYCL kernels using LLVM",
    )
    variant(
        "opencl",
        default=False,
        description="Enable OpenCL backend.",
    )

    depends_on("cmake@3.5:", type="build")
    depends_on("boost +filesystem", when="@23.10.0:")
    depends_on(
        "boost@1.60.0: +filesystem +fiber +context cxxstd=17", when="@23.10.0:"
    )
    depends_on("python@3:")
    # depends_on("llvm@8: +clang", when="~cuda")
    depends_on("llvm@9: +clang", when="+cuda", type=("build", "link", "run"))
    depends_on("llvm@9: +clang", when="+omp_llvm")
    depends_on("cuda", when="@23.10.0: +cuda")
    depends_on("cuda", when="+nvcxx")

    # Version 24.10.0 llvm backends do not work with LLVM 19 so we restrict
    # llvm to versions 15 to 18 as those are the versions the AdaptiveCpp CI
    # runs on.
    depends_on(
        "llvm@15:18 +clang",
        when="@24.10.0 +cuda",
        type=("build", "link", "run"),
    )
    depends_on(
        "llvm@15:18 +clang",
        when="@24.10.0 +omp_llvm",
        type=("build", "link", "run"),
    )

    # If we directly add nvhpc as build then the Adaptivecpp cmake finds the
    # openmp inside nvhpc. If we add nvhpc as link or run then nvhpc gets
    # loaded as a runtime dependency which then breaks downstream cmake
    # configuration. Downstream cmake finds nvc++ as the compiler which then
    # breaks the downstream projects.
    depends_on("nvhpc_transitive@22.9:", when="+nvcxx", type="run")
    depends_on("opencl@3.0", when="+opencl")

    patch("allow-disable-find-cuda-23.10.0.patch", when="@23.10.0")
    patch("macos-non-apple-clang-24.02.0.patch", when="@24.02.0")
    patch("macos-non-apple-clang-24.06.0.patch", when="@24.06.0")

    conflicts(
        "%gcc@:8",
        when="@23.10.0:",
        msg="AdaptiveCPP needs proper C++17 support to be built, %gcc is too old",
    )
    conflicts(
        "^llvm build_type=Debug",
        when="+cuda",
        msg="LLVM debug builds don't work with AdaptiveCPP CUDA backend; for "
        "further info please refer to: "
        "https://github.com/illuhad/hipSYCL/blob/master/doc/install-cuda.md",
    )

    # If we build against llvm then nvc++ ends up trying to use the linker
    # packaged with llvm and this ends up as a mess. Users should either do
    # +cuda +omp_llvm or +nvcxx and not both within the same installation.
    conflicts(
        "+nvcxx",
        when="+cuda",
        msg="Cannot use nvc++ and llvm backends simultaneously."
        "Choose one of +nvcxx or +cuda +omp_llvm.",
    )
    conflicts(
        "+nvcxx",
        when="+omp_llvm",
        msg="Cannot use nvc++ and llvm backends simultaneously."
        " Choose one of +nvcxx or +cuda +omp_llvm.",
    )

    def cmake_args(self):

        spec = self.spec
        args = [
            "-DACPP_VERSION_SUFFIX=spack",
            "-DWITH_CPU_BACKEND:Bool=TRUE",
            # TODO: no ROCm stuff available in spack yet
            "-DWITH_ROCM_BACKEND:Bool=FALSE",
            "-DWITH_STDPAR_COMPILER:Bool=FALSE",
        ]

        if "llvm" in spec:
            # prevent AdaptiveCPP's cmake to look for other LLVM installations
            # if the specified one isn't compatible
            args += [
                "-DDISABLE_LLVM_VERSION_CHECK:Bool=TRUE",
            ]

            # LLVM directory containing all installed CMake files
            # (e.g.: configs consumed by client projects)
            llvm_cmake_dirs = filesystem.find(
                spec["llvm"].prefix, "LLVMExports.cmake"
            )
            if len(llvm_cmake_dirs) != 1:
                raise InstallError(
                    "concretized llvm dependency must provide "
                    "a unique directory containing CMake client "
                    "files, found: {0}".format(llvm_cmake_dirs)
                )
            args.append(
                "-DLLVM_DIR:String={0}".format(path.dirname(llvm_cmake_dirs[0]))
            )
            # clang internal headers directory
            llvm_clang_include_dirs = filesystem.find(
                spec["llvm"].prefix, "__clang_cuda_runtime_wrapper.h"
            )
            if len(llvm_clang_include_dirs) != 1:
                raise InstallError(
                    "concretized llvm dependency must provide a "
                    "unique directory containing clang internal "
                    "headers, found: {0}".format(llvm_clang_include_dirs)
                )
            args.append(
                "-DCLANG_INCLUDE_PATH:String={0}".format(
                    path.dirname(llvm_clang_include_dirs[0])
                )
            )
            # target clang++ executable
            llvm_clang_bin = path.join(spec["llvm"].prefix.bin, "clang++")
            if not filesystem.is_exe(llvm_clang_bin):
                raise InstallError(
                    "concretized llvm dependency must provide a "
                    "valid clang++ executable, found invalid: "
                    "{0}".format(llvm_clang_bin)
                )
            args.append(
                "-DCLANG_EXECUTABLE_PATH:String={0}".format(llvm_clang_bin)
            )

        if ("+cuda" in spec) or ("+nvcxx" in spec):
            args += [
                "-DCUDA_TOOLKIT_ROOT_DIR:String={0}".format(
                    spec["cuda"].prefix
                ),
                "-DWITH_CUDA_BACKEND:Bool=TRUE",
            ]
        else:
            args += [
                "-DWITH_CUDA_BACKEND:Bool=FALSE",
                "-DDISABLE_FIND_PACKAGE_CUDA=TRUE",
            ]

        if "+nvcxx" in spec:
            nvcpp_cands = glob(
                path.join(spec["nvhpc"].prefix, "**/nvc++"), recursive=True
            )
            if len(nvcpp_cands) < 1:
                raise InstallError("Failed to find nvc++ executable")
            args.append("-DNVCXX_COMPILER={0}".format(nvcpp_cands[0]))

            if not ("llvm" in spec):
                args.append("-DWITH_CUDA_NVCXX_ONLY=ON")

        if not ("llvm" in spec):
            args += [
                "-DWITH_ACCELERATED_CPU=OFF",
                "-DBUILD_CLANG_PLUGIN=OFF",
            ]

        if "+opencl" in spec:
            args += [
                "-DWITH_SSCP_COMPILER:Bool=TRUE",
                "-DWITH_OPENCL_BACKEND=ON",
            ]
        else:
            args += [
                "-DWITH_SSCP_COMPILER:Bool=FALSE",
                "-DWITH_OPENCL_BACKEND=OFF",
            ]

        return args

    @run_after("install")
    def filter_config_file(self):

        # The config file name and location depends on version:
        # pre-24.02.0: syclcc.json
        # post-24.02.0: acpp-core.json
        config_file_paths = filesystem.find(
            self.prefix, ("syclcc.json", "acpp-core.json")
        )
        if len(config_file_paths) != 1:
            raise InstallError(
                "installed AdaptiveCPP must provide a unique compiler driver "
                "configuration file, found: {0}".format(config_file_paths)
            )
        config_file_path = config_file_paths[0]
        with open(config_file_path) as f:
            config = json.load(f)

        # There may be a separate cuda config file
        cuda_config = None
        cuda_config_file_path = None
        config_file_paths = filesystem.find(self.prefix, ("acpp-cuda.json",))
        if len(config_file_paths) > 0:
            cuda_config_file_path = config_file_paths[0]
            with open(cuda_config_file_path) as f:
                cuda_config = json.load(f)

        # 1. Fix compiler: use the real one in place of the Spack wrapper
        config["default-cpu-cxx"] = self.compiler.cxx
        # 2. Fix stdlib: we need to make sure cuda-enabled binaries find
        #    the libc++.so and libc++abi.so dyn linked to the sycl
        #    ptx backend

        # Find the rpaths for cpp
        rpaths = set()
        if "llvm" in self.spec:
            so_paths = filesystem.find(self.spec["llvm"].prefix, "libc++.so")
            if len(so_paths) != 1:
                raise InstallError(
                    "concretized llvm dependency must provide a "
                    "unique directory containing libc++.so, "
                    "found: {0}".format(so_paths)
                )
            rpaths.add(path.dirname(so_paths[0]))
            so_paths = filesystem.find(self.spec["llvm"].prefix, "libc++abi.so")
            if len(so_paths) != 1:
                raise InstallError(
                    "concretized llvm dependency must provide a "
                    "unique directory containing libc++abi.so, "
                    "found: {0}".format(so_paths)
                )
            rpaths.add(path.dirname(so_paths[0]))

            # the omp llvm backend may link against the libomp.so in llvm
            so_paths = filesystem.find(self.spec["llvm"].prefix, "libomp.so")
            rpaths.add(path.dirname(so_paths[0]))

            # Add the rpaths for llvm c++
            default_cuda_link_line = "default-cuda-link-line"
            if cuda_config is not None:
                if default_cuda_link_line in cuda_config.keys():
                    cuda_config[default_cuda_link_line] += " " + " ".join(
                        "-rpath {0}".format(p) for p in rpaths
                    )
            else:
                if default_cuda_link_line in config.keys():
                    config[default_cuda_link_line] += " " + " ".join(
                        "-rpath {0}".format(p) for p in rpaths
                    )

            # add the rpaths for llvm omp
            default_omp_link_line = "default-omp-link-line"
            if default_omp_link_line in config.keys():
                config[default_omp_link_line] += " " + " ".join(
                    "-Wl,-rpath {0}".format(p) for p in rpaths
                )

        else:
            # If llvm is not in the spec then explicitly use "omp.library-only"
            # as the default backend.
            default_targets = "default-targets"
            if default_targets in config.keys():
                config[default_targets] = "omp.library-only"

        # Replace the installed config file
        with open(config_file_path, "w") as f:
            json.dump(config, f, indent=2)

        # replace the cuda config if it exists
        if cuda_config is not None:
            with open(cuda_config_file_path, "w") as f:
                json.dump(cuda_config, f, indent=2)
