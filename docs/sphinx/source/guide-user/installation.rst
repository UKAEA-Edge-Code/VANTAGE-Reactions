************
Installation
************

Pre-requisites
==============

* gcc 11.3.0+: Tested up to 14.2.0
* clang 18.1.+: Tested up to 18.1.8
* spack v0.23.0+: Tested up to v0.23.1

Spack environment setup
=======================

To start with, it's necessary to clone spack:
::

    git clone -c feature.manyFiles=true -b v0.23.1 https://github.com/spack/spack.git $HOME/.spack

It can also be useful to create a temporary directory in your home directory in case there are any permission issues (eg. ``mkdir $HOME/temp_dir``).
Set these environment variables and run the spack environment setup:
::

    export SPACK_ROOT=$HOME/.spack
    source $SPACK_ROOT/share/spack/setup-env.sh
    export TMP=$HOME/temp_dir

For convenience these commands can be placed at the end of ``.bashrc``. The last step of the spack environment setup is to let spack find the ``gcc`` compilers it can use.
::

    spack compiler find

Spack can operate with multiple installations of ``gcc`` but will choose the default one that the system has aliased to ``gcc``. To see all the compilers that spack found use the command ``spack compilers``.

Compiler setup
==============
GCC
~~~
For ``gcc``, a pre-existing installation higher than version 11.3.0 should work without issue. If there is no compatible version available then it's necessary to install one through spack if a build of Reactions with gcc is necessary.
If permissions allow, installing ``gcc`` through a package manager (such as ``apt`` or ``dnf``) and running ``spack compiler find`` should work but if not then the compiler will need to be installed via spack.
For this a version of ``gcc`` needs to be present that is older than the version that you wish to install. To install the new compiler, first run:
::

    spack compiler find

This should let ``spack`` find the pre-existing compiler. If for example, ``gcc-11.3.0`` needs to be installed with a pre-existing ``gcc-9.4.0``, the command to install the new compiler would be:
::

    spack install gcc@11.3.0%gcc@9.4.0

Now it's necessary to remove all the compilers listed in ``spack compilers``, the reasoning for this is that the newly installed compiler will be tied to the ``Reactions`` installation rather than using it system-wide. This can be done using ``spack compiler remove {compiler}``, where ``{compiler}`` is the entry from ``spack compilers`` that is to be removed.
Run the following command:
::

    spack find --paths gcc

Note the paths listed for the ``gcc`` command (for convenience the path could be set to an environment variable) and run:
::

    spack compiler find ${gcc_compiler_install_path}

From here the ``Standard Installation`` should be followed.

Clang
~~~~~
For ``clang``, a pre-existing installation (ie. one installed with the OS) may not have the full ``llvm`` installation. Additionally, the version may not be one that's been validated for ``Reactions``. 
If permissions allow, installing a compatible version of ``llvm`` through a package manager (such as ``apt`` or ``dnf``) and running ``spack compiler find`` should work but if not then the compiler will need to be installed via spack.
For compatibility, it's recommended to install ``llvm@18.1.8`` using a pre-existing ``gcc`` compiler (ensuring that it's listed in ``spack compilers``), for example:
::

    spack install llvm@18.1.8%gcc@9.4.0


Again remove existing compilers from ``spack compilers`` using ``spack compiler remove {compiler}`` where ``{compiler}`` is the entry from ``spack compilers`` that is to be removed.
Run the following command:
::

    spack find --paths llvm

Note the paths listed for the ``llvm`` command (for convenience the path could be set to an environment variable) and run:
::

    spack compiler find ${clang_compiler_install_path}

From here the ``Standard Installation`` should be followed.

Defining external packages
==========================

If compatible versions of ``cmake`` (3.24+), ``python`` (3) and ``llvm`` (18.1.+) are pre-installed, then they can be designated as external packages that spack will try and use when installing Reactions. This reduces the number of dependencies that spack has to install and hence speeds up the first-time install significantly. To designate a package as an external one, the path of the root directory of the package must be known, then the following command sets the package as external:
::

    spack external find --path {path_to_package} {name_of_package}

Note, this must be done outside the Reactions spack environment (for example in the $HOME directory). For example, to set ``llvm`` as an external package where it's been installed via a package manager (``apt`` in this case):
::

    spack external find --path /usr/lib/llvm-18 llvm

(note, for package manager installed llvm-18, there is also a symbolic link created in ``/usr/bin`` that shows the clang compilers but these are renamed as ``clang-18`` and ``clang++-18`` and some packages that spack needs to install require the names ``clang`` and ``clang++`` for the compilers so it's necessary to point to ``/usr/lib/llvm-18``. This may differ for package managers other than ``apt``).
This will modify a file in ``$SPACK_ROOT`` called ``packages.yaml`` or create one if it doesn't exist. It is recommended to assign the listed packages as external if possible to smooth the experience of the first time install.

Non-cluster specific externals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The above packages are necessary for general installations on non-clusters and clusters, but on non-clusters there is an additional package that can be designated as external. Namely the ``mpich`` package, if pre-installed, is a useful external package. The same general command listed in this section should work for ``mpich``.

Standard Installation
=====================

Clone the repo:
::

    git clone --recurse-submodules git@github.com:UKAEA-Edge-Code/Reactions.git $HOME/NEC_Reactions
    cd $HOME/NEC_Reactions

Feel free to replace ``$HOME/NEC_Reactions`` with a directory name of your choice.
Next activate the spack environment (the details of the config are in ``spack.yaml``):
::

    spack env activate -p -d .

You can exit the spack environment using the (``spack env deactivate`` command).

**NOTE: All commands following this must be executed inside this environment.**

Within the matrix of spec definitions in ``spack.yaml``, there is a entry relating to compilers containing ``"%gcc@11.3.0:"`` and ``%clang@18.1:`` which denotes the compilers that ``spack`` will try and concretize with. If either is not present, simply comment out that entry with ``#`` before concretizing.

Concretize the current specs to be installed:
::

    spack -C ./scopes/{system-scope} concretize -f -U

, where ``{system-scope}`` is either ``general`` or ``CSD3_GPU_node`` depending on which system type ``Reactions`` is being installed on.

Install
~~~~~~~
For a standard install (CPU-only, no tests) run the commands:
::

    spack install --only-concrete reactions%{compiler}~enable_tests ^adaptivecpp compilationflow={omplibraryonly, ompaccelerated}

Where the ``{compiler}`` is the compiler you wish to install with. Additionally choose one of the two options listed for the ``compilationflow``. Note if there is a compiler error when running the command then add ``-j1`` after ``spack install`` and try again.

CUDA installation (Optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a NVIDIA-GPU specific installation, if not already done then repeat the steps for cloning the repo, activating the environment and concretizing the specs. For the installation:
::

    spack install --only-concrete reactions~enable_tests%{compiler} ^adaptivecpp compilationflow={cudanvcxx, cudallvm}

Unit tests (Optional)
=====================

Building the unit-tests is mostly the same as a standard installation with some install options changed. The compilation will produce a ``unit_tests`` executable in the ``test/unit`` directory inside the build directory created by spack during installation.

Note that for running the tests, it might be necessary to load the relevant MPI package that spack has installed. It's possible to identify this by using ``spack find mpich`` and subsequently loading the relevant package with ``spack load`` before running the unit tests. This is only necessary if not running on CSD3 (or any other system with an externally defined openmpi or mpich).

Build unit-tests (CPU)
~~~~~~~~~~~~~~~~~~~~~~

For the CPU specific version:
::

    spack install --only-concrete reactions%{compiler}+enable_tests ^neso-particles~build_tests ^adaptivecpp compilationflow={omplibraryonly, ompaccelerated}

This will build the unit tests for ``Reactions`` but not for ``neso-particles``. To build the ``neso-particles`` tests as well, run:
::

    spack install --only-concrete reactions%{compiler}+enable_tests ^neso-particles+build_tests ^adaptivecpp compilationflow={omplibraryonly, ompaccelerated}

Build unit-tests (GPU)
~~~~~~~~~~~~~~~~~~~~~~

For the GPU specific version:
::

    spack install --only-concrete reactions%{compiler}+enable_tests ^neso-particles~build_tests ^adaptivecpp compilationflow={cudanvcxx, cudallvm}

This will build the unit tests for ``Reactions`` but not for ``neso-particles``. To build the ``neso-particles`` tests as well, run:
::

    spack install --only-concrete reactions%{compiler}+enable_tests ^neso-particles+build_tests ^adaptivecpp compilationflow={cudanvcxx, cudallvm}

Run unit-tests (CPU)
~~~~~~~~~~~~~~~~~~~~

The CPU specific command to run the unit tests is:
::

    OMP_NUM_THREADS=1 mpirun -n 1 {build_dir}/test/unit/unit_tests

Certain tests have additional checks to ensure that invalid inputs/states throw errors.
Ordinarily, if a check using NESOASSERT fails, the executable aborts via MPI which makes testing failure cases difficult.
To ensure that invalid/failure states are tested, run the unit tests with the command:
::

    TEST_NESOASSERT=ON OMP_NUM_THREADS=1 mpirun -n 1 {build_dir}/test/unit/unit_tests

Run unit-tests (GPU)
~~~~~~~~~~~~~~~~~~~~

The GPU specific command to run the unit tests is:
::

    SYCL_DEVICE_FILTER=GPU mpirun -n 1 {build_dir}/test/unit/unit_tests

Similarly, to enable testing of failure states on GPU:
::

    TEST_NESOASSERT=ON SYCL_DEVICE_FILTER=GPU mpirun -n 1 {build_dir}/test/unit/unit_tests
