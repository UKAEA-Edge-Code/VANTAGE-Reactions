************
Installation
************

Pre-requisites
==============

* gcc 11.3.0+: Tested up to 14.2.0
* spack v1.0.2
* python 3.8+ : Tested up to python 3.12

Spack environment setup
=======================

To start with, it's necessary to clone spack:
::

    git clone -c feature.manyFiles=true --depth=2 -b v1.0.2 https://github.com/spack/spack.git $HOME/.spack

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
A pre-existing ``gcc`` installation higher than version 11.3.0 should work without issue.
If necessary and permissions allow, installing ``gcc`` through a package manager (such as ``apt`` or ``dnf``) and running ``spack compiler find`` should work but if not then the compiler will need to be installed via spack.
For installing via spack a version of ``gcc`` needs to be present that is older than the version that you wish to install. To install the new compiler, first run:
::

    spack compiler find

This should let ``spack`` find the pre-existing compiler. If for example, ``gcc-11.3.0`` needs to be installed with a pre-existing ``gcc-9.4.0``, the command to install the new compiler would be:
::

    spack install gcc@11.3.0%gcc@9.4.0

Now it's necessary to remove all the compilers listed in ``spack compilers``, the reasoning for this is that the newly installed compiler will be tied to the ``VANTAGE-Reactions`` installation rather than using it system-wide.
This can be done using ``spack compiler remove {compiler}``, where ``{compiler}`` is the entry from ``spack compilers`` that is to be removed.
Then, run the following command:
::

    spack find --paths gcc

Note the paths listed for the ``gcc`` command (for convenience the path can be set to an environment variable) and run:
::

    spack compiler find ${gcc_compiler_install_path}

Defining external packages (optional)
=====================================

If compatible versions of ``cmake`` (3.24+), ``python`` (3) and ``llvm`` (18:20) are pre-installed, then they can be designated as external packages that spack will try and use when installing Reactions.
This reduces the number of dependencies that spack has to install and hence speeds up the first-time install significantly.
To designate a package as an external one, the path of the root directory of the package must be known, then the following command sets the package as external:
::

    spack external find --path {path_to_package} {name_of_package}

Note, this must be done outside the Reactions spack environment (for example in the $HOME directory).
This will modify a file in ``$SPACK_ROOT`` called ``packages.yaml`` or create one if it doesn't exist. It is recommended to assign the listed packages as external if possible to smooth the experience of the first time install.
NOTE: For ``llvm``, if it's pre-installed then it's best to let spack find it using ``spack compiler find ${llvm_install_path}`` instead.

Installation
=====================

Clone the repo:
::

    git clone --recurse-submodules git@github.com:UKAEA-Edge-Code/VANTAGE-Reactions.git $HOME/VANTAGE_Reactions
    cd $HOME/VANTAGE_Reactions

Feel free to replace ``$HOME/VANTAGE_Reactions`` with a directory name of your choice.
Next activate the default spack environment (the details of the config are in ``$HOME/VANTAGE_Reactions/environments/spack_default/spack.yaml``):
::

    spack env activate -p -d environments/spack_default

You can exit the spack environment using the (``spack env deactivate`` command).

**NOTE: All commands following this must be executed inside this environment.**

Default Install
~~~~~~~~~~~~~~~
For a standard install (CPU-only, using GCC) run the commands:
::

    spack install

Note if there is a compiler error or out-of-RAM crash when running the command then add ``-j1`` after ``spack install`` and try again.

NOTE - It is recommended to not exceed ``-j2`` if there's less than 16GB of system RAM.

Optional variants
~~~~~~~~~~~~~~~~~
In addition to the default environment, there are also some other variants included in the ``environments``.

Each sub-folder has it's own ``spack.yaml`` that defines an environment and spec that's specific to that sub-folder. For example, the ``spack_omp_accelerated``
contains a spec that allows for an installation with ``adaptivecpp`` designating ``llvm`` as the backend for compilation.

To use any of these environments, activate the environment with ``spack env activate -p -d environments/{desired_environment}`` (make sure that you're outside of the default environment using ``spack env deactivate`` before activating another one). Ensure any existing loaded packages are unloaded with ``spack unload --all`` and then simply run ``spack install``.

For CUDA-specific installations, the environments are provided but given the subtleties associated with this installation, there is no guarantee that these will work out-of-the-box and might require more modifications (possibly outside of the ``spack.yaml``) to function.

Similarly for HIP-specific installations (for AMD GPUs), the environment is provided but again depending on the subtleties of your system, there may need to be more modifications.

The ``spack.yaml`` files do have some limited guidance on things like specifying GPU architecture and some dependencies but there may need to be more modifications system-side to make things work smoothly.

If any compatibility issues are present when attempting these optional variants, please contact the repo maintainers for support.

Run unit-tests (CPU)
~~~~~~~~~~~~~~~~~~~~

Unload any existing loaded packages, then load the ``vantagereactions`` package:
::

    spack unload --all
    spack load vantagereactions

The CPU specific command to run the unit tests is:
::

    OMP_NUM_THREADS=1 mpirun -n 1 unit_tests

Certain tests have additional checks to ensure that invalid inputs/states throw errors.
Ordinarily, if a check using NESOASSERT fails, the executable aborts via MPI which makes testing failure cases difficult.
To ensure that invalid/failure states are tested, run the unit tests with the command:
::

    TEST_NESOASSERT=ON OMP_NUM_THREADS=1 mpirun -n 1 unit_tests

Run unit-tests (GPU)
~~~~~~~~~~~~~~~~~~~~
Unload any existing loaded packages, then load the ``vantagereactions`` package again (ensuring you're within the correct environment).
Then the GPU specific command to run the unit tests is:
::

    SYCL_DEVICE_FILTER=GPU mpirun -n 1 unit_tests

Similarly, to enable testing of failure states on GPU:
::

    TEST_NESOASSERT=ON SYCL_DEVICE_FILTER=GPU mpirun -n 1 unit_tests

Compile and run an individual unit test
=======================================

By default the unit tests are built as a single ``unit_tests`` executable that
contains every ``test_*.cpp`` under ``test/unit/``. That is what the commands
above run, and what CI and ``run_tests.sh`` use. For faster iteration on a
single test (or a small set), enable *split mode*, which builds one executable
per ``test_*.cpp`` linked against a shared object library of the non-test
translation units (the GoogleTest/MPI ``Main.cpp`` plus the two test-only
template-instantiation sources).

There are two ways to use split mode: through the Spack package (the
supported path, recommended if you installed via Spack) or by configuring the
source tree with CMake directly (useful for fast iteration in a clone without
re-running a Spack install).

Via the Spack package
~~~~~~~~~~~~~~~~~~~~~

The ``vantagereactions`` Spack package exposes two variants that map onto the
CMake options:

* ``tests_split`` (bool, default off) — enables split mode.
* ``test_filter`` (multi-valued string, default ``all``) — selects which
  ``test_*.cpp`` stems to build. ``all`` (the default) builds every test;
  otherwise pass one or more stems, comma-separated.

Both are only meaningful when tests are enabled (``+enable_tests``);
requesting ``+tests_split`` without ``+enable_tests`` results in a concretization
error.

.. note::

    ``vantagereactions`` and ``neso-particles`` live in custom Spack repos
    (``VANTAGE-repo`` and ``NESO-Spack``) that are only registered inside the
    project's environments under ``environments/``, so split-mode specs must be
    installed **inside one of those environments** — a bare ``spack install
    vantagereactions...`` outside an env fails with ``'vantagereactions' does not
    exist``. Inside an activated environment, Spack further refuses specs that
    are not already listed in the env's ``spack.yaml`` (``Cannot install '...'
    because no matching specs are in the current environment.``). The commands
    below use ``spack -e <env> install --add``, which adds the spec to the
    environment and installs it in one step. Remove it afterwards with
    ``spack -e <env> rm '<spec>'`` (and ``concretize -f``) so it stops being
    concretized alongside the monolith.

.. warning::

    The split spec **must repeat the** ``^`` **dependency constraints** that the
    environment's monolith root spec pins. The project environments only pin the SYCL
    *provider* (``adaptivecpp``) in their ``packages:`` block, not the
    ``compilationflow`` variant or ``neso-particles~build_tests``; those live on
    the root spec itself. A bare split spec (``vantagereactions+enable_tests
    +tests_split`` with no ``^`` constraints) concretizes a separate DAG (the
    project envs use ``concretizer: unify: false``) but against the *default*
    ``adaptivecpp compilationflow`` and ``neso-particles+build_tests`` — i.e.
    the wrong backend and an unnecessary test build of neso-particles.

    The fix is to carry the same ``^`` suffix as the monolith root spec. For
    ``spack_default`` that is ``build_type=Release ^neso-particles~build_tests
    ^neso.adaptivecpp compilationflow=omplibraryonly ^googletest``. The commands
    below include it. (For ``spack_omp_accelerated``, copy the ``^`` suffix from
    that env's own root spec — it pins the omp-accelerated ``compilationflow``.)

Installation with split mode on, building every ``test_*.cpp`` as its own
executable (substitute the environment you are using, e.g.
``environments/spack_default``, and its matching ``^`` suffix)::

    spack -e environments/spack_default install --add \
        'vantagereactions+enable_tests+tests_split build_type=Release ^neso-particles~build_tests ^neso.adaptivecpp compilationflow=omplibraryonly ^googletest'

To build only one test (the fastest path when iterating on a single file),
pass its stem via ``test_filter``:
::

    spack -e environments/spack_default install --add \
        'vantagereactions+enable_tests+tests_split test_filter=test_reaction_controller build_type=Release ^neso-particles~build_tests ^neso.adaptivecpp compilationflow=omplibraryonly ^googletest'

A comma-separated list selects several:
::

    spack -e environments/spack_default install --add \
        'vantagereactions+enable_tests+tests_split test_filter=test_properties,test_species build_type=Release ^neso-particles~build_tests ^neso.adaptivecpp compilationflow=omplibraryonly ^googletest'

An unknown stem fails at concretization (typo-proofing). The built executables
land under the per-package build stage — they are a development convenience and
are **not** installed into the Spack prefix, so ``spack load vantagereactions``
does not put them on ``PATH``. Find them with:
::

    spack -e environments/spack_default location -b \
        'vantagereactions+enable_tests+tests_split test_filter=test_reaction_controller build_type=Release ^neso-particles~build_tests ^neso.adaptivecpp compilationflow=omplibraryonly ^googletest'

and run with MPI as usual:
::

    OMP_NUM_THREADS=1 mpirun -n 1 <build_dir>/test/unit/test_reaction_controller

Via a direct CMake configure of the source tree
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you are working in a clone of the repository and already have the
dependencies installed (e.g. via a prior ``spack install``), you can configure
the source tree with CMake directly and pass the same two options as CMake
cache variables:
::

    cmake -S . -B build -DCMAKE_PREFIX_PATH="<installed deps>" \
        -DREACTIONS_ENABLE_TESTS=ON -DREACTIONS_TESTS_SPLIT=ON \
        -DCMAKE_BUILD_TYPE=Release
    cmake --build build

With no further options this produces one executable per ``test_*.cpp``, named
after the source stem (e.g. ``test_reaction_controller``, ``test_species``).
Run the one you want with MPI as before:
::

    OMP_NUM_THREADS=1 mpirun -n 1 build/test/unit/test_reaction_controller

To build only a single test, pass its source stem via
``REACTIONS_TEST_FILTER``:
::

    cmake -S . -B build -DCMAKE_PREFIX_PATH="<installed deps>" \
        -DREACTIONS_ENABLE_TESTS=ON -DREACTIONS_TESTS_SPLIT=ON \
        -DREACTIONS_TEST_FILTER=test_reaction_controller \
        -DCMAKE_BUILD_TYPE=Release
    cmake --build build
    OMP_NUM_THREADS=1 mpirun -n 1 build/test/unit/test_reaction_controller

``REACTIONS_TEST_FILTER`` accepts a single stem or a semicolon-separated list,
e.g. ``-DREACTIONS_TEST_FILTER="test_properties;test_species"``. An empty
value (the default) builds every ``test_*.cpp``. A stem that does not match any
``test_*.cpp`` is a hard configure error, so typos are caught early.
