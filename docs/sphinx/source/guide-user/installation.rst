************
Installation
************

Pre-requisites
==============

* gcc 11.3.0+: Tested up to 13.3.0
* spack v0.21.0+: Tested up to v0.22.3

Spack environment setup
=======================

To start with, it's necessary to clone spack:
::

    git clone -c feature.manyFiles=true -b v0.22.3 https://github.com/spack/spack.git $HOME/.spack

It can also be useful to create a temporary directory in your home directory in case there are any permission issues (eg. ``mkdir $HOME/temp_dir``).
Set the environment variables and run the spack environment setup:
::

    export SPACK_ROOT=$HOME/.spack
    source $SPACK_ROOT/share/spack/setup-env.sh
    export TMP=$HOME/temp_dir

For convenience these commands can be placed at the end of ``.bashrc``. The last step of the spack environment setup is to let spack find the ``gcc`` compilers it can use.
::

    spack compiler find

Spack can operate with multiple installations of gcc but will choose the default one that the system has aliased to ``gcc``. To see all the compilers that spack found use the command ``spack compilers``. If a specific version is desired then the ``spack compiler remove {compiler}`` command (with ``{compiler}`` replaced with the compiler that is to be removed) can be used to reduce the number of compilers down to the specified one. For example:
::

    spack compiler remove gcc@11.4.0

Standard Installation
=====================

Clone the repo:
::

    git clone git@github.com:UKAEA-Edge-Code/Reactions.git $HOME/NEC_Reactions
    cd $HOME/NEC_Reactions
    git submodule update --init

Feel free to replace ``NEC_Reactions`` with a directory name of your choice.
Next activate the spack environment (the details of the config are in ``spack.yaml``):
::

    spack env activate -p -d .

You can exit the spack environment using the (``spack env deactivate`` command).
Concretize the current specs to be installed:
::

    spack -C ./scopes/general concretize -f -U

For a standard install (CPU-only, no tests) run the commands:
::

    spack install --only-concrete --only dependencies neso-particles~nvcxx~build_tests
    spack install -j1 --only-concrete --only dependencies reactions~nvcxx build_option=RELEASE
    spack install -j1 --only-concrete reactions~nvcxx build_option=RELEASE

This will place a ``libreactions.so`` in the ``lib`` directory of the repo, this is the library to link against to use Reactions in another project.

CUDA installation (Optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a NVIDIA-GPU specific installation, if not already done then repeat the steps for cloning the repo, activating the environment and concretizing the specs. For the installation:
::

    spack install --only-concrete --only dependencies neso-particles+nvcxx~build_tests
    spack install -j1 --only-concrete --only dependencies reactions+nvcxx build_option=RELEASE
    spack install -j1 --only-concrete reactions+nvcxx build_option=RELEASE

Again this will place ``libreactions.so`` in the ``lib`` directory of the repo.

Unit tests (Optional)
=====================

Building the unit-tests is mostly the same as a standard installation with some install options changed. The compilation will produce ``libreactions.so`` as with the standard installation but will also produce a ``unit_tests`` executable in the ``bin`` directory of the repo.

Note that for running the tests, it might be necessary to load the relevant MPI package that spack has installed. It's possible to identify this by using ``spack find mpich`` or ``spack find openmpi`` and subsequently loading the relevant package (if both are present feel free to choose either) with ``spack load`` before running the unit tests.

Build unit-tests (CPU)
~~~~~~~~~~~~~~~~~~~~~~

For the CPU specific version:
::

    spack install --only-concrete --only dependencies neso-particles~nvcxx+build_tests
    spack install -j1 --only-concrete --only dependencies reactions~nvcxx build_option=TEST
    spack install -j1 --only-concrete reactions~nvcxx build_option=TEST

Build unit-tests (GPU)
~~~~~~~~~~~~~~~~~~~~~~

For the GPU specific version:
::

    spack install --only-concrete --only dependencies neso-particles+nvcxx+build_tests
    spack install -j1 --only-concrete --only dependencies reactions+nvcxx build_option=TEST
    spack install -j1 --only-concrete reactions+nvcxx build_option=TEST

Run unit-tests (CPU)
~~~~~~~~~~~~~~~~~~~~

The CPU specific command to run the unit tests is:
::

    OMP_NUM_THREADS=1 mpirun -n 1 ./bin/unit_tests

Run unit-tests (GPU)
~~~~~~~~~~~~~~~~~~~~

The GPU specific command to run the unit tests is:
::

    SYCL_DEVICE_FILTER=GPU mpirun -n 1 ./bin/unit_tests
