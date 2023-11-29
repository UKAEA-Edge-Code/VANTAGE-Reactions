# Edge Code Reactions Library

Repository for Particle Tracker Reactions library, based on NESO-Particles.

## NESO-Particles installation instructions for CSD3 GPU node:

Navigate to the designated home directory for your user account within the rds system:
```
cd /home/$USER/rds/rds-ukaea-ap002-mOlK9qn0PlQ/$USER
```
Next activate an interactive session on a GPU node on CSD3:
```
sintr -A UKAEA-AP002-GPU -p ampere -N1 -n32 --gres=gpu:1 -t 1:0:0 --qos=INTR
```
Set the `HOME` environment variable and clone spack (the version is important since that's the only one that I can confirm works):
```
export HOME=$PWD
git clone -c feature.manyFiles=true -b v0.21.0 https://github.com/spack/spack.git $HOME/.spack
```
Set the spack environment variables and run the `setup-env.sh` script:
```
export SPACK_ROOT=~/.spack
source $SPACK_ROOT/share/spack/setup-env.sh
```
Make a temporary directory, that will be used by spack instead of the default pathway that spack tries to use and doesn't have permissions for.
```
mkdir temp_dir
export TMP=~/temp_dir
```
*********************
Now everytime you log on to the GPU node (from the rds-based home directory), the following commands need to be executed:
```
export HOME=$PWD
export SPACK_ROOT=~/.spack
source $SPACK_ROOT/share/spack/setup-env.sh
export TMP=~/temp_dir
```
*********************
Since this next step might take some time and the interactive session on the GPU node has a timelimit of 1 hour, it's recommended that a new session is initialized before proceeding.

In your rds-based home folder, install gcc-11.3.0. 
:
```
spack install gcc@11.3.0%gcc@8.5.0
spack load gcc@11.3.0
spack compiler find
spack unload gcc@11.3.0
```
Confirm that gcc-11.3.0 is installed by executing:
```
spack compilers
```
Look for a section that resembles:
```
-- gcc rocky8-x86_64 --------------------------------------------
gcc@8.5.0  gcc@11.3.0
```
****************************
Again probably best to start a new session on the GPU node.

In your rds-based home folder, clone NEC_Reactions and navigate into it:
```
git clone git@github.com:UKAEA-Edge-Code/Reactions.git $HOME/NEC_Reactions
cd ~/NEC_Reactions
git submodule update --init
```
****************************
Start-up the spack environment:
```
spack env activate -p -d .
```
You can exit the spack environment using the (`spack env deactivate` command).

Concretize first (choose the appropriate scope) then install. For example, for a non-CSD3 install using a non-nvcxx compiler:
```
spack -C ./scopes/general concretize -f -U
spack install --only-concrete ~nvcxx
```
For a CSD3 install using a nvcxx compiler:
```
spack -C ./scopes/CSD3_GPU_node concretize -f -U
spack install --only-concrete +nvcxx
```
This should create a build directory for NESO-Particles in `neso-particles/`. The format should be in the form of `spack-build-*` where there will be a hash in place of `*` that can be used to keep track of different builds.
***********************
To test both builds, navigate to the `test` directory that should be present within both builds. 

For the CPU build, run:
```
OMP_NUM_THREADS=1 ./testNESOParticles
```
For the GPU build, run:
```
SYCL_DEVICE_FILTER=GPU ./testNESOParticles
```

## Installation of Reactions code:
```
mkdir build && cd build
cmake -G "Unix Makefiles" ..
make -j
make test
```
Add the `-DBUILD_TYPE=DEBUG` flag to the cmake command for debug flags (`-g -O0`) and `-DBUILD_TYPE=RELEASE` for optimization flags (`-O2`).

For coverage info run the CMake command with `-DBUILD_TYPE=TEST` flag. To generate documentation run `make doc` instead of `make -j`. The command `make test` will run unit tests.