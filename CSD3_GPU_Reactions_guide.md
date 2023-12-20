Navigate to the designated home directory for your user account within the rds system:
```
cd /home/$USER/rds/rds-ukaea-ap002-mOlK9qn0PlQ/$USER
```
Clone spack (the version is important since that's the only one that I can confirm works) and make a temporary directory, that will be used by spack instead of the default pathway that spack tries to use and doesn't have permissions for:
```
git clone -c feature.manyFiles=true -b v0.21.0 https://github.com/spack/spack.git ./.spack
mkdir temp_dir
```
Move `interactive.sh` into this directory. Run the following command to create a symbolic link to an alternative version of git (needed for operation inside a GPU node session):
```
mkdir git-bin && cd git-bin && ln -s /usr/bin/git git && cd ..
```
Set up ssh keys with `ssh-keygen` and add to github ssh keys.
Next activate an interactive session on a GPU node on CSD3 and run the start-up bash script:
```
sintr -A UKAEA-AP002-GPU -p ampere -N1 -n32 --gres=gpu:1 -t 1:0:0 --qos=INTR
source interactive.sh
```
*********************
Now everytime you log on to the GPU node (from the rds-based home directory), execute `source interactive_session.sh`
*********************
Since this next step might take some time and the interactive session on the GPU node has a timelimit of 1 hour, it's recommended that a new session is initialized before proceeding.

In your rds-based home folder, install gcc-11.3.0:
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

In your rds-based home folder, clone Reactions and navigate into it:
```
git clone git@github.com:UKAEA-Edge-Code/Reactions.git $HOME/NEC_Reactions
cd ~/NEC_Reactions
git submodule update --init
```
Feel free to replace `NEC_Reactions` with a directory name of your choice.
**************************
Start-up the spack environment:
```
spack env activate -p -d .
```
You can exit the spack environment using the (`spack env deactivate` command).
Concretize the current specs to be installed:
```
spack -C ./scopes/CSD3_GPU_node concretize -f -U
```
It is advised to install CPU and GPU variants separately and install the dependencies first, then load the relevant dependencies and then install the main `reactions` package:

For the CPU variant:
```
spack install -j16 --only-concrete --only dependencies reactions~nvcxx
spack load neso-particles/neso-cpu-hash hipsycl/hipsycl-cpu-hash openmpi cmake
spack install -j16 --only-concrete reactions~nvcxx
```
The `neso-cpu-hash` and `hipsycl-cpu-hash` refers to the hashes of the variants of neso-particles and hipsycl that are configured for CPU usage.

For the GPU variant:
```
spack install -j16 --only-concrete --only dependencies reactions+nvcxx
spack load neso-particles/neso-gpu-hash hipsycl/hipsycl-gpu-hash openmpi cmake
spack install -j16 --only-concrete reactions+nvcxx
```
Similarly the `neso-gpu-hash` and `hipsycl-gpu-hash` refers to the hashes of the variants of neso-particles and hipsycl that are configured for GPU usage.

These hashes can be found by using the `spack find -v -L neso-particles hipsycl`. In the output from the command look for the `+nvcxx` and `~nvcxx` flags to find the relevant hashes.

NOTE that for now every reinstall of the `reactions` package overwrites the `bin` and `lib` directories in the main repo directory.

NOTE when switching between CPU and GPU builds it is necessary to manually delete `bin` and `lib` directories in the main repo directory and depending on which variant is currently installed run either `spack clean reactions~nvcxx` or `spack clean reactions+nvcxx`. After this the normal installation procedure in this section can be followed.
***********************
To test the CPU build, install `reactions~nvcxx` and run from the repo directory:
```
OMP_NUM_THREADS=1 mpirun -n 1 bin/unit_tests
```
For the GPU build, install `reactions+nvcxx` and run from the repo directory:
```
SYCL_DEVICE_FILTER=GPU mpirun -n 1 bin/unit_tests
```