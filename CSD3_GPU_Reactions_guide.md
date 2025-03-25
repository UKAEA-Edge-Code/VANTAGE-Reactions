Navigate to the designated home directory for your user account within the rds system:
```
cd /home/$USER/rds/rds-ukaea-ap002-mOlK9qn0PlQ/$USER
```
The above command is an example of if you're in the ap002 group, replace the command with whichever would be the most relevant in your case.

Clone spack (the version is important since that's the only one that I can confirm works) and make a temporary directory, that will be used by spack instead of the default pathway that spack tries to use and doesn't have permissions for:
```
git clone -c feature.manyFiles=true -b v0.23.0 https://github.com/spack/spack.git ./.spack
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

In your rds-based home folder, install gcc-14.2.0:
```
spack install gcc@14.2.0%gcc@8.5.0
spack load gcc@14.2.0
spack compiler find
spack unload gcc@14.2.0
```
Confirm that gcc-14.2.0 is installed by executing:
```
spack compilers
```
Look for a section that resembles:
```
-- gcc rocky8-x86_64 --------------------------------------------
gcc@8.5.0  gcc@14.2.0
```
Remove the extra gcc compiler with the command:
```
spack compiler remove gcc@8.5.0
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
spack install --only-concrete --only dependencies reactions~nvcxx
spack install --only-concrete reactions~nvcxx
```
For the GPU variant:
```
spack install --only-concrete --only dependencies reactions+nvcxx
spack install --only-concrete reactions+nvcxx
```
***********************
For running the unit tests, re-install the `reactions` package with the `+enable_tests` flag added to the `spack install` command. For example for the CPU:
```
spack install --only-concrete reactions+enable_tests~nvcxx
```
and for the GPU:
```
spack install --only-concrete reactions+enable_tests+nvcxx
```

To test the CPU build, run from the repo directory:
```
OMP_NUM_THREADS=1 mpirun -n 1 build-linux-rocky8-zen3-$CPU_hash/spack-build-$CPU_hash/test/unit/unit_tests
```
For the GPU build, run from the repo directory:
```
SYCL_DEVICE_FILTER=GPU mpirun -n 1 build-linux-rocky8-zen3-$GPU_hash/spack-build-$GPU_hash/test/unit/unit_tests
```
Replace `$CPU_hash` and `$GPU_hash` with values that correspond to the hash that spack assigns to the CPU and GPU builds respectively. These can be found using `spack find -vl reactions` and checking whether `~nvcxx` or `+nvcxx` are present in the installation flags listed in the output.