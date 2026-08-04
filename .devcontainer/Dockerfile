# Build stage with Spack pre-installed and ready to be used
FROM spack/ubuntu-noble:1.0.2

RUN apt update && apt upgrade -y
RUN apt install -y git nano python3 cmake openmpi-bin libhdf5-openmpi-dev

RUN spack compiler find

RUN spack external find python
RUN spack external find cmake
RUN spack external find openmpi
RUN spack external find hdf5

ENTRYPOINT ["/bin/bash"]
