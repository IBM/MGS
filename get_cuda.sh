FILENAME=cuda-10-1_10.1.168-1_amd64.deb
CUDA_VER=10-1

#wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1204/x86_64/cuda-repo-ubuntu1204_${CUDA_VER}_amd64.deb
wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/${FILENAME}

#sudo dpkg -i cuda-repo-ubuntu1204_${CUDA_VER}_amd64.deb
sudo dpkg -i ${FILENAME}
CUDA_APT_VER=${CUDA_VER%-*}
CUDA_APT_VER=${CUDA_APT_VER/./-}
CUDA_PACKAGES="cuda-drivers cuda-core-${CUDA_APT_VER} cuda-cudart-dev-${CUDA_APT_VER} cuda-cufft-dev-${CUDA_APT_VER}"
echo "Installing ${CUDA_PACKAGES}"
sudo apt-get install -y ${CUDA_PACKAGES}
export CUDA_HOME=/usr/local/cuda-${CUDA_VER%%-*}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
export PATH=${CUDA_HOME}/bin:${PATH}
    
