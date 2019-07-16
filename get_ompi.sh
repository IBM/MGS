if [ -f mpich/lib/libmpi.so ]; then
  echo "libmpi.so found -- nothing to build."
else
  FILENAME=openmpi-3.1.4
  echo "Downloading openMPI source."
  wget https://download.open-mpi.org/release/open-mpi/v3.1/${FILENAME}.tar.gz
  tar xfz ${FILENAME}.tar.gz
  rm ${FILENAME}.tar.gz
  echo "configuring and building openMPI."
  cd ${FILENAME}
  ./configure \
    --enable-mpi-cxx \
    --prefix=`pwd`/../ompi \
    --with-cuda=/usr/local/cuda
  make -j4
  make install
  cd -
  rm -rf ${FILENAME}
fi
