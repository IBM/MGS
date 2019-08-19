if [ -z "$TRAVIS_BUILD_DIR" ]; then
  TRAVIS_BUILD_DIR=`pwd`
fi
if [ -f mpich/lib/libmpi.so ]; then
  echo "libmpi.so found -- nothing to build."
else
  FILENAME=openmpi-3.1.4
  DIRNAME=$FILENAME
  echo "Downloading openMPI source."
  wget https://download.open-mpi.org/release/open-mpi/v3.1/${FILENAME}.tar.gz
  tar xfz ${FILENAME}.tar.gz
  rm ${FILENAME}.tar.gz
  echo "configuring and building openMPI."
  cd ${DIRNAME}
  ./configure \
    --enable-mpi-cxx \
    --prefix=$TRAVIS_BUILD_DIR/ompi \
    --with-cuda=/usr/local/cuda > /dev/null 
  make -j4 > /dev/null 2>&1
  make install
  cd -
  rm -rf ${DIRNAME}
fi
