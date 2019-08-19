if [ -z "$TRAVIS_BUILD_DIR" ]; then
  TRAVIS_BUILD_DIR=`pwd`
fi
if [ -f mpich/lib/libmpich.so ]; then
  echo "libmpich.so found -- nothing to build."
else
  FILENAME=mpich-3.2
  DIRNAME=$FILENAME
  echo "Downloading mpich source."
  wget http://www.mpich.org/static/downloads/3.2/${FILENAME}.tar.gz
  tar xfz ${FILENAME}.tar.gz
  rm ${FILENAME}.tar.gz
  echo "configuring and building mpich."
  cd ${DIRNAME}
  ./configure \
          --prefix=${TRAVIS_BUILD_DIR}/mpich \
          --enable-static=false \
          --enable-alloca=true \
          --disable-long-double \
          --enable-threads=single \
          --enable-fortran=no \
          --enable-fast=all \
          --enable-g=none \
          --enable-timing=none
  make -j4
  make install
  cd -
  rm -rf ${DIRNAME}
fi
