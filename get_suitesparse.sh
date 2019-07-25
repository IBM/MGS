FILENAME=SuiteSparse-5.4.0
DIRNAME=SuiteSparse
if [ ! -f $FILENAME.tar.gz ]; then
wget "http://faculty.cse.tamu.edu/davis/SuiteSparse/$FILENAME.tar.gz"
fi

if [ -z "$TRAVIS_BUILD_DIR" ]; then
  TRAVIS_BUILD_DIR=`pwd`
fi
tar -xvf ${FILENAME}.tar.gz > /dev/null
cd ${DIRNAME}/CXSparse
make -j10
make install INSTALL=$TRAVIS_BUILD_DIR/suitesparse
cd -
rm -rf $DIRNAME
