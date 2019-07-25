FILENAME=SuiteSparse-5.4.0
DIRNAME=SuiteSparse
if [ ! -f $FILENAME.tar.gz ]; then
wget "http://faculty.cse.tamu.edu/davis/SuiteSparse/$FILENAME.tar.gz"
fi

if [ -z "$TRAVIS_BUILD_DIR" ]; then
  TRAVIS_BUILD_DIR=`pwd`
fi
tar -xvf ${FILENAME}.tar.gz > /dev/null
cd ${DIRNAME}/SuiteSparse_config 
make install INSTALL=$TRAVIS_BUILD_DIR/suitesparse
make config
make -j10
cd -
cd ${DIRNAME}/CXSparse
make -j10 library
cd -
rm -rf $DIRNAME
