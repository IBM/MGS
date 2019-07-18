wget "http://faculty.cse.tamu.edu/davis/SuiteSparse/SuiteSparse-5.4.0.tar.gz"

FILENAME=SuiteSparse-5.4.0
tar -xvf ${FILENAME}.tar.gz
cd ${FILENAME}
./configure --prefix=$TRAVIS_BUILD_DIR/suitesparse
make -j10
make install
cd -
rm -rf $FILENAME
