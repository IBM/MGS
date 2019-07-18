FILENAME=SuiteSparse-5.4.0
DIRNAME=SuiteSparse
wget "http://faculty.cse.tamu.edu/davis/SuiteSparse/$FILENAME.tar.gz"

tar -xvf ${FILENAME}.tar.gz > /dev/null
cd ${DIRNAME}
./configure --prefix=$TRAVIS_BUILD_DIR/suitesparse
make -j10
make install
cd -
rm -rf $DIRNAME
