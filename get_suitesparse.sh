FILENAME=SuiteSparse-5.4.0
DIRNAME=SuiteSparse
if [ ! -f $FILENAME.tar.gz ]; then
wget "http://faculty.cse.tamu.edu/davis/SuiteSparse/$FILENAME.tar.gz"
fi

tar -xvf ${FILENAME}.tar.gz > /dev/null
#cd ${DIRNAME}
#./configure --prefix=$TRAVIS_BUILD_DIR/suitesparse
cd ${DIRNAME}/CXSparse
make -j10
make install
cd -
rm -rf $DIRNAME
