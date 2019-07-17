wget "https://ftp.gnu.org/gnu/bison/bison-2.4.1.tar.gz"
FILENAME=bison-2.4.1
tar -xvf ${FILENAME}.tar.gz
cd ${FILENAME}
./configure --prefix=$TRAVIS_BUILD_DIR/bison
make -j10
make install
cd -
rm -rf $FILENAME
