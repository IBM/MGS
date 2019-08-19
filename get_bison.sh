#FILENAME=bison-2.4.1
FILENAME=bison-3.3.1
DIRNAME=$FILENAME
wget --tries=4 \
    "http://ftp.vim.org/ftp/gnu/bison/${FILENAME}.tar.gz"
if [ ! -f $FILENAME.tar.gz ]; then
  wget \
    "https://ftp.gnu.org/gnu/bison/${FILENAME}.tar.gz"
fi

if [ -z "$TRAVIS_BUILD_DIR" ]; then
  TRAVIS_BUILD_DIR=`pwd`
fi

tar -xvf ${FILENAME}.tar.gz > /dev/null
cd ${DIRNAME}
./configure --prefix=$TRAVIS_BUILD_DIR/bison
make -j10
make install
cd -
rm -rf $DIRNAME
