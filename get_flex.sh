VER=2.6.4
FILENAME=flex-$VER
DIRNAME=$FILENAME
wget "https://github.com/westes/flex/releases/download/v${VER}/${FILENAME}.tar.gz"

if [ -z "$TRAVIS_BUILD_DIR" ]; then
  TRAVIS_BUILD_DIR=`pwd`
fi

tar -xvf ${FILENAME}.tar.gz
cd ${DIRNAME}
./configure --prefix=$TRAVIS_BUILD_DIR/flex
make -j10
make install
cd -
rm -rf ${DIRNAME}