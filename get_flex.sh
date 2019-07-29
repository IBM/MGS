wget "https://github.com/westes/flex/releases/download/v2.6.4/flex-2.6.4.tar.gz"
tar -xvf flex-2.6.4.tar.gz
cd flex-2.6.4
./configure --prefix=$TRAVIS_BUILD_DIR/flex
make -j10
make install
cd -
rm -rf flex-2.6.4
