#!/bin/bash

# Ensure we are not using MacPorts, but the native OS X compilers
export PATH=$PREFIX/bin:/bin:/sbin:/usr/bin:/usr/sbin

# This is really important. Conda build sets the deployment target to 10.5 and
# this seems to be the main reason why the build environment is different in
# conda compared to compiling on the command line. Linking against libc++ does
# not work for old deployment targets.
export MACOSX_DEPLOYMENT_TARGET="10.9"

mkdir -vp ${PREFIX}/lib;
./bootstrap.sh \
    --prefix="${PREFIX}/" \
    --with-libraries=python

# On Linux we use g++ and its libstdc++. On OS X we use clang (which is the
# default anyway) with its libc++. (On OS X, libstdc++ does not work for C++11,
# because it is too old.)
if [ "$OSX_ARCH" == "" ]; then
    # Linux
    ./b2 \
        --layout=tagged \
        cxxflags="-std=c++11" \
        stage;
else
    # OS X
    ./b2 \
        --layout=tagged \
        toolset=clang \
        cxxflags="-std=c++11 -stdlib=libc++" \
        linkflags="-stdlib=libc++" \
        stage;
fi

cp stage/lib/libboost_* ${PREFIX}/lib/

exit 0
