#!/bin/bash

mkdir build
mkdir build/include
cp *.h build/include
mkdir build/lib

# x86
make -f makefile.x86 clean
make -f makefile.x86 libpng.a
mkdir build/lib/x86
cp libpng.a build/lib/x86

# csky
make -f makefile.csky clean
make -f makefile.csky libpng.a
mkdir build/lib/csky
cp libpng.a build/lib/csky

# rv
make -f makefile.rv clean
make -f makefile.rv libpng.a
mkdir build/lib/rv
cp libpng.a build/lib/rv