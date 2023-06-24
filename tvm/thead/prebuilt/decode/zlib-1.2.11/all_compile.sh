#!/bin/bash

mkdir build
mkdir build/include
cp zconf.h build/include
cp zlib.h build/include
mkdir build/lib

# x86
make -f makefile.x86 clean
make -f makefile.x86 static
mkdir build/lib/x86
cp libz.a build/lib/x86

# csky
make -f makefile.csky clean
make -f makefile.csky static
mkdir build/lib/csky
cp libz.a build/lib/csky

# rv
make -f makefile.rv clean
make -f makefile.rv static
mkdir build/lib/rv
cp libz.a build/lib/rv