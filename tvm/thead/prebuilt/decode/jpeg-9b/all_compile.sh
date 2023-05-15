#!/bin/bash

mkdir build
mkdir build/include
cp *.h build/include
mkdir build/lib

make -f makefile.x86 clean
make -f makefile.x86 libjpeg.a
mkdir build/lib/x86
cp libjpeg.a build/lib/x86

make -f makefile.csky clean
make -f makefile.csky libjpeg.a
mkdir build/lib/csky
cp libjpeg.a build/lib/csky

make -f makefile.rv clean
make -f makefile.rv libjpeg.a
mkdir build/lib/rv
cp libjpeg.a build/lib/rv