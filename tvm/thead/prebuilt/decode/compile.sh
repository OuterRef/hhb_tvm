#!/bin/bash

mkdir install
mkdir install/include
mkdir install/lib
mkdir install/lib/x86
mkdir install/lib/csky
mkdir install/lib/rv

# compile libjpeg
cd jpeg-9b
./all_compile.sh
cp build/include/* ../install/include
cp build/lib/x86/* ../install/lib/x86
cp build/lib/csky/* ../install/lib/csky
cp build/lib/rv/* ../install/lib/rv
cd ..

# compile zlib
cd zlib-1.2.11
./all_compile.sh
cp build/include/* ../install/include
cp build/lib/x86/* ../install/lib/x86
cp build/lib/csky/* ../install/lib/csky
cp build/lib/rv/* ../install/lib/rv
cd ..

# compile libpng
cd lpng1637
./all_compile.sh
cp build/include/* ../install/include
cp build/lib/x86/* ../install/lib/x86
cp build/lib/csky/* ../install/lib/csky
cp build/lib/rv/* ../install/lib/rv
cd ..

