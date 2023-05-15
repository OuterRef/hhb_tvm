#!/bin/bash -x

shl_dir=../../../../install_nn2
c_header_dir=../../../hhb/config/process/include/
decode_header_dir=../../decode/install/include/
c_code_dir=../../../hhb/config/process/src/

gcc -O2 -g -c ${c_code_dir}io.c -o io.o -I ${c_header_dir} -I ${shl_dir}/include
gcc -O2 -g -c ${c_code_dir}process.c -o process.o -I ${decode_header_dir} -I ${c_header_dir} -I ${shl_dir}/include
g++ -c ../cmd_parse/cmd_parse.cpp -o cmd_parse.o
ar -rcs libprebuilt_runtime.a process.o io.o cmd_parse.o
