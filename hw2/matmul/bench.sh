#!/bin/bash

make clean > /dev/null
echo
echo "-g flag"
make CFLAGS_OPT=-g > /dev/null
./main

make clean > /dev/null
echo
echo "-O3 flag"
make CFLAGS_OPT=-O3 > /dev/null
./main

make clean > /dev/null
