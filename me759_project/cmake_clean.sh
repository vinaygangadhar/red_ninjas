#!/bin/bash

##remove cmake contents in current directory
TOP=`pwd`

CACHE=CMakeCache.txt
FILES=CMakeFiles
INSTALL=cmake_install.cmake
MK=Makefile
BIN=bin

echo "Cleaning CMAKE docs in " $TOP
rm -rf $CACHE $FILES $INSTALL $MK 
rm -rf $BIN/*

## go through the project directory and do the same
for d in project*/; do
	echo "Cleaning in project directory $d"
	cd $d && rm -rf $CACHE $FILES $INSTALL $MK

done

