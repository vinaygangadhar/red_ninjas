#! /bin/bash

TOP=/home/vinayg/red_ninjas/project
FD=$TOP/facedetect
BIN=$TOP/facedetect-bin

rm -rf $b1 $b2 $b3 $b4 $b5

b1=base1-shmem
b2=base2-pinned
b3=base3-fastmath
b4=base4-womaxreg
b5=base5-diverge

###base 1
echo $'\n'
echo "\$\$ Executing Facedetect with base 1  \$\$"
cd $TOP/facedetect
$BIN/facedetect-$b1 group.pgm img.log > $b1.stat

###base 2
echo $'\n'
echo "\$\$ Executing Facedetect with base 2  \$\$"
cd $TOP/facedetect
$BIN/facedetect-$b2 group.pgm img.log > $b2.stat

###base 3
echo $'\n'
echo "\$\$ Executing Facedetect with base 3  \$\$"
cd $TOP/facedetect
$BIN/facedetect-$b3 group.pgm img.log > $b3.stat

###base 4
echo $'\n'
echo "\$\$ Executing Facedetect with base 4  \$\$"
cd $TOP/facedetect
$BIN/facedetect-$b4 group.pgm img.log > $b4.stat

###base 5
echo $'\n'
echo "\$\$ Executing Facedetect with base 5 \$\$"
cd $TOP/facedetect
$BIN/facedetect-$b5 group.pgm img.log > $b5.stat

