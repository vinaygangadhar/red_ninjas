#! /bin/bash

TOP=/home/vinayg/red_ninjas/project
FD=$TOP/facedetect
BIN=$TOP/bin/facedetect-optim


b1=1024
b2=512
b3=256
b4=128
b5=64
b6=32
b7=25


rm -rf $b1.stat $b2.stat $b3.stat $b4.stat $b5.stat $b6.stat $b7.stat

##
#####base 1
##echo $'\n'
##echo "\$\$ Executing Facedetect with 1024x1024  \$\$"
##$BIN $b1\_ks.pgm img.log > $b1.stat
##
#####base 2
##echo $'\n'
##echo "\$\$ Executing Facedetect with 512x512  \$\$"
##$BIN $b2\_ks.pgm img.log > $b2.stat
##
#####base 3
##echo $'\n'
##echo "\$\$ Executing Facedetect with 256x256  \$\$"
##$BIN $b3\_ks.pgm img.log > $b3.stat
##
#####base 4
##echo $'\n'
##echo "\$\$ Executing Facedetect with 128x128  \$\$"
##$BIN $b4\_ks.pgm img.log > $b4.stat
##
#####base 5
##echo $'\n'
##echo "\$\$ Executing Facedetect with 64x64  \$\$"
##$BIN $b5\_ks.pgm img.log > $b5.stat
##
#####base 5
##echo $'\n'
##echo "\$\$ Executing Facedetect with 32x32  \$\$"
##$BIN $b6\_ks.pgm img.log > $b6.stat
##
##
#####base 5
##echo $'\n'
##echo "\$\$ Executing Facedetect with 25x25  \$\$"
##$BIN $b7\_ks.pgm img.log > $b7.stat


