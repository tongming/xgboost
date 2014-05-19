#!/bin/bash
wget http://research.microsoft.com/en-us/um/beijing/projects/mslr/data/MSLR-WEB30K.zip
unzip MSLR-WEB30K.zip -d ./MSLR
mv -f ./MSLR/Fold1/*.txt .
