#!/bin/bash
wget http://research.microsoft.com/en-us/um/beijing/projects/letor/LETOR4.0/Data/MQ2007-list.rar
unrar x MQ2007-list.rar
mv -f MQ2007-list/Fold1/*.txt .
