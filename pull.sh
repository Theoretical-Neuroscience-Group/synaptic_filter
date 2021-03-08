#!/bin/bash

folder=$1
mkdir ./../$folder

#ssh jegminat@euler.ethz.ch bash ./$folder/run_post.sh

# copy output
scp -r jegminat@euler.ethz.ch:/cluster/home/jegminat/$folder/* ./../$folder

# clean up
cat ./../$folder/lsf.o* >> ./../$folder/log.txt
rm ./../$folder/lsf.*

#echo run main.py post on local machine and save dataframe DF
#python ./../$folder/maintoy.py -i -2

# rename
mv ./../$folder ./../"c_$(date '+%d-%b-%Y-')$folder"
