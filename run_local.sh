#!/bin/bash

folder=$1
file=$2
kwargs=$4
echo syn $file to folder: $folder
ssh jegminat@euler.ethz.ch mkdir ./$folder

rsync -a --ignore-existing ./* jegminat@euler.ethz.ch:/cluster/home/jegminat/$folder/

i=0
N=$3
echo runs $N
while [ $i -le $((N-1)) ]
    do
    echo submit job with args $folder -M $3 -i $i
    ssh jegminat@euler.ethz.ch bash ./$folder/run_remote.sh $folder $file i=$i $kwargs
    ((i++))
done
