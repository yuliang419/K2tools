#! /bin/bash

cnt=44

for ((cnt=75; cnt<=99; cnt++))
do
	wget -r -l1 -H -t1 -nd -N -A.gz -erobots=off https://archive.stsci.edu/pub/k2/target_pixel_files/c6/212200000/$cnt\000/
	gunzip *2122$cnt\*
done
