#! /bin/bash

for ((xx=22; xx<=28; xx++))
do
	for ((yy=10; yy=99; yy++))
	do
		wget -r -l1 -H -nd -A.gz -erobots=off https://archive.stsci.edu/pub/k2/target_pixel_files/c6/21$xx\00000/$yy\000/
	done

	for ((y=0; y=9; yy++))
	do
		wget -r -l1 -H -nd -A.gz -erobots=off https://archive.stsci.edu/pub/k2/target_pixel_files/c6/21$xx\00000/0$y\000/
	done
	gunzip ktwo*gz
done

