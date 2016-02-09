#! /bin/bash

for xx in {22..28}
do
	for yy in {10..99}
	do
		wget -r -l1 -H -nd -A.gz -erobots=off https://archive.stsci.edu/pub/k2/target_pixel_files/c6/21$xx\00000/$yy\000/
	done

	for y in {0..9}
	do
		wget -r -l1 -H -nd -A.gz -erobots=off https://archive.stsci.edu/pub/k2/target_pixel_files/c6/21$xx\00000/0$y\000/
	done
	gunzip ktwo*gz
done

