#! /bin/bash

files=*wget.txt
for entry in $files
do
	echo $entry
	chmod +x $entry
	./$entry
done

gunzip ktwo*

ls ktwo* > targs.dat