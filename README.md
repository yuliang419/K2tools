# K2tools
Set of Python routines to extract flux from raw K2 images, clean and vet light curves.
This code borrows elements from Vincent van Eylen, Roberto Sanchis-Ojeda and Andrew Vanderburg's K2 photometry pipelines.

circ.py is main pipeline that extracts flux from downloaded .fits files and produces detrended ligth curves, but loops over all files. circ_test.py is a simpler version that runs on one specified .fits file at a time for debugging purposes.

analysis.py is the vetting code to be run after BLS. 
