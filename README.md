# K2tools
Set of Python routines to extract flux from raw K2 images and remove thruster fire events.
This code borrows elements from Vincent van Eylen, Roberto Sanchis-Ojeda and Andrew Vanderburg's K2 photometry pipelines.

Run ecentroid.py on a group of reference stars to obtain thruster fire times and reference centroid positions. Then run pixel2flux.py to extract light curves from pixel-level images. 