# K2tools
Set of Python routines to extract flux from raw K2 images and remove thruster fire events.
This code borrows elements from Vincent van Eylen, Roberto Sanchis-Ojeda and Andrew Vanderburg's K2 photometry pipelines.

Run ecentroid.py on a group of reference stars to obtain thruster fire times and reference centroid positions. Then run pixel2flux.py to extract light curves from pixel-level images. 

# Usage
Before processing any K2 field, select a few well-behaved guide stars from the field and write their EPIC numbers into a file called guide_stars.txt. Then run the following to generate a list of good cadence numbers and reference centroids:

```
python ecentroid.py [field number]
```

In the same directory as the reference centroid file, make a list of all EPIC numbers of targets in the field called "Keplc2.ls". Then run pixel2flux.py to automatically select the optimal aperture, perform aperture photometry, and extract flux from pixel-level images. 
