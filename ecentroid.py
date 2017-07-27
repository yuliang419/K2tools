"""
Plots any selected star(s) to use as reference star(s) for detrending.
Writes good frame numbers into file named ref_cad.dat, and centroids into ref_centroid.dat (for one chosen star only,
but we can probably average a few).

"""
from pixel2flux import *
import matplotlib.pyplot as plt
import numpy as np
import sys
from collections import Counter

epics = np.loadtxt('starlist.txt', dtype=str)
field = sys.argv[1]

write = True
master_mask = []
master_x = []
master_y = []
for epic in epics:
    print epic
    targ = PixelTarget(epic, field, 'l')
    targ.read_fits(clean=False)
    print len(targ)
    if targ.kmag > 15:
        print 'Reference star too faint'
        continue
    elif targ.kmag <= 10:
        continue

    aperture = targ.find_aper()
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    draw_aper(targ, aperture.labels, ax)
    plt.savefig('outputs/'+epic+'_aper.png', dpi=150)

    ftot = targ.aper_phot(aperture)
    goodcads = targ.find_thrust(printtimes=False)
    print 'No thruster fire:', len(goodcads)
    master_mask.append(goodcads)
    master_x.append(targ.data['x'])
    master_y.append(targ.data['y'])

ref_x = np.nanmedian(master_x, axis=0)  # for entire light curve, no points removed
ref_y = np.nanmedian(master_y, axis=0)

master_mask = np.array(master_mask)
cnt = Counter(np.hstack(master_mask))
refcad = [k for k, v in cnt.iteritems() if v > 1]  # point is good if it's good in at least two targets

print 'no. of good points=', len(refcad)

# fig = plt.figure(figsize=(10,4))
# targ = PixelTarget(epics[1], field, 'l')
# targ.read_fits(clean=True)
# labels = targ.find_aper()
# ftot = targ.aper_phot(labels)
# plt.plot(targ.data['jd'], ftot, 'b.')
# plt.plot(targ.data['jd'][~thruster_mask], ftot[~thruster_mask], 'ro')
# plt.show()


for epic in epics:
    targ = PixelTarget(epic, field, 'l')
    targ.read_fits(clean=False)
    aperture = targ.find_aper()
    ftot = targ.aper_phot(aperture)
    bad = [i for i in targ.data['cadence'] if i not in refcad]

    fig, ax = plt.subplots(2, 2, figsize=(15,12))
    ax[0, 0].plot(targ.data['jd'], ftot, lw=0, marker='.')
    ax[0, 0].plot(targ.data['jd'][bad], ftot[bad], 'ro')
    ax[0, 0].set_xlabel('t')
    ax[0, 0].set_ylabel('Flux')

    colors = ['r', 'y', 'g', 'c', 'b', 'm']
    inds = np.linspace(0, len(targ.data['jd']), 7)
    for i in range(len(inds) - 1):
        start = int(inds[i])
        end = int(inds[i + 1])
        ax[0, 1].plot(targ.data['x'][start:end], targ.data['y'][start:end], marker='.', lw=0, color=colors[i])

    ax[0, 1].set_xlabel('x')
    ax[0, 1].set_ylabel('y')
    # ax[0, 1].plot(targ.data['x'][outlier], targ.data['y'][outlier], lw=0, marker='o', color='g')
    ax[0, 1].plot(targ.data['x'][bad], targ.data['y'][bad], 'ro')

    ax[1, 0].plot(targ.data['jd'], targ.data['x'], lw=0, marker='.')
    ax[1, 0].set_xlabel('t')
    ax[1, 0].set_ylabel('x')
    for time in targ.data['jd'][bad]:
        plt.axvline(x=time, color='r')
    ax[1, 1].plot(targ.data['jd'], targ.data['y'], lw=0, marker='.')
    ax[1, 1].set_xlabel('t')
    ax[1, 1].set_ylabel('y')
    plt.show()

if write:
    np.savetxt('ref_cad.dat', np.transpose(refcad), fmt='%d')
targ.remove_thrust(refcad)
ref_x = ref_x[refcad]
ref_y = ref_y[refcad]

fig = plt.figure()
plt.plot(ref_x, ref_y, 'b.')
plt.show()

if write:
    outfile = open('ref_centroid.dat', 'w')
    print>> outfile, '# cadence x y seg'

    print len(targ.data['cadence']), len(ref_x)
    seg = 0
    for i in range(0, len(targ.data['jd'])):
        if (targ.data['cadence'][i] > 0) and (targ.data['cadence'][i] - targ.data['cadence'][i - 1] > 1):
            seg += 1
        print>> outfile, targ.data['cadence'][i], ref_x[i], ref_y[i], seg
    outfile.close()
