"""
Plots any selected star(s) to use as reference star(s) for detrending.
Writes good frame numbers into file named ref_cad.dat, and centroids into ref_centroid.dat (for one chosen star only,
but we can probably average a few).

"""
from pixel2flux import *
import matplotlib.pyplot as plt
import numpy as np
import sys

epics = np.loadtxt('starlist.txt', dtype=str)
field = sys.argv[1]

write = True
master_mask = []
for epic in epics:
    print epic
    targ = PixelTarget(epic, field, 'l')
    targ.read_fits(clean=False)
    print len(targ)
    if targ.data['kmag'] > 15:
        print 'Reference star too faint'
        continue
    elif targ.data['kmag'] <= 10:
        continue

    labels = targ.find_aper()
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    draw_aper(targ, labels, ax)
    plt.savefig('outputs/'+epic+'_aper.png', dpi=150)

    ftot = targ.aper_phot(labels)
    thruster_mask, _ = targ.find_thrust(targ.data['x'], targ.data['y'], printtimes=False)
    print len(thruster_mask[thruster_mask > 0])
    master_mask.append(thruster_mask)

thruster_mask = np.nanmedian(master_mask, axis=0).astype(bool)
print 'no. of good points=', len(thruster_mask[thruster_mask > 0])

fig = plt.figure(figsize=(10,4))
targ = PixelTarget(epics[1], field, 'l')
targ.read_fits(clean=True)
labels = targ.find_aper()
ftot = targ.aper_phot(labels)
plt.plot(targ.data['jd'], ftot, 'b.')
plt.plot(targ.data['jd'][~thruster_mask], ftot[~thruster_mask], 'ro')
plt.show()

refcad = targ.data['cadence'][thruster_mask]

outlier = np.where(targ.data['jd'] < 2468)

fig, ax = plt.subplots(2, 2, figsize=(15,12))
ax[0, 0].plot(targ.data['jd'], ftot, lw=0, marker='.')
ax[0, 0].plot(targ.data['jd'][~thruster_mask], ftot[~thruster_mask], 'ro')
ax[0, 0].set_xlabel('t')
ax[0, 0].set_ylabel('Flux')

colors = ['r', 'y', 'g', 'c', 'b', 'm']
inds = np.linspace(0, len(targ.data['jd']), 7)
for i in range(len(inds) - 1):
    start = inds[i]
    end = inds[i + 1]
    ax[0, 1].plot(targ.data['x'][start:end], targ.data['y'][start:end], marker='.', lw=0, color=colors[i])

ax[0, 1].set_xlabel('x')
ax[0, 1].set_ylabel('y')
ax[0, 1].plot(targ.data['x'][outlier], targ.data['y'][outlier], lw=0, marker='o', color='g')
ax[0, 1].plot(targ.data['x'][~thruster_mask], targ.data['y'][~thruster_mask], 'ro')

ax[1, 0].plot(targ.data['jd'], targ.data['x'], lw=0, marker='.')
ax[1, 0].set_xlabel('t')
ax[1, 0].set_ylabel('x')
for time in targ.data['jd'][~thruster_mask]:
    plt.axvline(x=time, color='r')
ax[1, 1].plot(targ.data['jd'], targ.data['y'], lw=0, marker='.')
ax[1, 1].set_xlabel('t')
ax[1, 1].set_ylabel('y')
plt.show()

if write:
    np.savetxt('ref_cad.dat', np.transpose(refcad), fmt='%d')
targ.remove_thrust(thruster_mask)

if write:
    outfile = open('ref_centroid.dat', 'w')
    print>> outfile, '# cadence x y'
    for i in range(0, len(targ.data['jd'])):
        print>> outfile, targ.data['cadence'][i], targ.data['x'][i], targ.data['y'][i]
    outfile.close()
