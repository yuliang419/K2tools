import numpy as np
from scipy import ndimage as ndi
import pyfits
from skimage.morphology import watershed
from skimage.feature import peak_local_max as plm
from matplotlib.colors import LogNorm
import matplotlib
import logging
import glob
import multiprocessing
import sys
from astropy.stats import median_absolute_deviation
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class BadApertureError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class Aperture:
    def __init__(self, labels, aperinfo):
        self.aperinfo = aperinfo
        self.labels = labels

    def __str__(self):
        return 'Aperture shape: %s', self.aperinfo


class PixelTarget:
    def __init__(self, epic, field, cad):
        """
        :param epic: str, EPIC of target
        :param field: str, field number
        :param cad: 'l' (long) or 's' (short)
        :return:
        """
        self.data = {'jd': [], 'rlc': [], 'x': [], 'y': [], 'cadence': []}

        self.epic = epic
        self.field = field
        self.cad = cad
        self.saturated = False
        self.pixeldat = []
        self.bgframe = []
        self.readnoise = 0
        self.cutoff_limit = 3
        self.start_aper = 5
        self.poisson = np.nan
        self.ra = np.nan
        self.dec = np.nan
        self.kmag = np.nan
        self.aperinfo = 'unknown'

    def __len__(self):
        return len(self.data['jd'])

    def read_fits(self, indir='pixel_files/', clean=True, outliers=[]):
        """
        Read pixel-level image.
        :param indir: directory where fits file is saved. Download fits files with wget before starting.
        :param clean: if True, empty frames and known outliers will be removed.
        :param outliers: indices of known outliers to be removed.
        :return:
        """
        filename = indir + 'ktwo' + str(self.epic) + '-c' + str(self.field) + '_' + self.cad + 'pd-targ.fits'
        # flux may contain nans

        try:
            hdulist = pyfits.open(filename)
        except IOError:
            print 'fits file does not exist. Download all files from MAST before starting.'
            raise
        data = hdulist[1].data
        time = data['TIME']
        flux = data['FLUX']
        bg = data['FLUX_BKG']
        self.readnoise = hdulist[1].header['READNOIS']

        good = np.where(~np.isnan(time))  # only keep image frames with valid time
        time = time[good]
        flux = flux[good]
        bg = bg[good]

        header = hdulist[0].header
        # print len(time), len(flux), len(bg), len(self.data['cadence'])

        kepmag = header['Kepmag']
        # mostly derived from trial and error... may need to be modified for especially bright or faint stars
        if kepmag >= 18:
            self.cutoff_limit = 1.5
            self.start_aper = 2
        elif kepmag < 13:
            self.cutoff_limit = 5
            self.start_aper = 4
        else:
            self.cutoff_limit = 5 - (3.5 / 5) * (kepmag - 13)
            self.start_aper = 3

        RA = header['RA_OBJ']
        DEC = header['DEC_OBJ']
        # x = hdulist[2].header['CRVAL2P']  # x position of pixel
        # y = hdulist[2].header['CRVAL1P']  # y position of pixel

        if kepmag <= 10:
            logger.warning('Target %s is saturated', self.epic)
            self.saturated = True
            self.start_aper = 8

        self.data['jd'] = np.array(time)
        self.pixeldat = np.array(flux)
        self.ra = RA
        self.dec = DEC
        self.kmag = kepmag
        self.bgframe = np.array(bg)
        self.data['cadence'] = np.arange(len(flux), dtype=int)  # note: refcad can be used as indices before thruster
        #  or nan removal

        if clean:
            self.remove_nan()
            self.remove_known_outliers(outliers)

    def remove_known_outliers(self, inds):
        """
        Do this along with remove_nan before centroiding.
        :param inds: indices of known outliers.
        :return:
        """
        if len(inds) == 0:
            return
        self.data['jd'] = np.delete(self.data['jd'], inds)
        self.pixeldat = np.delete(self.pixeldat, inds, axis=0)
        self.bgframe = np.delete(self.bgframe, inds, axis=0)
        self.data['cadence'] = np.delete(self.data['cadence'], inds)

    def find_aper(self, cutoff_limit=None, faint=False):
        """
        Get custom shaped aperture.
        :param cutoff_limit: select all pixels brighter than cutoff_limit*median. Somehow this works better than
        sigma clipping.
        :param saturated: True if target is saturated.
        :return:
        """
        if cutoff_limit is None:
            cutoff_limit = self.cutoff_limit

        fsum = np.nansum(self.pixeldat, axis=0)  # sum over all images
        fsum -= min(fsum.ravel())
        cutoff = cutoff_limit * np.median(fsum)
        aper = np.array([fsum > cutoff])
        aper = 1 * aper  # convert to arrays of 0s and 1s (1s inside aper)

        while np.sum(aper) <= 1:
            logger.warning('%s : Cut off limit too high', self.epic)
            # cutoff_limit too high so that 1 or 0 pixels are selected
            cutoff_limit -= 0.2
            logger.info('Cut off limit set to %s', str(cutoff_limit))
            cutoff = cutoff_limit * np.median(fsum)
            aper = np.array([fsum > cutoff])
            aper = 1 * aper  # arrays of 0s and 1s
        size = np.sum(aper)  # total no. of pixels in aperture

        if (faint is False) and (self.saturated is True):
            min_dist = aper.shape[1] / 2  # minimum distance between two maxima
        else:
            min_dist = max([1, size ** 0.5 / 2.9])

        threshold_rel = 0.0002
        if cutoff_limit < 2.5:
            threshold_rel /= 10  # dealing with faint sources

        # detect local maxima in 2D image
        local_max = plm(fsum, indices=False, min_distance=min_dist, exclude_border=False,
                        threshold_rel=threshold_rel)  # threshold_rel determined by trial & error
        coords = plm(fsum, indices=True, min_distance=min_dist, exclude_border=False, threshold_rel=threshold_rel)

        dist = 100
        markers = ndi.label(local_max)[0]
        # in case of blended targets, use watershed to determine where to draw boundary
        labels = watershed(-fsum, markers, mask=aper[0])

        for coord in coords:
            # get centroid closest to centre of image
            newdist = ((coord[0] - aper.shape[1] / 2.) ** 2 + (coord[1] - aper.shape[2] / 2.) ** 2) ** 0.5
            if (newdist < dist) and (markers[coord[0], coord[1]] in np.unique(labels)):
                centnum = markers[coord[0], coord[1]]
                dist = newdist

        # in case there are more than one maxima
        if len(np.unique(labels)) > 2:
            labels = 1 * (labels == centnum)

        labels /= labels.max()
        iter = 0
        while np.sum(labels) <= 1:
            if iter >= 5:
                fig = plt.figure(figsize=(8, 8))
                plt.imshow(fsum, norm=LogNorm(), interpolation='none')
                plt.set_cmap('gray')
                plt.savefig('outputs/' + self.epic + '_badaper.png', dpi=150)
                plt.close()
                logger.exception('%s : Failed aperture. Target abandoned.', self.epic)
                raise BadApertureError('Failed aperture. Giving up on target.')

            logger.info('%s : Flux centroid detection failed. Retrying with smaller cutoff_limit. iter = %s',
                        self.epic, iter)
            cutoff_limit -= 0.2
            labels = self.find_aper(cutoff_limit=cutoff_limit, faint=True)
            iter += 1

        if type(labels) == int:
            logger.exception('%s : Failed aperture. Target abandoned.', self.epic)
            raise BadApertureError('Failed aperture. Giving up on target.')

        aperture = Aperture(labels, 'arbitrary')
        return aperture

    def remove_nan(self):
        """
        Remove all empty frames (why do these exist?)
        :return:
        """
        bad = []
        for i in range(0, len(self.data['jd'])):
            f = np.array(self.pixeldat[i])
            if np.isnan(f).all():
                bad.append(i)

        self.data['jd'] = np.delete(self.data['jd'], bad)
        self.pixeldat = np.delete(self.pixeldat, bad, axis=0)
        self.data['cadence'] = np.delete(self.data['cadence'], bad)
        self.bgframe = np.delete(self.bgframe, bad, axis=0)

    def aper_phot(self, aperture, getbg=True):
        """
        Get flux and centroids for given aperture.
        :param aperture: Aperture object returned by find_aper or find_circ_aper.
        :return:
        """
        aper = aperture.labels

        aperture_fluxes = self.pixeldat * aper  # retain only pixels inside aper

        # sum over axis 2 and 1 (the X and Y positions), (axis 0 is the time)
        f_t = np.nansum(np.nansum(aperture_fluxes, axis=2), axis=1)  # subtract bg from this if needed
        if getbg:
            aperture_bg = self.bgframe * aper
            bg_t = np.nansum(np.nansum(aperture_bg, axis=2), axis=1)

        # first make a matrix that contains the x and y positions
        x_pixels = [range(0, np.shape(aperture_fluxes)[2])] * np.shape(aperture_fluxes)[1]
        y_pixels = np.transpose([range(0, np.shape(aperture_fluxes)[1])] * np.shape(aperture_fluxes)[2])

        # multiply the position matrix with the aperture fluxes to obtain x_i*f_i and y_i*f_i
        xpos_times_flux = np.nansum(np.nansum(x_pixels * aperture_fluxes, axis=2), axis=1)
        ypos_times_flux = np.nansum(np.nansum(y_pixels * aperture_fluxes, axis=2), axis=1)

        # calculate centroids
        xc = xpos_times_flux / f_t
        yc = ypos_times_flux / f_t

        ftot = f_t
        ftot = np.array(ftot)
        if np.median(ftot) < 0:
            logger.exception('%s: total flux < 0', self.epic)
            raise ValueError('Bad fits file: total flux < 0')

        na = np.sum(aper)
        if getbg:
            if self.cad == 's':
                t_int = 6.02*9
                numreads = 9
            else:
                t_int = 6.02*270
                numreads = 270
            poisson = np.sqrt(np.median(ftot)*t_int + np.median(bg_t)*t_int + na*self.readnoise*numreads) / \
                      (np.median(ftot)*t_int)
            self.poisson = poisson  # fractional poisson noise

        ftot /= np.median(ftot)
        bad = np.where(ftot <= 0)
        xc[bad] = -100
        yc[bad] = -100

        # good = np.where(ftot > 0)
        # self.data['jd'] = self.data['jd'][good]
        # self.data['cadence'] = self.data['cadence'][good]
        self.data['x'] = xc
        self.data['y'] = yc
        self.data['rlc'] = ftot
        if getbg:
            self.data['bg'] = bg_t
        else:
            self.data['bg'] = []
        self.aperinfo = aperture.aperinfo

        return ftot

    def find_circ_aper(self, rad):
        """
        Get circular aperture.
        :param rad: radius of circular aperture
        :return:
        """
        # use centroid position found from custom aperture
        # first make a matrix that contains the x and y positions
        # remember x is second coordinate
        if len(self.data['x']) == 0:
            print 'Error: centroid coordinates not found. Run aper_phot first to find centroid.'
            return
        med_x = np.median(self.data['x'])
        med_y = np.median(self.data['y'])
        frame = self.pixeldat[0]
        x_pixels = [range(0, np.shape(frame)[1])] * np.shape(frame)[0]
        y_pixels = np.transpose([range(0, np.shape(frame)[0])] * np.shape(frame)[1])

        inside = ((x_pixels - med_x) ** 2. + (y_pixels - med_y) ** 2.) < rad ** 2.
        labels = 1 * inside
        aperinfo = 'circular %s' % (rad)
        aperture = Aperture(labels, aperinfo)
        return aperture

    def find_thrust(self, printtimes=False):
        """
        Find points in middle of thruster events. For use on reference stars only
        :param printtimes: True if you want to see list of identified thruster fire times.
        :param remove: True if you want thruster fire points to be removed. Otherwise returns thruster_mask without
        altering target data.
        :return:
        refcad: array of cadence numbers that don't fall in thruster fires
        """

        refx = self.data['x']
        refy = self.data['y']
        if len(refx) < 1:
            raise ValueError('No centroids available. Run aper_phot first.')
        diff_centroid = np.sqrt(np.diff(refx) ** 2 + np.diff(refy) ** 2)

        thruster_mask = diff_centroid < 4 * np.median(diff_centroid)  # True=gap not in middle of thruster event
        thruster_mask1 = np.insert(thruster_mask, 0, False)  # True=gap before is not thruster event
        thruster_mask2 = np.append(thruster_mask, False)  # True=gap after is not thruster event
        thruster_mask = thruster_mask1 * thruster_mask2  # True=gaps before and after are not thruster events

        # time_thruster = self.data['jd'][thruster_mask]

        if printtimes:
            print 'thruster fire frame numbers', self.data['cadence'][np.where(~thruster_mask)[0]]
        return self.data['cadence'][np.where(thruster_mask)]

    def remove_thrust(self, refcad):
        """
        Remove thruster fires identified from reference stars.
        :param refcad: cadence numbers of points free from thruster fires.
        :return:
        """
        # refcad = np.intersect1d(refcad, self.data['cadence'])  # exclude frames that have already
        # been cleaned out in remove_nan
        thruster_mask = np.array([self.data['cadence'][i] in refcad for i in range(len(self.data['cadence']))])
        self.data['jd'] = self.data['jd'][thruster_mask]
        self.pixeldat = self.pixeldat[thruster_mask]
        if len(self.bgframe) > 0:
            self.bgframe = self.bgframe[thruster_mask]
        self.data['cadence'] = self.data['cadence'][thruster_mask]


def draw_aper(pixeltarg, aper, ax):
    """
    Plot aperture on pixel level image.
    :param pixeltarg: PixelTarget object
    :param aper: aperture array (Aperture.labels)
    :param ax: plot handle
    :return:
    """
    flux = pixeltarg.pixeldat
    fsum = np.nansum(flux, axis=0)
    fsum -= min(fsum.ravel())

    plt.imshow(fsum, norm=LogNorm(), interpolation='none')
    plt.set_cmap('gray')

    # find edge pixels in each row
    ver_seg = np.where(aper[:, 1:] != aper[:, :-1])
    hor_seg = np.where(aper[1:, :] != aper[:-1, :])

    l = []
    for p in zip(*hor_seg):
        l.append((p[1], p[0] + 1))
        l.append((p[1] + 1, p[0] + 1))
        l.append((np.nan, np.nan))

    for p in zip(*ver_seg):
        l.append((p[1] + 1, p[0]))
        l.append((p[1] + 1, p[0] + 1))
        l.append((np.nan, np.nan))

    seg = np.array(l)
    x0 = -0.5
    x1 = aper.shape[1] + x0
    y0 = -0.5
    y1 = aper.shape[0] + y0

    if len(seg) != 0:
        seg[:, 0] = x0 + (x1 - x0) * seg[:, 0] / aper.shape[1]
        seg[:, 1] = y0 + (y1 - y0) * seg[:, 1] / aper.shape[0]
        ax.plot(seg[:, 0], seg[:, 1], color='r', zorder=10, lw=2.5)

    ax.set_xlim(0, aper.shape[1])
    ax.set_ylim(0, aper.shape[0])
    return ax


def test(epic, field, cad, refcadfile):
    targ = PixelTarget(epic, field, cad)
    print 'Working on target ', epic
    targ.read_fits()
    print "Kep mag=", targ.kmag

    refcad = np.loadtxt(refcadfile, dtype=int)
    aperture = targ.find_aper()
    ftot = targ.aper_phot(aperture)
    fig = plt.figure(figsize=(15,4))
    plt.plot(targ.data['jd'], ftot, 'b.')

    # remove thruster fires and do aperture photometry
    targ.remove_thrust(refcad)
    ftot = targ.aper_phot(aperture)
    plt.plot(targ.data['jd'], ftot, 'r.')
    plt.show()
    circ_aper = targ.find_circ_aper(rad=targ.start_aper)
    ftot_circ = targ.aper_phot(circ_aper)

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    ax = draw_aper(targ, aperture.labels, ax)
    plt.savefig('outputs/'+epic+'_aper.png', dpi=150)
    plt.close('all')

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax = draw_aper(targ, circ_aper.labels, ax)
    plt.savefig('outputs/' + epic + '_circ_aper.png', dpi=150)

    fig = plt.figure(figsize=(10,3))
    plt.plot(targ.data['jd'], ftot, 'b.')
    plt.plot(targ.data['jd'], ftot_circ, 'r.')
    plt.savefig('outputs/' + epic + '_lc.png', dpi=150)
    plt.pause(0.01)


def main(epic, field, cad, refcad):
    targ = PixelTarget(epic, field, cad)
    targ.read_fits()

    targ.remove_thrust(refcad)
    aperture = targ.find_aper()

    ftot = targ.aper_phot(aperture)

    mad = median_absolute_deviation(ftot)
    best_rad = 'arbitrary'
    best_aper = aperture.labels

    ftot_all = {'arbitrary': ftot}
    poisson_all = {'arbitrary': targ.poisson}
    rads = np.arange(targ.start_aper-1, targ.start_aper+3)
    for r in rads:
        circ_apers = targ.find_circ_aper(rad=r)
        ftot_circ = targ.aper_phot(circ_apers)
        ftot_all[str(r)] = ftot_circ
        poisson_all[str(r)] = targ.poisson
        mad_new = median_absolute_deviation(ftot_circ)
        if mad_new < mad:
            mad = mad_new
            best_rad = str(r)
            best_aper = circ_apers.labels

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    ax = draw_aper(targ, best_aper, ax)
    ax.set_title('Rad ='+best_rad)
    plt.savefig('outputs/' + epic + '_aper.png', dpi=150)
    plt.pause(0.01)
    plt.close()

    return targ, ftot_all, poisson_all, best_rad


def extract_multi(args, outdir='rawlc/'):
    epic = args[0]
    field = args[1]
    cad = args[2]
    refcad = args[3]
    logger.info('Working on %s', epic)
    try:
        targ, ftot, poisson_all, best_rad = main(epic, field, cad, refcad)
    except (BadApertureError, ValueError):
        return
    outfile = open(outdir+epic+'_rawlc.dat', 'w')
    rads = sorted(ftot.keys())

    logger.info('%s: best aperture = %s', epic, best_rad)
    headers = rads[:]
    headers[np.where(np.array(rads) == best_rad)[0][0]] = 'rlc'  # make sure the best aperture column is named 'rlc'

    print>>outfile, '# cad  jd   %s  %s  %s  %s  %s  x   y' % (headers[0], headers[1], headers[2], headers[3],
                                                              headers[4])
    for i in range(len(targ)):
        print>>outfile, targ.data['cadence'][i], targ.data['jd'][i], ftot[rads[0]][i], ftot[rads[1]][i], ftot[rads[2]][
            i], ftot[rads[3]][i], ftot[rads[4]][i], targ.data['x'][i], targ.data['y'][i]

    outfile.close()

    with open('poisson.txt', 'a') as outfile:
        print>>outfile, epic, targ.kmag, poisson_all[best_rad]


if __name__ == '__main__':

    epics = np.loadtxt('filelist.txt', dtype=str)
    field = '05'
    cad = 'l'
    refcad = np.loadtxt('ref_cad.dat', dtype=int)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = multiprocessing.get_logger()
    hdlr = logging.FileHandler('pixel2flux.log', mode='w')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr) 
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel(logging.INFO)
    logger.info('Process started')

    with open('poisson.txt', 'a') as outfile:
        print>>outfile, '# epic kepmag  poisson'

    pool = multiprocessing.Pool(processes=3)
    TASK = [(epics[i], field, cad, refcad) for i in range(len(epics))]
    pool.map(extract_multi, TASK)



