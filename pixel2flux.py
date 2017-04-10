import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
import pyfits
from skimage.morphology import watershed
from skimage.feature import peak_local_max as plm
from matplotlib.colors import LogNorm


class BadApertureError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class PixelTarget:
    def __init__(self, epic, field, cad):
        """

        :param epic: str, EPIC of target
        :param field: str, field number
        :param cad: 'l' (long) or 's' (short)
        :return:
        """
        self.data = {'jd': [], 'rlc': [], 'x': [], 'y': [], 'cadence': cad, 'ra': [], 'dec': [], 'kmag': []}
        self.epic = epic
        self.field = field
        self.cad = cad
        self.saturated = False
        self.pixeldat = []
        self.cutoff_limit = 3
        self.start_aper = 5

    def read_fits(self, indir='pixel_files/'):
        """
        Read pixel-level image.
        :param indir: directory where fits file is saved. Download fits files with wget before starting.
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
        good = np.where(~np.isnan(time))  # only keep image frames with valid time
        time = time[good]
        flux = flux[good]
        header = hdulist[0].header

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
            print 'WARNING: saturated target'
            self.saturated = True
            self.start_aper = 8

        self.data['jd'] = np.array(time)
        self.pixeldat = np.array(flux)
        self.data['ra'] = RA
        self.data['dec'] = DEC
        self.data['kmag'] = kepmag

        self.remove_nan()

    def find_aper(self, cutoff_limit=None, saturated=True):
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
            print 'bad aper'
            # cutoff_limit too high so that 1 or 0 pixels are selected
            cutoff_limit -= 0.2
            print 'New cutoff_limit:', cutoff_limit
            cutoff = cutoff_limit * np.median(fsum)
            aper = np.array([fsum > cutoff])
            aper = 1 * aper  # arrays of 0s and 1s
        size = np.sum(aper)  # total no. of pixels in aperture

        if (saturated is True) and (self.saturated is True):
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
            newdist = ((coord[0] - aper.shape[1] / 2.) ** 2 + (coord[1] - aper.shape[2] / 2.) ** 2) ** 0.5
            if (newdist < dist) and (markers[coord[0], coord[1]] in np.unique(labels)):
                centnum = markers[coord[0], coord[1]]
                dist = newdist

        # in case there are more than one maxima
        if len(np.unique(labels)) > 2:
            labels = 1 * (labels == centnum)

        labels /= labels.max()
        while np.sum(labels) <= 1:
            print 'Flux centroid detection failed. Retrying with smaller cutoff_limit.'
            cutoff_limit -= 0.2
            labels = self.find_aper(cutoff_limit=cutoff_limit, saturated=False)

        if type(labels) == int:
            raise BadApertureError('Failed aperture. Giving up on target.')

        return labels

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

    def aper_phot(self, aper):
        """
        Get flux and centroids for given aperture.
        :param aper: Aperture mask from find_aper or find_circ_aper.
        :return:
        """

        aperture_fluxes = self.pixeldat * aper  # retain only pixels inside aper

        # sum over axis 2 and 1 (the X and Y positions), (axis 0 is the time)
        f_t = np.nansum(np.nansum(aperture_fluxes, axis=2), axis=1)  # subtract bg from this if needed

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
        ftot /= np.median(ftot)

        self.data['x'] = xc
        self.data['y'] = yc
        self.data['rlc'] = ftot
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
        return labels

    def remove_thrust(self, refx, refy, printtimes=False):
        """

        :param refx: x coords of reference star
        :param refy: y coords of reference star
        :param printtimes:
        :return:
        """
        # find and remove points in middle of thruster events, divide LC into segments
        diff_centroid = np.sqrt(np.diff(refx) ** 2 + np.diff(refy) ** 2)

        thruster_mask = diff_centroid < 2 * np.mean(diff_centroid)  # True=gap not in middle of thruster event
        thruster_mask1 = np.insert(thruster_mask, 0, False)  # True=gap before is not thruster event
        thruster_mask2 = np.append(thruster_mask, False)  # True=gap after is not thruster event
        thruster_mask = thruster_mask1 * thruster_mask2  # True=gaps before and after are not thruster events

        # time_thruster = self.data['jd'][thruster_mask]
        # diff_centroid_thruster = diff_centroid[thruster_mask[1:]]
        # firetimes = self.data['jd'][np.where(~thruster_mask)[0]]

        if printtimes:
            print 'fire times', self.data['jd'][np.where(~thruster_mask)[0]]
        self.data['x'] = self.data['x'][thruster_mask]
        self.data['y'] = self.data['y'][thruster_mask]
        self.data['jd'] = self.data['jd'][thruster_mask]
        self.pixeldat = self.pixeldat[thruster_mask]
        self.data['rlc'] = self.data['rlc'][thruster_mask]


def draw_aper(pixeltarg, aper, ax):
    """
    Plot aperture on pixel level image.
    :param pixeltarg: PixelTarget object
    :param aper: aperture array
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


def main(epic, field, cad):
    targ = PixelTarget(epic, field, cad)
    print 'Working on target ', epic
    targ.read_fits()
    print "Kep mag=", targ.data['kmag']

    # We won't need these steps once we have reference stars to determine where thruster fires are
    labels = targ.find_aper()
    ftot = targ.aper_phot(labels)

    targ.remove_thrust(refx=targ.data['x'], refy=targ.data['y'], printtimes=True)

    # remove thruster fires and do aperture photometry
    labels = targ.find_aper()
    ftot = targ.aper_phot(labels)
    circ_labels = targ.find_circ_aper(rad=targ.start_aper)
    ftot_circ = targ.aper_phot(circ_labels)

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    ax = draw_aper(targ, labels, ax)
    plt.savefig('outputs/'+epic+'_aper.png', dpi=150)
    plt.close('all')

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax = draw_aper(targ, circ_labels, ax)
    plt.savefig('outputs/' + epic + '_circ_aper.png', dpi=150)

    fig = plt.figure(figsize=(10,3))
    plt.plot(targ.data['jd'], ftot, 'b.')
    plt.plot(targ.data['jd'], ftot_circ, 'r.')
    plt.savefig('outputs/' + epic + '_lc.png', dpi=150)