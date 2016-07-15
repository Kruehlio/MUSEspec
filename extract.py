# -*- coding: utf-8 -*-

""" 
Extraction from a data cube along specific dimensions
"""

import numpy as np
import logging
import time

from .io import pdfout

logfmt = '%(levelname)s [%(asctime)s]: %(message)s'
datefmt= '%Y-%m-%d %H:%M:%S'
formatter = logging.Formatter(fmt=logfmt,datefmt=datefmt)
logger = logging.getLogger('__main__')
logging.root.setLevel(logging.DEBUG)
ch = logging.StreamHandler() #console handler
ch.setFormatter(formatter)
logger.handlers = []
logger.addHandler(ch)


RESTWL = {'oiia' : 3727.092, 'oii':3728.30, 'oiib' : 3729.875, 'hd': 4102.9351,
          'hg' : 4341.69, 'hb' : 4862.68, 'niia':6549.86,
          'oiiia' : 4960.30, 'oiiib': 5008.240, 'oiii': 4990., 'ha' : 6564.61,
          'nii': 6585.27, 'siia':6718.29, 'siib':6732.68,
          'neiii' : 3869.81}

c = 2.99792458E5


def extract3d(self, wl1=None, wl2=None):
    """Extracts a subcube between two wavelengths

    Parameters
    ----------
        wl1 : float
            Lower wavelength
        wl2 : float
            Upper wavelength

    Returns
    -------
        subcube : np.array
            2-d subcube between wl1 and wl2
        subwl : np.array
            Wavelengths of the subcube
    """

    pix1 = self.wltopix(wl1)
    pix2 = max(pix1+1, self.wltopix(wl2)+1)
    subcube = self.data[pix1:pix2]
    subwl = self.wave[pix1:pix2]
    return subcube, subwl



def extract2d(s3d, wl1='', wl2='', z=None, line=None, dv=100,
                 meth = 'sum', sC=0, v=0):
    """Extracts a single plane, summed/averaged/medianed between two wave-
    lenghts, or for a single emission line

    Parameters
    ----------
        wl1 : float
            Lower wavelength
        wl2 : float
            Upper wavelength
        z : float
            Redshift (default None)
        line : str
            Emission line to extract (default None)
        method : str
            Method to cobine, default sum, options average, median, error
            If method = error, produces the error plane
        dv : float
            Velocity width in km/s around which to sum the line flux
            default 100 kms (corresponds to an interval of +/- 200 km/s
            to be extracted)
        sC : int
            subtract the continuum in the extracted plane (default 0 = no)
    Returns
    -------
        currPlane : np.array
            2-d plane combining the data between wl1 and wl2
    """

    if z == None: z = s3d.z
    if z == None: z = 0
    if line in ['Halpha', 'Ha', 'ha']:
        wlline = RESTWL['ha'] * (1+z)
        cont1 = (RESTWL['niia']* (1+z)) - 2*dv/c*wlline
        cont2 = (RESTWL['nii'] * (1+z)) + 2*dv/c*wlline
    elif line in ['Hbeta', 'Hb', 'hb']:
        wlline = RESTWL['hb'] * (1+z)
        cont1 = wlline - 2*dv/c*wlline
        cont2 = wlline + 2*dv/c*wlline
    elif line in ['OIII', 'oiii', 'oiiib']:
        wlline = RESTWL['oiiib'] * (1+z)
        cont1 = wlline - 2*dv/c*wlline
        cont2 = wlline + 2*dv/c*wlline
    elif line in ['NII', 'nii', 'niib']:
        wlline = RESTWL['nii'] * (1+z)
        cont1 = (6549.86 * (1+z)) - 2*dv/c*wlline
        cont2 = wlline + 2*dv/c*wlline
    elif line in ['SIIa', 'siia']:
        wlline = RESTWL['siia'] * (1+z)
        wlline2 = RESTWL['siib'] * (1+z)
        cont1 = wlline - 2*dv/c*wlline
        cont2 = wlline2 + 2*dv/c*wlline2
    elif line in ['SIIb', 'siib']:
        wlline1 = RESTWL['siia'] * (1+z)
        wlline = RESTWL['siib'] * (1+z)
        cont1 = wlline1 - 2*dv/c*wlline1
        cont2 = wlline + 2*dv/c*wlline
    else:
        cont1 = wl1 - 5
        cont2 = wl2 + 5

    if line != None:
        wl1 = wlline - 2*dv/c*wlline
        wl2 = wlline + 2*dv/c*wlline
        logger.info( 'Summing data with %.1f < lambda < %.1f ' %(wl1, wl2))

    pix1 = s3d.wltopix(wl1)
    pix2 = max(pix1+1, s3d.wltopix(wl2))
    cpix1 = s3d.wltopix(cont1)
    cpix2 = s3d.wltopix(cont2)

    if meth in ['average', 'sum']:
        currPlane = np.nansum(s3d.data[pix1:pix2], axis = 0)
    elif meth == 'median':
        currPlane = np.nanmedian(s3d.data[pix1:pix2], axis = 0)
    elif meth == 'error':
        currPlane = np.nansum(s3d.erro[pix1:pix2]**2, axis = 0)**0.5

    if sC == 1:
        if s3d.verbose > 0:
            logger.info( 'Subtracting continuum using lambda < %.1f and lambda > %.1f' \
                %(cont1, cont2))
        currPlane = subtractCont(s3d, currPlane, pix1, pix2, cpix1, cpix2)

    if meth in ['sum', 'int']:
        currPlane = currPlane * s3d.wlinc
    return currPlane
    
    

def extract1d(s3d, ra=None, dec=None, x=None, y=None, 
             radius=None, size= None, verbose=1,
             method='sum', total=False, ell=None, exmask=None,
             pexmask=False):
    """Extracts a single spectrum at a given position.
    If neither radius, total or ell is given, extracts a single spaxel at
    ra, dec, or (if ra, dec are not provided), x and y. If radius, ell or
    total is given, first creates an extraction mask, containing the spaxels
    to be extracted and finally combines them.

    Parameters
    ----------
        ra : float
            Right ascension in cube of central pixel
        dec : float
            Declination in cube of central pixel
        x : int
            x pixel value in cube of central pixel (NB: Python notation)
        y : int
            y pixel value in cube of central pixel  (NB: Python notation)
        radius : float
            Radius around central pixel to extract in arcsec
        size : integer
            Size of square around central pixel to extract in pixel
        method : str
            How to extract multiple pixels
        total : bool
            Whether to extract the full cube. If an object mask is present,
            i.e., has been created previously, then it only extracts the
            spaxels in the object mask.
        ell : list
            Parameters of the ellipse for extraction. Ellipse is given in
            pixel coordinates with format [posx, posy, a, b, theta] where
            posx, posy are the central pixels, a, b the semi-major and minor
            axis length, and theta the rotation angle of the ellipse.
        exmask : np.array
            If present, uses the given array to extract specific spaxels.
            Format should be 0 for spaxels to ignore, 1 for spaxels to extract.
            Must be the same shape as the data cube.
        pexmask : bool
            plot extraction mask

    Returns
    -------
        self.wave : np.array
            Wavelength of the extracted spectrum
        spec : np.array
            Flux values of extracted/combined spaxels
        error : np.array
            Error values of extracted/combined spaxels
    """

    if ra != None and dec != None:
        try:
            posx, posy = s3d.skytopix(ra, dec)
        except TypeError:
            posx, posy = s3d.sexatopix(ra, dec)
    elif ell != None:
        posx, posy, a, b, theta = ell
    elif total == False:
        posx, posy = x, y

    if radius==None and size==None and total==False and ell==None:
        if verbose == 1:
            logger.info('Extracting pixel %i, %i' %(posx, posy))
        spec = np.array(s3d.data[:,posy, posx])
        err  = np.array(s3d.erro[:,posy, posx])
        return s3d.wave, spec, err

    if total==False and radius!=None:
        if verbose == 1:
            logger.info('Creating extraction mask with radius %i arcsec' %radius)
        radpix = radius / s3d.pixsky
        x, y = np.indices(s3d.data.shape[0:2])

        exmask = np.round(((x - posy)**2  +  (y - posx)**2)**0.5)
        exmask[exmask <= radpix] = 1
        exmask[exmask > radpix] = 0

    elif total==False and size!=None:
        if verbose == 1:
            logger.info('Extracting spectrum with size %ix%i pixel' \
            %(2*size+1, 2*size+1))
        miny = max(0, posy-size)
        maxy = min(s3d.leny-1, posy+size+1)
        minx = max(0, posx-size)
        maxx = min(s3d.lenx-1, posx+size+1)

        spec = np.array(s3d.data[:, miny:maxy, minx:maxx])
        err  = np.array(s3d.erro[:, miny:maxy, minx:maxx])

        rspec = np.nansum(np.nansum(spec, axis = 1), axis=1)
        rerr = np.nansum(np.nansum(spec**2, axis = 1), axis=1)**0.5

        return s3d.wave, rspec, rerr

    elif total==False and ell!=None:
        # Ellipse is in pixel coordinates
        logger.info( 'Creating extraction ellipse')
        x, y = np.indices(s3d.data.shape[0:2])
        ell = ((x - posx) * np.cos(theta) + (y-posy) *np.sin(theta))**2 / a**2 \
             +((x - posx) * np.sin(theta) - (y-posy) *np.cos(theta))**2 / b**2
        exmask = np.round(ell)
        exmask[exmask <= 1] = 1
        exmask[exmask > 1] = 0

    elif total in [True, 1, 'Y', 'y']:
        if s3d.objmask == None:
            logger.info('Extracting full cube (minus edges)')
            exmask = np.zeros(s3d.data.shape[1:3])
            exmask[20:-20, 20:-20] = 1
        else:
            logger.info('Using object mask')
            exmask = s3d.objmask

    spectra, errors, nspec = [], [], 0.
    t1 = time.time()
    for y in range(exmask.shape[0]):
        for x in range(exmask.shape[1]):
            if exmask[y, x] == 1:
                spectra.append(s3d.data[:, y, x])
                errors.append(s3d.erro[:, y, x])
                nspec += 1

    spectra = np.array(spectra)
    errors = np.array(errors)
    if s3d.verbose > 0:
        logger.info('Used %i spaxels' %(nspec))

    if method == 'sum':
        spec = np.nansum(spectra, axis=0)
        err = np.nansum(errors**2, axis=0)**0.5

    if method == 'median':
        spec = np.nanmedian(spectra, axis = 0)
        err = np.nanmedian(errors, axis = 0)

    if method in ['average', 'avg']:
        spec = np.nansum(spectra, axis=0) / nspec
        err = np.nansum(errors**2, axis=0)**0.5  / nspec
    if s3d.verbose > 0:
        logger.info('Extracting spectra took %.1f s' %(time.time()-t1))
    if pexmask == True:
        logger.info('Plotting extraction map')
        pdfout(s3d, exmask, name='exmask', cmap = 'gist_gray')
    return s3d.wave, spec, err

def subtractCont(s3d, plane, pix1, pix2, cpix1, cpix2, dx=10):
    cont1 = np.nanmedian(s3d.data[pix1-dx:pix1], axis=0)
    cont2 = np.nanmedian(s3d.data[pix2:pix2+dx], axis=0)
    cont = np.nanmean(np.array([cont1,cont2]), axis=0)
    return plane - cont * (pix2 - pix1)