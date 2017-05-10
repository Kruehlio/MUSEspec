# -*- coding: utf-8 -*-

""" 

Extraction from a data cube along specific dimensions
    getGalcen : Extracts center of galaxy
    extract1d : Extracts a single spectrum
    extract2d : Extracts a single plane
    extract3d : Extracts a subcube between two wavelengths
    cutCube : Extracts a subcube between pixel ranges
    
"""

import numpy as np
import logging
import time

from ..MUSEio.museio import pdfout, plotspec
from ..utils.fitter import onedgaussfit


logfmt = '%(levelname)s [%(asctime)s]: %(message)s'
datefmt= '%Y-%m-%d %H:%M:%S'
formatter = logging.Formatter(fmt=logfmt,datefmt=datefmt)
logger = logging.getLogger('__main__')
logging.root.setLevel(logging.DEBUG)
ch = logging.StreamHandler() #console handler
ch.setFormatter(formatter)
logger.handlers = []
logger.addHandler(ch)


RESTWL = {'oi': 6302.046,
      'oiia' : 3727.092, 'oii':3728.30, 'oiib' : 3729.875, 
      'oii7320': 7322.010, 'oii7331':7332.3,
      'oiiia' : 4960.30, 'oiiib': 5008.240, 'oiii':5008.240,
      'hd': 4102.9351, 'hg' : 4341.69, 'hb' : 4862.68, 'ha' : 6564.61,
      'siia':6718.29, 'siib': 6732.68, 'sii':6725.48,
      'siii6312': 6313.8, 'siii':9071.1, 'siii9531':9533.2,
      'sii4068': 4069.749,
      'neiii' : 3869.81,
      'nii5755': 5756.24, 'niia':6549.86, 'niib': 6585.27,
      'ariii7135':7137.8, 'ariii7751':7753.2, 'hei4922':4923.305,
      'heii5411':5413.030, 'hei5015':5017.0765, 'hei5876':5877.243, 'hei6680':6679.99}


CWLS = {'ha' : [RESTWL['ha'], RESTWL['niia'], RESTWL['niib']],
        'hb' : [RESTWL['hb'], RESTWL['hb'], RESTWL['hb']],
        'hg' : [RESTWL['hg'], RESTWL['hg'], RESTWL['hg']],
        'hd' : [RESTWL['hd'], RESTWL['hd'], RESTWL['hd']],
        'oiiia' : [RESTWL['oiiia'], RESTWL['oiiia'], RESTWL['oiiia']],                   
        'oiiib' : [RESTWL['oiiib'], RESTWL['oiiib'], RESTWL['oiiib']],                   
        'oi' : [RESTWL['oi'], RESTWL['oi'], RESTWL['oi']],                   
        'oiii' : [RESTWL['oiiib'], RESTWL['oiiib'], RESTWL['oiiib']],                   
        'nii' : [RESTWL['niib'], RESTWL['niia'], RESTWL['niib']],                   
        'niia' : [RESTWL['niia'], RESTWL['niia'], RESTWL['niib']],                   
        'niib' : [RESTWL['niib'], RESTWL['niia'], RESTWL['niib']],                   
        'nii5755' : [RESTWL['nii5755'], RESTWL['nii5755'], RESTWL['nii5755']],                   
        'siia' : [RESTWL['siia'],RESTWL['siia'], RESTWL['siib']],                   
        'siib' : [RESTWL['siib'], RESTWL['siia'], RESTWL['siib']],                   
        'sii' : [RESTWL['sii'], RESTWL['siia'], RESTWL['siib']],                   
        'siii' : [RESTWL['siii'], RESTWL['siii'], RESTWL['siii']],
        'hei4922' : [RESTWL['hei4922'], RESTWL['hei4922'], RESTWL['hei4922']],
        'oii7320' : [RESTWL['oii7320'], RESTWL['oii7320'], RESTWL['oii7320']],
        'oii7331' : [RESTWL['oii7331'], RESTWL['oii7331'], RESTWL['oii7331']],
        'ariii7135' : [RESTWL['ariii7135'], RESTWL['ariii7135'], RESTWL['ariii7135']],
        'siii6312' : [RESTWL['siii6312'], RESTWL['siii6312'], 
                  RESTWL['siii6312']]}


c = 2.99792458E5

def getGalcen(s3d, mask = True, line='ha', sC=1,
              xlim1=155, xlim2=165, ylim1=155, ylim2=165):
    """ Gets the central coordniates of the starlight-weighted light 
    distribution, usually a galaxy (or part thereof)
    Parameters
    ----------
        s3d : Spectrum3d class
            Initial spectrum class with the data and error
        mask : boolean
            default True, uses a mask derived from the EW of line to associate
            contributing pixels to the galaxy (exclude foreground stars)
        line : string
            Use this line for the mask creation
        xlim1, xlim2, ylim1, ylim2 : integer
            Search for brightest pixels within these region
    Returns
    -------
        xcen, yccn : floats
            xcen, ycen are the indices of the center of the weighted average
            of the light distribution
    """
    
    if mask == True:
        hamap, hamape = extract2d(line=line, sC=sC)
        mask = hamap/hamape > 3
    galcube, galcerr = extract2d(s3d, wl1=5000, wl2=9000)
    ysum, xsum = 2*[np.array([])]
    maxx, maxy = 0, 0
    for yindx in np.arange(galcube.shape[0]):
        ysumline = np.nansum(([a for a in galcube[yindx,:][mask[yindx,:]]]))
        ysum = np.append(ysum, ysumline)
        if ysumline > maxy  and (ylim1 <= yindx <= ylim2):
            bpixy, maxy = yindx, ysumline
    
    for xindx in np.arange(galcube.shape[1]):
        xsumline = np.nansum([a for a in galcube[:,xindx][mask[:,xindx]]])
        xsum = np.append(xsum, xsumline)
        if xsumline > maxx and (xlim1 <= xindx <= xlim2):
            bpixx, maxx = xindx, xsumline
            
    ycen = np.nansum(ysum[ysum > 0] * np.arange(s3d.leny)[ysum > 0])\
        /np.nansum(ysum[ysum > 0])
    xcen = np.nansum(xsum[xsum > 0] * np.arange(s3d.lenx)[xsum > 0])\
        /np.nansum(xsum[xsum > 0])
    logger.info('Galaxy center calculated at %.2f, %.2f' %(xcen, ycen))
    return xcen, ycen


def cutCube(s3d, x1=None, x2=None, y1=None, y2=None):
    """Extracts a subcube between pixel ranges

    Parameters
    ----------
        x1 : integer
            Lower x pixel (python Notation), uses minimum if not given
        x2 : float
            Upper x pixel (python Notation), uses maximum if not given
        y1 : integer
            Lower y pixel (python Notation), uses minimum if not given
        y2 : float
            Upper y pixel (python Notation), uses maximum if not given
    Returns
    -------
        subcube : Spectrum3d
            Spectrum3d instance of cut cube
    """

    if y2 == None: y2 = s3d.leny
    if y1 == None: y1 = 0
    if x2 == None: x2 = s3d.lenx
    if x1 == None: x1 = 0
    logger.info('Cutting cube between x = %i:%i, y = %i:%i' %(x1, x2, y1, y2))

    scut = s3d
    scut.data = s3d.data[:, y1:y2, x1:x2]
    scut.erro = s3d.erro[:, y1:y2, x1:x2]
    scut.head['NAXIS1'] = x2-x1
    scut.head['NAXIS2'] = y2-y1
    scut.lenx = x2-x1
    scut.leny = y2-y1
    scut.head['CRPIX1'] -= x1
    scut.head['CRPIX2'] -= y1
    return scut
    


def extract3d(s3d, wl1=None, wl2=None):
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

    pix1 = s3d.wltopix(wl1)
    pix2 = max(pix1+1, s3d.wltopix(wl2)+1)
    subcube = s3d.data[pix1:pix2]
    suberr = s3d.erro[pix1:pix2]
    subwl = s3d.wave[pix1:pix2]
    return subcube, suberr, subwl



def extract2d(s3d, wl1='', wl2='', c1='', c2 = '', 
              z=None, line=None, dv=120, dwl=None,
                 meth = 'sum', sC=0, v=0, pSpec=False):
    """Extracts a single plane, summed/averaged/medianed between two wave-
    lenghts, or for a single emission line. If the line is known to the code, 
    fits a Gaussian and determines mean redshift and velocity dispersion. Plots
    the line fit.

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
        currErro : np.array
            2-d plane combining the error of sumed data between wl1 and wl2
    """

    if z == None: z = s3d.z
    if z == None: z = 0

    if wl1 != '' and c1 == '':
        c1 = 5
    if wl2 != '' and c2 == '':
        c2 = 5        
        
    if line != None: 
        line = line.lower()
        if line in CWLS.keys():
            wl = CWLS[line][0] * (1+z)    
            cont1 = (CWLS[line][1] * (1+z)) - 2.3538 * dv/c*wl * 1.5
            cont2 = (CWLS[line][2] * (1+z)) + 2.3538 * dv/c*wl * 1.5
                
        elif line != None:
            logger.error('Line %s not known' %line)      
            raise SystemExit
        if cont2 > s3d.wave[-1]:
            logger.error('Line %s outside of WL range' %line)      
            raise SystemExit       
            
        if dwl == None:
            p1 = max(0, s3d.wltopix(wl - 2.3538*dv/c*wl))
            p2 = max(p1+1, s3d.wltopix(wl +  2.3538*dv/c*wl))
        else:
            p1 = max(0, s3d.wltopix(wl - dwl))
            p2 = max(p1+1, s3d.wltopix(wl +  dwl))
            
        if line == 'sii':
            p1 = max(0, s3d.wltopix(wl - 600./c*wl))
            p2 = max(p1+1, s3d.wltopix(wl +  600./c*wl))            

        cpix1 = s3d.wltopix(cont1)
        cpix2 = s3d.wltopix(cont2)
        
    elif wl1 != '' and wl2 != '' and c1 != '' and c2 != '':
        p1 = s3d.wltopix(wl1)
        p2 = s3d.wltopix(wl2)
        cpix1 = s3d.wltopix(wl1-c1)
        cpix2 = s3d.wltopix(wl2+c2)
        cont1, cont2 = wl1+c1, wl2+c2

    else:
        logger.error("Do not know what to extract")
        raise SystemError


    if meth in ['sum']:
        allPlane = np.nansum(s3d.data[p1:p2], axis = 0)
        allError = np.nansum(s3d.erro[p1:p2]**2, axis = 0)**0.5

    elif meth == 'median':
        allPlane = np.nanmedian(s3d.data[p1:p2], axis = 0)
        allError = np.nansum(s3d.erro[p1:p2]**2, axis = 0)**0.5 / (p2-p1)**0.5

    if sC == 1:
        if s3d.verbose > 0 or v > 0:
            logger.info( 'Subtracting continuum using lambda < %.1f and lambda > %.1f' \
                %(cont1, cont2))
        currPlane = subtractCont(s3d, allPlane, p1, p2, cpix1, cpix2)
        currError = contErro(s3d, allError, p1, p2, cpix1, cpix2)
    else:
        currPlane = allPlane
        currError = allError
    
    if meth in ['sum']:
        currPlane = currPlane * s3d.wlinc
        currError = currError * s3d.wlinc

    if line != None:
        y = np.nansum(np.nansum(s3d.data[p1:p2, 20:-20, 20:-20], 
                                   axis=1), axis=1)
        x = s3d.wave[p1:p2]
        gp = onedgaussfit(x, y/1E3,
          params = [np.median(y[0:2]/1E3), np.nanmax(y)/1E3, np.median(x), 2])
        gw = gp[0][3] / gp[0][2] * c
        
        if s3d.verbose > 0:
            logger.info("Width of line (Gaussian fit): %.0f km/s" %gw)
            if line in CWLS.keys():
                logger.info("Mean redshift: %.5f"  %(gp[0][2]/CWLS[line][0] - 1))
                
        if gw > dv:
            logger.warning("Width of line bigger than extraction window")
    
        if line in CWLS.keys() or pSpec != False:
            plotspec(s3d, x, y, name='%s_plane' %line, 
                     lines=[[gp[-1], gp[1]], [x, x*0 + gp[0][0]]])
    
    return currPlane, currError
    
    

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

    if ra!=None and dec!=None:
        try:
            posx, posy = s3d.skytopix(ra, dec)
        except TypeError:
            posx, posy = s3d.sexatopix(ra, dec)
    elif ell!=None:
        posx, posy, a, b, theta = ell
    elif total==False:
        posx, posy = x, y

    if radius==None and size==None and total==False and ell==None:
        if verbose==1:
            logger.info('Extracting pixel %i, %i' %(posx, posy))
        spec = np.array(s3d.data[:,posy, posx])
        err  = np.array(s3d.erro[:,posy, posx])
        return s3d.wave, spec, err

    if total==False and radius!=None:
        if verbose==1:
            logger.info('Creating extraction mask with radius %.1f arcsec' %radius)
        radpix = radius / s3d.pixsky
        x, y = np.indices(s3d.data.shape[0:2])
        exmask = ((x - posy)**2  +  (y - posx)**2)**0.5
        exmask[exmask <= radpix+1E-8] = -1
        exmask[exmask > radpix] = 0
        exmask = np.abs(exmask)

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
        rerr = np.nansum(np.nansum(err**2, axis = 1), axis=1)**0.5

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
    if verbose > 0:
        logger.info('Used %i spaxels' %(nspec))

    if method == 'sum':
        spec = np.nansum(spectra, axis=0)
        err = np.nansum(errors**2, axis=0)**0.5

    if method == 'median':
        spec = np.nanmedian(spectra, axis = 0)
        err = np.nansum(errors**2, axis=0)**0.5  / nspec

    if method in ['average', 'avg']:
        spec = np.nansum(spectra, axis=0) / nspec
        err = np.nansum(errors**2, axis=0)**0.5  / nspec
    
    # Add systematic error of 3%
#    err = (err**2 + (0.03*spec)**2)**0.5
    
    if s3d.verbose > 0:
        logger.info('Extracting spectra took %.1f s' %(time.time()-t1))

    if pexmask == True:
        logger.info('Plotting extraction map')
        pdfout(s3d, exmask, name='exmask', 
               cmap = 'gist_gray')
    
    return s3d.wave, spec, err


def contErro(s3d, erro, p1, p2, cpix1, cpix2, dx=10):
    cerr1 = np.nanmedian(s3d.erro[p1-dx:p1], axis = 0) / (dx)**0.5   
    cerr2 = np.nanmedian(s3d.erro[p2:p2+dx], axis = 0) / (dx)**0.5   
    cerro = (cerr1**2 + cerr2**2)**0.5 / np.sqrt(2.)
    return (erro**2 + cerro**2)**0.5

def subtractCont(s3d, plane, p1, p2, cpix1, cpix2, dx=10):
    cont1 = np.nanmedian(s3d.data[p1-dx:p1], axis=0)
    cont2 = np.nanmedian(s3d.data[p2:p2+dx], axis=0)
    cont = np.nanmean(np.array([cont1,cont2]), axis=0)
    return plane - cont * (p2 - p1)
    
def extractCont(s3d, line, dv=120, dx=10):
    z = s3d.z
    if line != None: 
        line = line.lower()
        if line in CWLS.keys():
            wl = CWLS[line][0] * (1+z)        
            cont1 = (CWLS[line][1] * (1+z)) - 2.3538 * dv/c*wl * 1.5
            cont2 = (CWLS[line][2] * (1+z)) + 2.3538 * dv/c*wl * 1.5
        elif line != None:
            logger.error('Line %s not known' %line)      
            raise SystemExit
    
        p1 = max(0, s3d.wltopix(wl - 2.3538*dv/c*wl))
        p2 = max(p1+1, s3d.wltopix(wl +  2.3538*dv/c*wl))
        
        if line == 'sii':
            p1 = max(0, s3d.wltopix(wl - 600./c*wl))
            p2 = max(p1+1, s3d.wltopix(wl +  600./c*wl))            
    
    cont1 = np.nanmedian(s3d.data[p1-dx:p1], axis=0)
    cont2 = np.nanmedian(s3d.data[p2:p2+dx], axis=0)
    cont = np.nanmean(np.array([cont1,cont2]), axis=0)
    return cont  