# -*- coding: utf-8 -*-

""" 
Make maps of physical parameters based on a spectrum3d object
"""

import scipy as sp
import numpy as np
import logging
import matplotlib.pyplot as plt
import time
import scipy.constants as spc
from joblib import Parallel, delayed
from .fitter import onedgaussfit
import scipy.ndimage.filters


logfmt = '%(levelname)s [%(asctime)s]: %(message)s'
datefmt= '%Y-%m-%d %H:%M:%S'
formatter = logging.Formatter(fmt=logfmt,datefmt=datefmt)
logger = logging.getLogger('__main__')
logging.root.setLevel(logging.DEBUG)
ch = logging.StreamHandler() #console handler
ch.setFormatter(formatter)
logger.handlers = []
logger.addHandler(ch)

c = 2.99792458E5

RESTWL = {'oiia' : 3727.092, 'oii':3728.30, 'oiib' : 3729.875, 'hd': 4102.9351,
          'hg' : 4341.69, 'hb' : 4862.68, 'niia':6549.86,
          'oiiia' : 4960.30, 'oiiib': 5008.240, 'oiii': 4990., 'ha' : 6564.61,
          'nii': 6585.27, 'siia':6718.29, 'siib':6732.68,
          'neiii' : 3869.81}


def getRGB(planes, minval=None, maxval=None, scale='lin'):
    """ Creates and rgb image from three input planes (bgr) """

    if len(planes) != 3:
        logger.error('There must be three input planes')
        raise SystemExit
    if (planes[0].shape != planes[1].shape) or (planes[1].shape != planes[2].shape):
        logger.error('Planes must be equal shape')
        raise SystemExit

    img = np.zeros((planes[0].shape[0], planes[0].shape[1], 3), dtype=np.float32)
    for i in range(3):
        # Calculate dynamic range
        if minval == None and maxval == None:
            planemed = scipy.ndimage.filters.median_filter(planes[i], 30)
            dyrange = np.nanmax(planemed)
            minsub = 0
        else:
            dyrange = maxval[i] - minval[i]
            minsub = minval[i]
        # Normalize individual planes
        if scale == 'sqrt':
            wp = ((planes[i]-minsub)/dyrange)**0.5
        if scale == 'log':
            wp = np.log10((planes[i]-minsub)/dyrange)
        if scale == 'lin':
            wp = (planes[i]-minsub)/dyrange
        wp[wp < 0] = 0
        wp[wp > 1] = 1
        img[:,:,i] = wp
    return img


def getDens(s3d):
    """ Derive electron density map, using the [SII] doublet and based on 
    the model of O'Dell et al. 2013 using Osterbrock & Ferland 2006
    Returns
    -------
        ne : np.array
            Contains the electorn map density, based on [SII] flux ratio
    """

    siia = s3d.extractPlane(line='SIIa', sC=1, meth='sum')
    siib = s3d.extractPlane(line='SIIb', sC=1, meth='sum')
    nemap = 10**(4.705 - 1.9875*siia/siib)
    return nemap
    
    
    
def getSFR(s3d):
    """ Uses Kennicut 1998 formulation to convert Ha flux into SFR. Assumes
    a Chabrier 2003 IMF, and corrects for host intrinsic E_B-V if the map
    has previously been calculated. No Parameters. Requires the redshift to
    be set, to calculate the luminosity distance.

    Returns
    -------
        sfrmap : np.array
            Contains the star-formation rate map density, based on Halpha flux
            values (corrected for galaxy E_B-V if applicable). Units is
            M_sun per year per kpc**2. Note the per kpc**2.
    """
    logger.info( 'Calculating SFR map')
    haflux = s3d.extractPlane(line='Ha', sC=1, meth='sum')
    halum = 4 * np.pi * s3d.LDMP**2 * haflux * 1E-20
    if s3d.ebvmap != None:
        logger.info( 'Correcting SFR for EBV')
        ebvcorr = s3d.ebvCor('ha')
        halum *= sp.ndimage.filters.median_filter(ebvcorr, 4)
    sfrmap = halum * 4.8E-42 / s3d.pixsky**2 / s3d.AngD
    return sfrmap    
    
    
    
def getOH(s3d, meth='o3n2'):
    """ Uses strong line diagnostics to calculate an oxygen abundance map
    based on spaxels. Extracts fits the necessary line fluxes and then uses
    the method defined through meth to calculate 12+log(O/H)

    Parameters
    ----------
        meth : str
            default 'o3n2', which is the Pettini & Pagel 2004 O3N2 abundance
            other options are:
                 n2: Pettini & Pagel 2004 N2
                 M13: Marino et al. 2013 O3N2
                 M13N2: Marino et al. 2013 N2
                 s2: Dopita et al. 2016 S2
                 D02N2: Denicolo et. al. 2002 N2
    Returns
    -------
        ohmap : np.array
            Contains the values of 12 + log(O/H) for the given method
    """

    logger.info( 'Calculating oxygen abundance map')
    ha = s3d.extractPlane(line='Ha', sC=1, meth='sum')
    hb = s3d.extractPlane(line='Hb', sC=1, meth='sum')
    oiii = s3d.extractPlane(line='OIII', sC=1, meth='sum')
    nii = s3d.extractPlane(line='NII', sC=1, meth='sum')
    siia = s3d.extractPlane(line='SIIa', sC=1, meth='sum')
    siib = s3d.extractPlane(line='SIIb', sC=1, meth='sum')

    o3n2 = np.log10((oiii/hb)/(nii/ha))
    n2 = np.log10(nii/ha)
    s2 = np.log10(nii/(siia+siib)) + 0.264 * np.log10(nii/ha)

    if meth in ['o3n2', 'O3N2', 'PP04']:
        ohmap = 8.73 - 0.32 * o3n2
    if meth in ['n2', 'NII']:
        ohmap = 9.37 + 2.03*n2 + 1.26*n2**2 + 0.32*n2**3
    if meth in ['M13']:
        ohmap = 8.533 - 0.214 * o3n2
    if meth in ['M13N2']:
        ohmap = 8.743 + 0.462*n2
    if meth in ['D02N2']:
        ohmap = 9.12 + 0.73*n2
    if meth in ['s2', 'S2', 'D16']:
        ohmap = 8.77 + s2 + 0.45 * (s2 + 0.3)**5
    return ohmap    
   

def getIon(s3d):
    """ Uses the ratio between a collisionally excited line ([OIII]5007)
    and the recombination line Hbeta as a tracer of ionization/excitation

    Returns
    -------
    ionmap : np.array
        Contains the values of [OIII]/Hbeta
    """

    logger.info( 'Calculating [OIII]/Hbeta map')
    hbflux = s3d.extractPlane(line='Hb', sC=1, meth='sum')
    oiiiflux = s3d.extractPlane(line='OIII', sC=1, meth='sum')
    ionmap = oiiiflux/hbflux
    return ionmap
   
   
   
def getEW(s3d, line, dv=100):
    """ Calculates the equivalent width (rest-frame) for a given line. Calls
    getCont, and extractPlane to derive line fluxes and continua

    Parameters
    ----------
        line : str
            Emission line name (Ha, Hb, OIII, NII, SIIa, SIIb)
        dv : float
            Velocity width in km/s around which to sum the line flux
            default 100 kms (corresponds to an interval of +/- 200 km/s
            to be extracted)

    Returns
    -------
        ewmap : np.array
            Equivalent width in AA for the given line
    """

    logger.info( 'Calculating map with equivalence width of line %s' %line)
    # Get line fluxes
    flux = s3d.extractPlane(line=line, sC=1, meth='sum')
    # Set continuum range, make sure no other emission line lies within
    if line in ['Ha', 'ha', 'Halpha']:
        contmin = RESTWL['niia'] * (1+s3d.z) - 2*dv/c*RESTWL['niia']
        contmax = RESTWL['nii'] * (1+s3d.z) + 2*dv/c*RESTWL['nii']
    elif line in ['Hbeta', 'Hb', 'hb']:
        contmin = RESTWL['hb'] * (1+s3d.z) - 2*dv/c*RESTWL['hb']
        contmax = RESTWL['hb'] * (1+s3d.z) + 2*dv/c*RESTWL['hb']
    elif line in ['OIII', 'oiii']:
        contmin = RESTWL['oiiib'] * (1+s3d.z) - 2*dv/c*RESTWL['oiiib']
        contmax = RESTWL['oiiib'] * (1+s3d.z) + 2*dv/c*RESTWL['oiiib']
    elif line in ['NII', 'nii']:
        contmin = RESTWL['niia'] * (1+s3d.z) - 2*dv/c*RESTWL['niia']
        contmax = RESTWL['nii'] * (1+s3d.z) + 2*dv/c*RESTWL['nii']
    elif line in ['SIIa', 'siia']:
        contmin = RESTWL['siia'] * (1+s3d.z) - 2*dv/c*RESTWL['siia']
        contmax = RESTWL['siib'] * (1+s3d.z) + 2*dv/c*RESTWL['siib']
    elif line in ['SIIb', 'siib']:
        contmin = RESTWL['siia'] * (1+s3d.z) - 2*dv/c*RESTWL['siia']
        contmax = RESTWL['siib'] * (1+s3d.z) + 2*dv/c*RESTWL['siib']
    cont = s3d.getCont(s3d.wltopix(contmin), s3d.wltopix(contmax))
    # Calculate emission line rest-frame equivalent width
    ewmap = flux/cont/(1+s3d.z)
    return ewmap   
    
    
   
def getEBV(self):
    """ Uses the Balmer decrement (Halpha/Hbeta) to calculate the relative
    color excess E_B-V using the intrinsic ratio of Osterbrook at 10^4 K of
    Halpha/Hbeta = 2.87. First extracts Halpha and Hbeta maps to derive
    the ebvmap.
    
    Returns
    -------
        ebvmap : np.array
            2-d map of the color excess E_B-V
    """
    
    logger.info( 'Calculating E_B-V map')
    Cha, Chb, Chg, Chd = 1, 0.348, 0.162, 0.089
    kha, khb, khg, khd = 2.446, 3.560, 4.019, 4.253
    haflux = self.extractPlane(line='Ha', sC = 1, meth = 'sum')
    hbflux = self.extractPlane(line='Hb', sC = 1, meth = 'sum')
    ebvmap = np.log10((Cha/Chb)/(haflux/hbflux)) / (0.4*(kha-khb))
    #        ebvmap2 = 1.98 * (np.log10(haflux/hbflux) - np.log10(2.85))
    ebvmap[ebvmap < 0] = 1E-6
    #        ebvmap[np.isnan(ebvmap)] = 1E-6
    self.ebvmap = ebvmap
    return ebvmap
 
   
 
def getBPT(s3d, snf=5, snb=5):
    """ Calculates the diagnostic line ratios of the Baldwin-Philips-Terlevich
    diagram ([NII]/Halpha) and [OIII]/Hbeta. Applies a signal-to-noise cut
    for the individual line, and plots the resulting values in an inten-
    sity map. Return nothing, but produces a pdf plot.

    Parameters
    ----------
        snf : float
            Signal to noise ratio cut of the faint lines, [NII] and Hbeta
            detault (5)
        snb : float
            Signal to noise ratio cut of the bright lines, [OIII] and Halpha
            detault (5)
    """

    logger.info( 'Deriving BPT diagram')
    ha = s3d.extractPlane(line='ha', sC=1)
    hae = s3d.extractPlane(line='ha', meth = 'error')
    oiii = s3d.extractPlane(line='oiiib', sC=1)
    oiiie = s3d.extractPlane(line='oiiib', meth = 'error')
    nii = s3d.extractPlane(line='nii', sC=1)
    niie = s3d.extractPlane(line='nii', meth = 'error')
    hb = s3d.extractPlane(line='hb', sC=1)
    hbe = s3d.extractPlane(line='hb', meth = 'error')
    sn1, sn2, sn3, sn4 = nii/niie, ha/hae, oiii/oiiie, hb/hbe
    sel = (sn1 > snf) * (sn2 > snb) * (sn3 > snb) * (sn4 > snf)

    niiha = np.log10(nii[sel].flatten()/ha[sel].flatten())
    oiiihb = np.log10(oiii[sel].flatten()/hb[sel].flatten())

    # The rest is just for the plot
    bins = [120,120]
    xyrange = [[-1.5,0.5],[-1.2,1.2]]
    hh, locx, locy = sp.histogram2d(niiha, oiiihb, range=xyrange, bins=bins)
    thresh = 4
    hh[hh < thresh] = np.nan
    fig = plt.figure(facecolor='white', figsize = (6, 5))
    fig.subplots_adjust(hspace=-0.75, wspace=0.3)
    fig.subplots_adjust(bottom=0.12, top=0.84, left=0.16, right=0.98)
    ax1 = fig.add_subplot(1,1,1)
    ax1.imshow(np.flipud(hh.T), alpha = 0.7, aspect=0.7,
           extent=np.array(xyrange).flatten(), interpolation='none')
    ax1.plot(np.log10(np.nansum(nii[sel])/np.nansum(ha[sel])),
             np.log10(np.nansum(oiii[sel])/np.nansum(hb[sel])), 'o', ms = 10,
             color = 'black', mec = 'grey', mew=2)
    ax1.plot(np.log10(np.nansum(nii)/np.nansum(ha)),
             np.log10(np.nansum(oiii)/np.nansum(hb)), 'o', ms = 10,
             color = 'firebrick', mec = 'white', mew=2)

    kf3 = np.arange(-1.7, 1.2, 0.01)
    kf0 = np.arange(-1.7, 0.0, 0.01)

    x = -0.596*kf3**2 - 0.687 * kf3 -0.655
    kfz0 = 0.61/((kf0)-0.02-0.1833*0)+1.2+0.03*0

    # SDSS ridgeline
    ax1.plot(x, kf3,  '-', lw = 1.5, color = '0.0')
    # AGN/SF discrimination at z = 0
    ax1.plot(kf0, kfz0, '--', lw = 2.5, color = '0.2')
    ax1.set_xlim(-1.65, 0.3)
    ax1.set_ylim(-1.0, 1.0)

    ax1.set_xlabel(r'$\log({[\mathrm{NII}]\lambda 6584/\mathrm{H}\alpha})$',
               {'color' : 'black', 'fontsize' : 15})
    ax1.set_ylabel(r'$\log({[\mathrm{OIII}]\lambda 5007/\mathrm{H}\beta})$',
               {'color' : 'black', 'fontsize' : 15})

    plt.savefig('%s_%s_BPT.pdf' %(s3d.inst, s3d.target))
    plt.close(fig)



def gaussfit(x, y):
    gaussparams = onedgaussfit(x, y,
              params = [np.median(y[0:5]), np.nanmax(y), np.median(x), 2])
    return gaussparams[0][2], gaussparams[0][3],\
            gaussparams[2][2], gaussparams[2][3],\
            gaussparams[0][1]/gaussparams[2][1]


def getVel(s3d, line='ha', dv=250, R=2500):
    
    """Produces a velocity map. Fits the emission line profile of the given
    line with a Gaussian, to derive central positions and Gaussian widths.
    Should be parallelized, but doesnt work as the output cube is scrambled
    when using more then 1 thread. Would love to know why.
    
    Parameters
    ----------
        line : str
            Emission line to use for velocity map (default Halpha)
        dv : float
            Velocity width in km/s around which to fit is performed
            default 250 kms
        R : float
            Resolving power R in dimensionless units (default 2500)
    Returns
    -------
        velmap : np.array
            Velocity map in km/s difference to the central value
        sigmap : np.array
            Map of line broadening in km/s (not corrected for resulotion)
        R : float
            Resolution in km/s (sigma)
    """
    
    logger.info('Calculating velocity map - this might take a bit')
    if line in ['Halpha', 'Ha', 'ha']:
        wlline = RESTWL['ha'] * (1 + s3d.z)
        minwl = wlline - 2 * dv/c * wlline
        maxwl = wlline + 2 * dv/c * wlline
    fitcube, subwl = s3d.subCube(wl1=minwl, wl2=maxwl)
    meanmap, sigmamap = [], []
    meanmape, sigmamape = [], []
    snmap = []
    t1 = time.time()
    for y in range(s3d.data.shape[1]): #np.arange(100, 200, 1):
        result = Parallel(n_jobs = 1, max_nbytes='1G',)\
        (delayed(gaussfit)(subwl, fitcube[:,y,i]) for i in range(s3d.data.shape[2]))
        #np.arange(100, 200, 1))
        meanmap.append(np.array(result)[:,0])
        sigmamap.append(np.array(result)[:,1])
        meanmape.append(np.array(result)[:,2])
        sigmamape.append(np.array(result)[:,3])
        snmap.append(np.array(result)[:,4])
    
    snmap = np.array(snmap)
    meanmap = np.array(meanmap)
    sigmamap = np.array(sigmamap)
    meanmape = np.array(meanmape)
    sigmamape = np.array(sigmamape)
    meanmape[meanmape == 0] = np.max(meanmape)
    sigmamape[sigmamape == 0] = np.max(sigmamape)
    
    if s3d.objmask != None:
        logger.info('Limiting range to objectmask')
        wlmean = np.nanmedian(meanmap[s3d.objmask == 1])
    else:
        wlmean = np.nansum(meanmap / meanmape**2) / np.nansum(1./meanmape**2)
    
    velmap = (meanmap - wlmean) / wlmean * spc.c/1E3
    sigmamap = (sigmamap / meanmap) * spc.c/1E3
    logger.info('Velocity map took %.1f s' %(time.time() - t1))
    velmap[snmap < 2] = np.nan
    sigmamap[snmap < 2] = np.nan
    Rsig = spc.c/(1E3 * R * 2 * (2*np.log(2))**0.5)
    
    return np.array(velmap), np.array(sigmamap), Rsig
      
      

def getSeg(self, plane, thresh=15, median=5):
    """ HII segregation algorithm. Work in progress. """

    logger.info('HII region segregation with EW threshold %i A' %(thresh))
    plane = scipy.ndimage.filters.median_filter(plane, median)
    logger.info('Median filtering input plane')
    maxdist = int(.5/self.AngD/self.pixsky)
    minpix = int(round(max(10, (0.05/self.AngD/self.pixsky)**2 * np.pi)))
    logger.info('Maximum distance from brightest region in px: %i' %maxdist)
    logger.info('Minimum area of HII region in px: %i' %minpix)
    segmap, h2count = plane * 0, 20
    maxy, maxx = plane.shape
    h2inf = {}
    while True:
        # Highest count pixel in EW map
        h2indx = np.where(plane == np.nanmax(plane))
        h2indx = h2indx[0][0], h2indx[1][0]
        h2save = []
        # Highest count pixel below threshold we break segration loop
        if plane[h2indx] < thresh:
            break

        # Go radially around until maxdist
        for r in np.arange(maxdist):
           samereg = np.array([])
           for j in np.arange(-r, r+1, 1):
                for i in np.arange(-r, r+1, 1):
                    if r-1 < (j**2 + i**2)**0.5 <= r:
                        posy, posx = h2indx[0] + i, h2indx[1] + j

                        # Are we still within the cube ?
                        if posy  < maxy and posx < maxx:
                            # Check pixel value at positions above thesh
                            if plane[posy, posx] > thresh:
                                h2save.append([posy, posx])
                                samereg = np.append(samereg, 1)
                            # Below threshold, should we stop growing the region
                            elif plane[posy, posx] < -10:
                                samereg = np.append(samereg, -2)
                            else:
                                samereg = np.append(samereg, 0)
           # If there are more pixels below the threshold than above, we stop
           # growing the individual region
           # This is somewhat arbitrary
           if np.mean(samereg) < 0.6:
               break

        # Save info on the specific region in dict
        if len(h2save) > minpix:
            h2count += 1
            h2inf['%s' %h2count] = {}
            h2inf['%s' %h2count]['npix'] = len(h2save)
            h2inf['%s' %h2count]['posx'] = np.median(np.array(h2save)[:,0])+1
            h2inf['%s' %h2count]['posy'] = np.median(np.array(h2save)[:,1])+1
            for pix in h2save:
                segmap[pix[0], pix[1]] = h2count
                plane[pix[0], pix[1]] = np.nan
        else:
            for pix in h2save:
                plane[pix[0], pix[1]] = np.nan
    return segmap