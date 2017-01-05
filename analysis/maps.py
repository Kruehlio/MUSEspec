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

from ..utils.fitter import onedgaussfit
from .extract import RESTWL, extract2d
from .formulas import calcebv, calcohD16, calcDens
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


def _getProp(s3d, propmap, ra, dec, rad):
    """ Return properties at ra, dec with radius rad
    
    Parameters
    ----------    
        propmap : 2d array
            Map with the respective property values
        ra : float or sexagesimal
            Right ascension to return
        dec : float or sexagesimal
            Declination to return
        rad : radius in pixel over which the property is calculated
     Returns
    -------
        cenprop : float
            Value at pixel closest to ra, dec         
        avgprop : float
            Average value in the circle around ra, dec, with radius rad
        medprop : float
            Median value in the circle around ra, dec, with radius rad
        stdprop : float
            Std deviation in the circle around ra, dec, with radius rad
    """

    try:
        posx, posy = s3d.skytopix(ra, dec)
    except TypeError:
        posx, posy = s3d.sexatopix(ra, dec)        

    x, y = np.indices(s3d.data.shape[0:2])
    logger.info('Getting properties at %s %s within %i spaxel' \
            %(ra, dec, rad))
    logger.info('Central spaxel %i %i' %(posx+1, posy+1))                
    exmask = np.round(((x - posy)**2  +  (y - posx)**2)**0.5)
    exmask[exmask < rad] = 1
    exmask[exmask >= rad] = 0
    cenprop = propmap[posy, posx]
    
    logger.info('Total number of  spaxels %i' %(len(propmap[exmask==1])))                
   
    avgprop = np.nanmean(propmap[exmask==1])
    medprop = np.nanmedian(propmap[exmask==1])
    stdprop = np.nanstd(propmap[exmask==1])
    logger.info('CenVal, AvgVal, MedVal, StdDev: %.3f, %.3f, %.3f, %.3f' 
        %(cenprop, avgprop, medprop, stdprop))
    return cenprop, avgprop, medprop, stdprop
    

def getTemp(s3d, sC=0, meth='SIII', kappa=30, ne=100):

    """ Calculates electron temperatures absed on auroral to nebular line ratios.
    Following Nicholls et al. 2013. Uses [OIII](4363) to [OIII](5007),
    [NII](5755) to [NII](6583) or [SIII](6312) to [SIII](9069) fluxes.
    
    Parameters
    ----------
        meth : str
            Oxygen, Sulfur or Nitrogen temperatures. Options [SIII], [NII],
            and [OIII]
        sC : int
            default 0, whether to estimate and subtract the continuum from line 
        kappa : float
            default 30. Kappa value with which to modify the typically obtained
            Maxwell-Boltzmann value
        ne : float
            Assumed electron density
    Returns
    -------
        toiii : float
            Inferred or calculated electron temperature T[OIII]
        toii : float
            Inferred or calculated electron temperature T[OII]
        tsiii : float
            Inferred or calculated electron temperature T[SIII]
        tkin : float
            Inferred electron temperature with the kappa value
        sn : float
            Signal to noise ratio of the auroral line detection
    """
    
    t = 1E4
    ebvmap, snmap = getEBV(s3d, sC=sC)
    
    if meth in ['SIII', 'siii']:
        nl, aul = 'siii', 'siii6312'
        a, b, c, d = 10719, 0.09519, 1.03510, 6.5E-3
        a1, a2, a3 = 1.00075, 1.09519, 3.21668
        b1, b2, b3 = 13.3016, 24396.2, 57160.4   
    
    if meth in ['OIII', 'oiii']:
        nl, aul = 'oiiib', 'oiii4363'
        a, b, c, d = 13229, 0.79432, 0.98196, 3.8895E-04
        a1, a2, a3 = 1.00036, 1.27142, 3.55371
        b1, b2, b3 = 21.1751, 42693.5, 103086.
        
    if meth in ['NII', 'nii']:
        nl, aul = 'niib', 'nii5755'
        a, b, c, d  = 10873, 0.76348, 1.01350, 3.6E-3     
        a1, a2, a3 = 1.0008, 1.26281, 3.06569
        b1, b2, b3 = 19.432, 31701.9, 70903.4  

    nf, nfe = s3d.extractPlane(line=nl, sC=sC, meth='sum')
    auf, aufe = s3d.extractPlane(line=aul, sC=sC, meth='sum')
    nf *= s3d.ebvCor(nl)
    auf *= s3d.ebvCor(aul)
    aufe *= s3d.ebvCor(aul)
    
    for n in range(40):
        t = a * (-np.log10((auf/nf)/(1 + d*(100./t**0.5))) - b)**(-c) 

    a, b, c = -0.546, 2.645, -1.276
    
    if meth in ['SIII', 'siii']:
        c -= t/1E4
        toiii = ((-b + (b**2 - 4*a*c)**0.5) / (2*a))*1E4
        tsiii = t
        toii = (-0.744 + toiii/1E4*(2.338 - 0.610*toiii/1E4)) * 1E4

    if meth in ['OIII', 'oiii']:
        toiii = t
        toii = (-0.744 + toiii/1E4*(2.338 - 0.610*toiii/1E4)) * 1E4
        tsiii = (a*toiii/1E4**2 + b*toiii/1E4 + c) * 1E4
        
    a = (a1 + a2/kappa + a3/kappa**2)
    b = (b1 + b2/kappa + b3/kappa**2)
    tkin = a * t - b
    sn = auf/aufe
    
    return toiii, toii, tsiii, tkin, sn



def getOHT(s3d, toiii, toii, tsiii, meth = 'O', sC=0):

    """ Calculates abundances based on electron temperatures and line fluxes.
    Following Nicholls et al. 2013. Uses [OII](7320, 7331), [OIII](5007),
    Hbeta, [SII](6717, 6731) and [SIII](6312) fluxes.
    
    Parameters
    ----------
        toiii : float
            [OIII] electron temperature in 10^4 K
        toii : float
            [OII] electron temperature in 10^4 K
        tsiii : float
            [SIII] electron temperature in 10^4 K 
        meth : str
            Oxygen or Sulfur abundances. Options O or S
        sC : int
            default 0, whether to estimate and subtract the continuum from line            
    Returns
    -------
        logoh : float
            total abundance of element  (summed over ionization)         
        logoih : float
            abundance of single ionized element   
        logoiih : float
            abundance of double ionized element   
    """
    
    x = 10**(-4)*100*toiii**(-0.5)
    hb, hbe = s3d.extractPlane(line='Hb', sC=sC, meth='sum')
    hb *= s3d.ebvCor('hb')

    if meth == 'O':
        oiii, oiiie = s3d.extractPlane(line='oiiib', sC=sC, meth='sum')
        oiia, oiiae = s3d.extractPlane(line='oii7320', sC=sC, meth='sum')
        oiib, oiibe = s3d.extractPlane(line='oii7331', sC=sC, meth='sum')
        oiia *= s3d.ebvCor('oii7320')
        oiib *= s3d.ebvCor('oii7331')
        oiii *= s3d.ebvCor('oiiib')
        
        logoih = np.log10((oiia+oiib)/hb) + 6.901 + 2.487 / toii\
            - 0.483 * np.log10(toii) - 0.013*toii + np.log10(1 - 3.48*x)       
        logoiih = np.log10(1.33*oiii/hb) + 6.200 + 1.251 / toiii \
            - 0.55 * np.log10(toiii) - 0.014 * toiii 
        logoh = np.log10((10**(logoih-12)+10**(logoiih-12)))+12
        return logoh, logoih, logoiih

    elif meth == 'S':
        siia, siiae = s3d.extractPlane(line='siia', sC=sC, meth='sum')
        siib, siibe = s3d.extractPlane(line='siib', sC=sC, meth='sum')
        siiib, siiibe = s3d.extractPlane(line='siii6312', sC=sC, meth='sum')
        siia *= s3d.ebvCor('siia')
        siib *= s3d.ebvCor('siib')
        siiib *= s3d.ebvCor('siii6312')
        
        logsih = np.log10((siia+siib)/hb) + 5.439 + 0.929 / toii\
            - 0.28 * np.log10(toii) - 0.018*toii + np.log10(1 + 1.39*x)       
        logsiih = np.log10(siiib/hb) + 6.690 + 1.678 / tsiii \
            - 0.47 * np.log10(tsiii) - 0.010 * tsiii 
        logsh = np.log10((10**(logsiih-12)+10**(logsih-12)))+12
        return logsh, logsih, logsiih


def getRGB(planes, minval=None, maxval=None, scale='lin'):
    
    """ Creates an RGB image from three input planes (bgr) with scaling
    
    Parameters
    ----------
        planes : list of three np.array
            List of exactly three input planes of equal size, which will be 
            mapped to the RGB channels of the output image. Order is bgr.
        minval : list of three floats
            Minimum cube value for scaling, must also be in a list length 3
        maxval : list of three floats
            Maximum cube value for scaling, must also be in a list length 3   
        scale : str
            Options lin, sqrt, log. Scaling of the output images, options 
            correspond to linear, square root and logarithmic scaling
    Returns
    -------
        img : np.array
            Contains the scaled image to be plotted using matplotlib imshow         
    """

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


def getDens(s3d, sC=1):
    
    """ Derive electron density map, using the [SII] doublet and based on 
    the model of O'Dell et al. 2013 using Osterbrock & Ferland 2006

    Parameters
    ----------
        sC : int
            default 1, whether to subtract the continuum from line
    Returns
    -------
        nemap : np.array
            Contains the log of the electron map density, based on [SII] flux 
            ratio
        nemape : np.array
            Contains the error in the log of the electron map density
    """

    siia, siiae = s3d.extractPlane(line='siia', sC=sC, meth='sum')
    siib, siibe = s3d.extractPlane(line='siib', sC=sC, meth='sum')
    nemap = calcDens(siia,siib)
    snmap = (1./(1./(siia/siiae)**2 + 1./(siib/siibe)**2))**0.5
    return nemap, snmap


    
def getSFR(s3d, sC=1, EBV=1):
    
    """ Uses Kennicut 1998 formulation to convert Ha flux into SFR. Assumes
    a Chabrier 2003 IMF, and corrects for host intrinsic E_B-V if the map
    has previously been calculated. No Parameters. Requires the redshift to
    be set, to calculate the luminosity distance.

    Parameters
    ----------
        sC : int
            default 1, whether to subtract the continuum from line
        EBV : int
            default 1, whether to correct the SFR for reddening
    Returns
    -------
        sfrmap : np.array
            Contains the star-formation rate map density, based on Halpha flux
            values (corrected for galaxy E_B-V if applicable). Units is
            M_sun per year per kpc**2. Note the per kpc**2.
        sfrmape : np.array
            Contains the error star-formation rate map density.
    """
    
    logger.info( 'Calculating SFR map')
    haflux, haerr = s3d.extractPlane(line='ha', sC=sC, meth='sum')
    halum = 4 * np.pi * s3d.LDMP**2 * haflux * 1E-20
    
    if s3d.ebvmap != None and EBV==1:
        logger.info( 'Correcting SFR for EBV')
        ebvcorr = s3d.ebvCor('ha')
        halum *= sp.ndimage.filters.median_filter(ebvcorr, 4)
    sfrmap = halum * 4.8E-42 / s3d.pixsky**2 / s3d.AngD
    snmap = haflux/haerr
    return sfrmap, snmap
    
    
    
def getOH(s3d, meth='o3n2', sC=1, ra=None, dec=None, rad=None):
    
    """ Uses strong line diagnostics to calculate an oxygen abundance map
    based on spaxels. Extracts fits the necessary line fluxes and then uses
    the method defined through meth to calculate 12+log(O/H)

    Parameters
    ----------
        meth : str
            default 'o3n2', which is the Pettini & Pagel 2004 O3N2 abundance
            other options are:
                 N2: Pettini & Pagel 2004 N2
                 M13: Marino et al. 2013 O3N2
                 M13N2: Marino et al. 2013 N2
                 S2: Dopita et al. 2016 S2
                 D02N2: Denicolo et. al. 2002 N2
                 Ar3O3: [ArIII]/[OIII] Stasinska (2006)
                 S3O3: [SIII]/[OIII] Stasinska (2006)
    Returns
    -------
        ohmap : np.array
            Contains the values of 12 + log(O/H) for the given method
        snmap : np.array
            Contains the values of S/N for 12 + log(O/H) for the given method
    """

    logger.info( 'Calculating oxygen abundance map %s method' %meth)
    ha, hae = s3d.extractPlane(line='ha', sC=sC, meth='sum')
    nii, niie = s3d.extractPlane(line='nii', sC=sC, meth='sum')
    snnii, snha = nii/niie, ha/hae
    n2 = np.log10(nii/ha)
    tmp = s3d.getEBV(sC=sC)
    
    if meth in ['o3n2', 'O3N2', 'PP04', 'OIIINII', 'PP04O3N2']:
        hb, hbe = s3d.extractPlane(line='hb', sC=sC, meth='sum')
        oiii, oiiie = s3d.extractPlane(line='oiiib', sC=sC, meth='sum')
        snoiii, snhb = oiii/oiiie, hb/hbe
        
        o3n2 = np.log10((oiii/hb)/(nii/ha))
        ohmap = 8.73 - 0.32 * o3n2
        snmap = (1./(1./snhb**2 + 1./snha**2 + \
                            1./snnii**2 + 1./snoiii**2))**0.5

    elif meth in ['Ar3O3']:
        oiii, oiiie = s3d.extractPlane(line='oiiib', sC=sC, meth='sum')
        ariii, ariiie = s3d.extractPlane(line='ariii7135', sC=sC, meth='sum')
        ar3o3 = np.log10(ariii/oiii *\
            s3d.ebvCor('ariii7135') / s3d.ebvCor('oiiib'))
        snmap = ariii/ariiie
        ohmap = 8.91 + 0.34*ar3o3 + 0.27*ar3o3**2 + 0.20*ar3o3**3

    elif meth in ['S3O3']:
        oiii, oiiie = s3d.extractPlane(line='oiiib', sC=sC, meth='sum')
        siii, siiie = s3d.extractPlane(line='siii', sC=sC, meth='sum')
        s3o3 = np.log10(siii/oiii *\
            s3d.ebvCor('siii') / s3d.ebvCor('oiiib'))
        snmap = siii/siiie
        ohmap = 8.70 + 0.28*s3o3 + 0.03*s3o3**2 + 0.10*s3o3**3

    elif meth in ['s2', 'S2', 'D16']:
        siia, siiae = s3d.extractPlane(line='siia', sC=sC, meth='sum')
        siib, siibe = s3d.extractPlane(line='siib', sC=sC, meth='sum')
        siisn = (siia/siiae)**2 + (siib/siibe)**2

        ohmap = calcohD16(siia, siib, ha, nii)
        snmap = (1./(1./siisn**2 + 1./snnii**2 + (1./snha)**2))**0.5

    elif meth in ['n2', 'NII', 'N2', 'PP04N2']:
        ohmap = 9.37 + 2.03*n2 + 1.26*n2**2 + 0.32*n2**3
        snmap = (1./(1./snha**2 + 1./snnii**2))**0.5

    elif meth in ['M13']:
        ohmap = 8.533 - 0.214 * o3n2
        snmap = (1./(1./snhb**2 + 1./snha**2 + \
                            1./snnii**2 + 1./snoiii**2))**0.5            
    elif meth in ['M13N2']:
        ohmap = 8.743 + 0.462*n2
        snmap = (1./(1./snha**2 + 1./snnii**2))**0.5

    elif meth in ['D02N2']:
        ohmap = 9.12 + 0.73*n2
        snmap = (1./(1./snha**2 + 1./snnii**2))**0.5

    else:
        logger.error('Method %s not defined' %meth)
        raise SystemExit
        
    if ra != None and dec != None and rad != None:
        cenprop, avgprop, medprop, stdprop =\
            _getProp(s3d, ohmap, ra, dec, rad)
    
    return ohmap, snmap  
   

def getQ(s3d, sC=1, ra=None, dec=None, rad=None):
    # Fit to Kewley & Dopita 2002 [SIII]/[SII] vs. q models - looks weird
#    a0, a1, a2, a3 = 6.7075, 1.1318, -0.0145, 0.1674
#    qmap = 10**(a0 + a1*ls3s2 + a2*ls3s2**2 + a3*ls3s2**3)

    logger.info( 'Deriving q map')
    s3s2, s3s2sn = getIon(s3d, sC=sC, ra=ra, dec=dec, rad=rad)
    ls3s2 = np.log10(1/s3s2)
    # Morisset et al 2016.
    qmap = -2.62 - 1.22*ls3s2
    # Dors 2011
    qmap = -3.09 - 1.36*ls3s2
    if ra != None and dec != None and rad != None:
        cenprop, avgprop, medprop, stdprop =\
            _getProp(s3d, qmap, ra, dec, rad)
    return qmap, s3s2sn

def getIon(s3d, meth='S', sC=1, ra=None, dec=None, rad=None):
    """    Creates an ionization map, based on SIII/SII Kewley & Dopita 2002, using
    [SIII](9532) = 2.44 * [SIII](9069) from Mendoza & Zeipen 1982,
    Returns
    -------
    ionmap : np.array
        Contains the values of [SIII]/[SII] or [OIII]/Hbeta
    ionmape : np.array
        Contains the S/N of [SIII]/[SII] or [OIII]/Hbeta
    """
#matplotlib.rc('text', usetex=True)

    if meth == 'S':
        logger.info( 'Calculating [SIII]/[SII] map')
        siii, siiie = s3d.extractPlane(line='SIII', sC=sC, meth='sum')
        siia, siiae = s3d.extractPlane(line='siia', sC=sC, meth='sum')
        siib, siibe = s3d.extractPlane(line='siib', sC=sC, meth='sum')
        ionmap = 3.44*siii/(siia+siib)
        # Ionization parameter U from Dors et al. 2011
#        umap = 10**(-1.36 * np.log10(1./ionmap) -3.09 )
        siisn = (siia/siiae)**2 + (siib/siibe)**2
        siiisn = siii/siiie
        snmap = (1./(1./siisn**2 + 1./siiisn**2))**0.5
    else:
        hb, hbe = s3d.extractPlane(line='Hb', sC=sC, meth='sum')
        oiii, oiiie = s3d.extractPlane(line='OIII', sC=sC, meth='sum')
        ionmap = oiii/hb
        snoiii, snhb = oiii/oiiie, hb/hbe
        snmap = (1./(1./snhb**2 + 1./snoiii**2))**0.5
    
    if ra != None and dec != None and rad != None:
        cenprop, avgprop, medprop, stdprop =\
            _getProp(s3d, ionmap, ra, dec, rad)
            
    return ionmap, snmap
   

   
def getEW(s3d, line, dv=120, plotC=False, plotF=False, 
          ra=None, dec=None, rad=None):
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
        ewerr : np.array
            Equivalent width error in AA for the given line
    """

    logger.info( 'Calculating map with equivalence width of line %s' %line)
    # Get line fluxes
    f1, fe1 = s3d.extractPlane(line=line, sC=1, meth='sum')
    cont = s3d.extractCont(line=line)

    # Calculate emission line rest-frame equivalent width
    ewmap = f1/cont/(1.+s3d.z)
    snmap = f1/fe1

    if plotC == True:
        s3d.pdfout(cont, name = '%s_%s' %(line, 'cont'), 
                   vmin=-np.nanstd(cont), vmax=8*np.nanstd(cont))
   
    if plotF == True:
        s3d.pdfout(f1, name = '%s_%s' %(line, 'flux'), 
                   vmin=-0.1*np.nanstd(f1), vmax=5*np.nanstd(f1))
   
    if ra != None and dec != None and rad != None:
        cenprop, avgprop, medprop, stdprop =\
            _getProp(s3d, ewmap, ra, dec, rad)
    return ewmap, np.abs(snmap), cont
    
    
   
def getEBV(s3d, sC=1, ra=None, dec=None, rad=None):
    """ Uses the Balmer decrement (Halpha/Hbeta) to calculate the relative
    color excess E_B-V using the intrinsic ratio of Osterbrook at 10^4 K of
    Halpha/Hbeta = 2.87. First extracts Halpha and Hbeta maps to derive
    the ebvmap.
    
    Returns
    -------
        ebvmap : np.array
            2-d map of the color excess E_B-V
    """
    del s3d.ebvmap
    logger.info( 'Calculating E_B-V map')
    ha, hae = s3d.extractPlane(line='ha', sC=sC, meth='sum')
    hb, hbe = s3d.extractPlane(line='hb', sC=sC, meth='sum')
    ebvmap = calcebv(ha, hb)
#    ebvmap = 1.98 * (np.log10(ha/hb) - np.log10(2.85))
    
    ebvmap[ebvmap < 0] = 1E-6
    ebvmap[ebvmap == np.nan] = 1E-6
    ebvmap[np.isnan(ebvmap)] = 1E-6
    s3d.ebvmap = ebvmap
    snha, snhb = ha/hae, hb/hbe
    snmap = (1./(1./snhb**2 + 1./snha**2))**0.5
    
    if ra != None and dec != None and rad != None:
        cenprop, avgprop, medprop, stdprop =\
            _getProp(s3d, ebvmap, ra, dec, rad)

    return ebvmap, snmap
 
   
 
def getBPT(s3d, snf=5, snb=5, sC=0, xlim1=-1.65, xlim2=0.3, ylim1=-1, ylim2=1,
           ra=None, dec=None):
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
        sC  : integer
    """
    logger.info( 'Deriving BPT diagram')
    ha, hae = extract2d(s3d, line='ha', sC=sC)
    oiii, oiiie = extract2d(s3d, line='oiiib', sC=sC)
    nii, niie = extract2d(s3d, line='nii', sC=sC)
    hb, hbe = extract2d(s3d, line='hb', sC=sC)
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
    fig = plt.figure(facecolor='white', figsize = (7, 6))
    fig.subplots_adjust(hspace=-0.75, wspace=0.3)
    fig.subplots_adjust(bottom=0.12, top=0.84, left=0.16, right=0.98)
    ax1 = fig.add_subplot(1,1,1)

    plt.imshow(np.flipud(hh.T), alpha = 0.7, aspect=0.7,
           extent=np.array(xyrange).flatten(), interpolation='none')
    
    ax1.plot(np.log10(np.nansum(nii[sel])/np.nansum(ha[sel])),
             np.log10(np.nansum(oiii[sel])/np.nansum(hb[sel])), 'o', ms = 10,
             color = 'navy', mec = 'grey', mew=2,
             label=r'$\mathrm{HII\,region\,average}$')

    ax1.plot(np.log10(np.nansum(nii)/np.nansum(ha)),
             np.log10(np.nansum(oiii)/np.nansum(hb)), 'o', ms = 10,
             color = 'firebrick', mec = 'white', mew=2,
             label=r'$\mathrm{Galaxy\,average}$')

    if ra != None and dec != None:
        try:
            posx, posy = s3d.skytopix(ra, dec)
        except TypeError:
            posx, posy = s3d.sexatopix(ra, dec)  
        ax1.plot(np.log10(nii[posy, posx]/ha[posy, posx]),
             np.log10(oiii[posy, posx]/hb[posy, posx]), 's', ms = 10,
             color = 'black', mec = 'white', mew=2, label=r'$\mathrm{SN\,site}$')

    bar = plt.colorbar(shrink = 0.9, pad = 0.01)
    bar.set_label(r'$\mathrm{Number\,of\,spaxels}$', size = 17)

    kf3 = np.arange(-1.7, 1.2, 0.01)
    kf0 = np.arange(-1.7, 0.0, 0.01)

    x = -0.596*kf3**2 - 0.687 * kf3 -0.655
    kfz0 = 0.61/((kf0)-0.02-0.1833*0)+1.2+0.03*0

    # SDSS ridgeline
    ax1.plot(x, kf3,  '-', lw = 1.5, color = '0.0')
    # AGN/SF discrimination at z = 0
    ax1.plot(kf0, kfz0, '--', lw = 2.5, color = '0.2')
    ax1.set_xlim(xlim1, xlim2)
    ax1.set_ylim(ylim1, ylim2)
    ax1.set_xlabel(r'$\log({[\mathrm{NII}]\lambda 6584/\mathrm{H}\alpha})$',
               {'color' : 'black', 'fontsize' : 15})
    ax1.set_ylabel(r'$\log({[\mathrm{OIII}]\lambda 5007/\mathrm{H}\beta})$',
               {'color' : 'black', 'fontsize' : 15})

    legend = ax1.legend(frameon=True,  markerscale = 0.9,  numpoints=1,
                        handletextpad = -0.2,
                        loc = 1, prop={'size':14})
    rect = legend.get_frame()
    rect.set_facecolor("0.9")
    rect.set_linewidth(0.0)
    rect.set_alpha(0.5)   

    plt.figtext(0.3, 0.22, r'$\mathrm{Star-forming}$', size = 17,  color = 'black')
    plt.figtext(0.75, 0.5, r'$\mathrm{AGN}$', size = 17, color = 'black')

    plt.savefig('%s_%s_BPT.pdf' %(s3d.inst, s3d.target))
    plt.close(fig)



def _gaussfit(x, y):
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
            Map of line broadening in km/s (not corrected for resolution)
        snmap : 
            SN map of Gaussfit
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
        (delayed(_gaussfit)(subwl, fitcube[:,y,i]) for i in range(s3d.data.shape[2]))
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
#    velmap[snmap < 2] = np.nan
#    sigmamap[snmap < 2] = np.nan
    Rsig = spc.c/(1E3 * R * 2 * (2*np.log(2))**0.5)
    
    return np.array(velmap), np.array(sigmamap), snmap, Rsig
      
      

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