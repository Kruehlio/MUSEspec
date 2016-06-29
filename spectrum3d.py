#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Spectrum class for 3d-spectra. Particularly MUSE."""

import matplotlib
matplotlib.use('Agg')

import pyfits
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patheffects import withStroke
import warnings
warnings.filterwarnings("ignore")

import time
import scipy as sp
import scipy.constants as spc
import os
import multiprocessing
import scipy.ndimage.filters
import logging
import sys

from joblib import Parallel, delayed
from spec.astro import (LDMP, Avlaws, airtovac, ergJy, 
                       abflux, getebv)

from spec.functions import (blur_image, deg2sexa, sexa2deg, ccmred)
from spec.fitter import onedgaussfit

logfmt = '%(levelname)s [%(asctime)s]: %(message)s'
datefmt= '%Y-%m-%d %H:%M:%S'
formatter = logging.Formatter(fmt=logfmt,datefmt=datefmt)
logger = logging.getLogger('__main__')
logging.root.setLevel(logging.DEBUG)
ch = logging.StreamHandler() #console handler
ch.setFormatter(formatter)
logger.handlers = []
logger.addHandler(ch)

import signal
def signal_handler(signal, frame):
    sys.exit("CTRL+C detected, stopping execution")
signal.signal(signal.SIGINT, signal_handler)

c = 2.99792458E5

RESTWL = {'oiia' : 3727.092, 'oii':3728.30, 'oiib' : 3729.875, 'hd': 4102.9351,
          'hg' : 4341.69, 'hb' : 4862.68, 'niia':6549.86,
          'oiiia' : 4960.30, 'oiiib': 5008.240, 'oiii': 4990., 'ha' : 6564.61, 
          'nii': 6585.27, 'siia':6718.29, 'siib':6732.68,
          'neiii' : 3869.81}

def gaussfit(x, y):
    gaussparams = onedgaussfit(x, y,
              params = [np.median(y[0:5]), np.nanmax(y), np.median(x), 2])
    return gaussparams[0][2], gaussparams[0][3],\
            gaussparams[2][2], gaussparams[2][3],\
            gaussparams[2][1]/gaussparams[0][1]
            
            
class Spectrum3d:
    """ Fits cube class for data exploration, analysis, modelling.
   
   Arguments:
        inst: Instrument that produced the spectrum (optional, default=MUSE)
        filen: Filename of fits data cube
        target: Target of observation (for output file)
        
    Methods:
        setFiles: Sets fits files
        setRedshift: Given an redshift z, sets cosmological parameters
        ebvGal: Uses fits header keywords RA, DEC to get Galactic Forground EB-V
        ebvCor: Corrects a given line for the EB-V map
        checkPhot: Checks the flux calibration of the cube through synthetic phtometry
        subtractCont: Subtracts continuum of plane
        getCont: Measures continuum of plane
        getSFR: Calculates the SFR density map based on Halpha
        getOHsimp: Calculates oxygen abundance map based on strong line diagnostics
        getIon: Calculates [OIII]/Hbeta map as ionization/excitation proxy
        getEW: Calculates equivalent width maps of given line
        BPT: Spaxels in the Baldwich-Philips-Terlevich diagram
        getEBV: Calculates EB-V maps from Balmer decrement
        subCube: Extracts cube cut in wavelength
        extractPlane: Extracts a plane, summed in wavelength
        extrSpec: Extracts a spectrum at given position
        astro: Corrects fits file astrometry
        wltopix: Wavelength to pixel conversion
        pixtowl: Pixel to wavelength conversion
        skytopix: Sky coordinates conversion (degree) to pixel conversion
        pixtosky: Pixel to sky coordinates conversion (degree)
        sexatopix: Sky coordinates conversion (sexagesimal) to pixel conversion
        pixtosexa: Pixel to sky coordinates conversion (sexagesimal)
        velMap: Calculates velocity map - clumsy, takes ages
        createaxis: Helper function for plot
        plotxy: Simple x vs. y scatter plot
        plotspec: Simple flux vs. wavelength line plot
        hiidetect: HII region detection algorithm (work in progress)
        pdfout: Plots a 2d-map as pdf
        fpack: fpacks the output file
        rgb: Provided three planes, creates an RGP image
        scaleCube: Scale cube by polynomial of given degree (default 1)
        writeFiles: writes the output into a fits file
        fitsout: write a given plane into a fits file
        fitsin: read a given fits file into a plane
    """
    
    
    def __init__(self, filen=None, inst='MUSE', target=''):
        self.inst = inst
        self.z = None
        self.datfile = ''
        self.target = target
        self.mask = None
        self.skymask = None
        self.maskpix = None
        self.ebvmap = None
        self.objmask = None
        self.scale = []
        if filen != None:
            self.setFiles(filen)
        self.ncores =  multiprocessing.cpu_count()/2
        logger.info('Using %i cores for analysis' %self.ncores)



    def setFiles(self, filen, fluxmult=1, dAxis=3, mult=1):
        """ Uses pyfits to set the header, data and error as instance attributes.
        The fits file should have at least two extension, where the first containst
        the data, the second the variance. Returns nothing.
        
        Parameters
        ----------
        filen : str
            required, this is the filname of the fitsfile to be read in
        """

        # Get primary header
        self.headprim = pyfits.getheader(filen, 0)
        # Get header of data extension
        self.head = pyfits.getheader(filen, 1)
        self.headerro = pyfits.getheader(filen, 2)

        wlkey, wlstart = 'NAXIS%i'%dAxis, 'CRVAL%i'%dAxis
        wlinc, wlpixst = 'CD%i_%i'%(dAxis, dAxis), 'CRPIX%i'%dAxis
        pix = np.arange(self.head[wlkey]) + self.head[wlpixst]
        self.pix = pix
        # Create wave array from fits info
        self.wave = airtovac(self.head[wlstart] + (pix - 1) * self.head[wlinc])

        self.pixsky = (self.head['CD1_1']**2 + self.head['CD1_2']**2) ** 0.5 * 3600
        self.lenx = self.head['NAXIS1']
        self.leny = self.head['NAXIS2']
        self.wlinc =  self.head[wlinc]
        # Read in data
        self.data = pyfits.getdata(filen, 1)
        # Read in variance and turn into stdev
        self.erro = pyfits.getdata(filen, 2)**0.5
        if self.target == '':
            self.target = self.headprim['OBJECT']
        self.fluxunit = self.head['BUNIT']
        self.output = filen.split('.fits')[0]
        self.base, self.ext = filen.split('.fits')[0], '.fits'
        logger.info( 'Fits cube loaded %s' %(filen))
        logger.info( 'Wavelength range %.1f - %.1f (vacuum)' %(self.wave[0], self.wave[-1]))



    def setRedshift(self, z):
        """ Setting luminosity distance and angular seperation here, provided
        a given redshift z. Returns nothing.
        
        Parameters
        ----------
        z : float
            required, this is the redshift of the source
        """
        
        LD, angsep = LDMP(z, v=2)
        logger.info('Luminosity distance at z=%.4f: %.2f MPc' %(z, LD))
        logger.info('Luminosity distance at z=%.4f: %.2e cm' %(z, LD * 3.0857E24))
        self.z = z
        self.LDMP = LD * 3.0857E24
        self.AngD = angsep



    def ebvGal(self, ebv = '', rv=3.08):
        """ If user does not provide ebv, it uses the header information of the 
        pointing to obtain the Galactic
        foreground reddening from IRSA. These are by default the Schlafly and
        Finkbeiner values. Immediatly dereddens the data and error using rv 
        (default 3.08) in place. Returns nothing.
        
        Parameters
        ----------
        ebv : float
            default '', and queries the web given header RA and DEC
        rv : float
            default 3.08, total to selective reddening RV
        """
        
        if ebv == '':
            ra, dec = self.headprim['RA'], self.headprim['DEC']
            ebv, std, ref, av = getebv(ra, dec, rv)
        ebvcorr = ccmred(self.wave, ebv, rv)    
        logger.info('Dereddening data using MW E_B-V = %.3f mag' %ebv)
        self.data *= ebvcorr[:,np.newaxis, np.newaxis]
        self.erro *= ebvcorr[:,np.newaxis, np.newaxis]



    def writeFiles(self):
        if os.path.isfile(self.output):
            os.remove(self.output)
        hdu = pyfits.HDUList()
        hdu.append(pyfits.PrimaryHDU(header = self.headprim))
        hdu.append(pyfits.ImageHDU(data = self.data, header = self.head))
        hdu.append(pyfits.ImageHDU(data = self.erro, header = self.headerro))
        hdu.writeto(self.output)



    def ebvCor(self, line, rv=3.08, redlaw='mw'):
        """ Uses a the instance attribut ebvmap, the previously calculated map
        of host reddening to calulate a correction map for a given line.
        
        Parameters
        ----------
        line : str
            default '', for example ha for Halpha
        rv : float
            default 3.08, total to selective reddening RV
        redlaw : str
            default mw, assumed reddening law
            
        Returns
        -------
        ebvcorr : np.array
            The correction map to be applied to the linefluxes to correct 
            for the galaxy's dust absorption
        """

        if len(self.ebvmap) != None:
            WL = RESTWL[line.lower()]/10.
            ebvcorr = 1./np.exp(-1./1.086*self.ebvmap * rv * Avlaws(WL, redlaw))
            ebvcorr[np.isnan(ebvcorr)] = 1
            ebvcorr[ebvcorr < 1] = 1
            return ebvcorr
        else:
            logger.error( 'Need an EBV-map / create via getEBV !!!')
            raise SystemExit


    def checkPhot(self, mag, band='r', ra=None, dec=None, radius=7):
        """ Uses synthetic photometry at a given position in a given band at a
        given magnitude to check the flux calibration of the spectrum. 
        Returns nothing.
        
        Parameters
        ----------
        mag : float
            default '', required magnitude of comparison
        band : str
            default r, photometric filter. VRI are assumed to be in Vega, griz 
            in the AB system
        ra : float
            default None, Right Ascension of comparison star
        dec : float
            default None, Declination of comparison star. If ra and/or dec are
            None, uses the spectrum of the full cube
        radius : int
            Radius in pixel around ra/dec for specturm extraction
        """

        ABcorD = {'g':-0.062, 'V':0.00, 'r':0.178,
                 'R':0.21, 'i':0.410, 'I':0.45, 'z':0.543}
        wls = {'g': [3856.2, 5347.7], 'r': [5599.5, 6749.0], 'i': [7154.9, 8156.6], 
           'V': [4920.9, 5980.2], 'R': [5698.9, 7344.4], 
           'I': [7210., 8750.], 'F814': [6884.0, 9659.4], 'z': [8250.0, 9530.4]}
        if band in 'VRI':
            mag = mag + ABcorD[band]
        if ra != None and dec != None:  
          wl, spec, err = self.extrSpec(ra = ra, dec = dec, radius = radius)
        else:
          wl, spec, err = self.extrSpec(total=True)
          
        bandsel = (wl > wls[band][0]) * (wl < wls[band][1]) 
        avgflux = np.nanmedian(spec[bandsel])*1E-20
        avgwl = np.nanmedian(wl[bandsel])
        fluxspec = ergJy(avgflux, avgwl)
        fluxref = abflux(mag)
        logger.info('Scale factor from spectrum to photometry for band %s-band: %.3f' \
          %(band, fluxref/fluxspec))
        self.scale.append([avgwl, fluxref/fluxspec])


    def subtractCont(self, plane, pix1, pix2, cpix1, cpix2, dx=10):
        cont1 = np.nanmedian(self.data[pix1-dx:pix1], axis=0)
        cont2 = np.nanmedian(self.data[pix2:pix2+dx], axis=0)
        cont = np.nanmean(np.array([cont1,cont2]), axis=0)
        return plane - cont * (pix2 - pix1)


    def getCont(self, pix1, pix2, dx=15):
        cont1 = np.nanmedian(self.data[pix1-dx:pix1], axis=0)
        cont2 = np.nanmedian(self.data[pix2:pix2+dx], axis=0)
        return np.nanmean(np.array([cont1,cont2]), axis=0)


    def getSFR(self):
        haflux = self.extractPlane(line='Ha', sC=1, meth='sum')
        halum = 4 * np.pi * self.LDMP**2 * haflux * 1E-20
        if self.ebvmap != None:
            logger.info( 'Correcting SFR for EBV')
            ebvcorr = self.ebvCor('ha')
            halum *= sp.ndimage.filters.median_filter(ebvcorr, 4)
        sfrmap = halum * 4.8E-42 / self.pixsky**2 / self.AngD
        return sfrmap


    def getOHsimp(self, meth='o3n2'):
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
        """        
        ha = self.extractPlane(line='Ha', sC = 1, meth = 'sum')
        hb = self.extractPlane(line='Hb', sC = 1, meth = 'sum')
        oiii = self.extractPlane(line='OIII', sC = 1, meth = 'sum')
        nii = self.extractPlane(line='NII', sC = 1, meth = 'sum')
        siia = self.extractPlane(line='SIIa', sC = 1, meth = 'sum')
        siib = self.extractPlane(line='SIIb', sC = 1, meth = 'sum')
        
        o3n2 = np.log10((oiii/hb)/(nii/ha))
        n2 = np.log10(nii/ha)
        s2 = np.log10(nii/(siia+siib)) + 0.264*np.log10(nii/ha)
        
        if meth in ['o3n2', 'O3N2', 'PP04']:
            ohmap = 8.73 - 0.32 * o3n2
        if meth in ['n2', 'NII']:
            ohmap = 9.37 + 2.03*n2 + 1.26*n2**2 + 0.32*n2**3 
        if meth in ['M13']:
            ohmap = 8.533 - 0.214 * o3n2
        if meth in ['M13N2']:
            ohmap = 8.743 + 0.462*n2
        if meth in ['s2', 'S2', 'D16']:
            ohmap = 8.77 + s2 + 0.45 * (s2 + 0.3)**5
        return ohmap



    def getIon(self):
        hbflux = self.extractPlane(line='Hb', sC = 1, meth = 'sum')
        oiiiflux = self.extractPlane(line='OIII', sC = 1, meth = 'sum')
        ionmap = oiiiflux/hbflux
        return ionmap



    def getEW(self, line, dv=100):
        flux = self.extractPlane(line=line, sC = 1, meth = 'sum')
        if line in ['Ha', 'ha', 'Halpha']:
            contmin = RESTWL['niia'] * (1+self.z) - 2*dv/c*RESTWL['niia']
            contmax = RESTWL['nii'] * (1+self.z) + 2*dv/c*RESTWL['nii']
        elif line in ['Hbeta', 'Hb', 'hb']:
            contmin = RESTWL['hb'] * (1+self.z) - 2*dv/c*RESTWL['hb']
            contmax = RESTWL['hb'] * (1+self.z) + 2*dv/c*RESTWL['hb']
        elif line in ['OIII', 'oiii']:
            contmin = RESTWL['oiiib'] * (1+self.z) - 2*dv/c*RESTWL['oiiib']
            contmax = RESTWL['oiiib'] * (1+self.z) + 2*dv/c*RESTWL['oiiib']
        elif line in ['NII', 'nii']:
            contmin = RESTWL['niia'] * (1+self.z) - 2*dv/c*RESTWL['niia']
            contmax = RESTWL['nii'] * (1+self.z) + 2*dv/c*RESTWL['nii']
        elif line in ['SIIa', 'siia']:
            contmin = RESTWL['siia'] * (1+self.z) - 2*dv/c*RESTWL['siia']
            contmax = RESTWL['siib'] * (1+self.z) + 2*dv/c*RESTWL['siib']
        elif line in ['SIIb', 'siib']:
            contmin = RESTWL['siia'] * (1+self.z) - 2*dv/c*RESTWL['siia']
            contmax = RESTWL['siib'] * (1+self.z) + 2*dv/c*RESTWL['siib']
        cont = self.getCont(self.wltopix(contmin), self.wltopix(contmax))
        ewmap = flux/cont/(1+self.z)
        return ewmap



    def BPT(self, snf=5, snb=5):
        ha = self.extractPlane(line='ha', sC = 1)
        hae = self.extractPlane(line='ha', meth = 'error')
        oiii = self.extractPlane(line='oiiib', sC = 1)
        oiiie = self.extractPlane(line='oiiib', meth = 'error')
        nii = self.extractPlane(line='nii', sC = 1)
        niie = self.extractPlane(line='nii', meth = 'error')
        hb = self.extractPlane(line='hb', sC = 1)
        hbe = self.extractPlane(line='hb', meth = 'error')
        sn1, sn2, sn3, sn4 = nii/niie, ha/hae, oiii/oiiie, hb/hbe
        sel = (sn1 > snf) * (sn2 > snb) * (sn3 > snb) * (sn4 > snf)
        
        niiha = np.log10(nii[sel].flatten()/ha[sel].flatten())
        oiiihb = np.log10(oiii[sel].flatten()/hb[sel].flatten())
        bins = [120,120]
        xyrange = [[-1.5,0.5],[-1.2,1.2]] 
        hh, locx, locy = scipy.histogram2d(niiha, oiiihb, range=xyrange, bins=bins)
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

        ax1.plot(x, kf3,  '-', lw = 1.5, color = '0.0')                      
        ax1.plot(kf0, kfz0, '--', lw = 2.5, color = '0.2')                 
        ax1.set_xlim(-1.65, 0.3)
        ax1.set_ylim(-1.0, 1.0)

        ax1.set_xlabel(r'$\log({[\mathrm{NII}]\lambda 6584/\mathrm{H}\alpha})$', 
                   {'color' : 'black', 'fontsize' : 15})
        ax1.set_ylabel(r'$\log({[\mathrm{OIII}]\lambda 5007/\mathrm{H}\beta})$', 
                   {'color' : 'black', 'fontsize' : 15})  

        plt.savefig('%s_%s_BPT.pdf' %(self.inst, self.target))
        plt.close(fig)



    def getEBV(self):
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



    def subCube(self, wl1=None, wl2=None):
        pix1 = self.wltopix(wl1)
        pix2 = max(pix1+1, self.wltopix(wl2)+1)
        subcube = self.data[pix1:pix2]
        subwl = self.wave[pix1:pix2]
        return subcube, subwl



    def extractPlane(self, wl1='', wl2='', z=None, line=None, dv=100,
                     meth = 'sum', sC = 0):

        if z == None: z = self.z
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
        
        pix1 = self.wltopix(wl1)
        pix2 = max(pix1+1, self.wltopix(wl2))
        cpix1 = self.wltopix(cont1)
        cpix2 = self.wltopix(cont2)

        if meth in ['average', 'sum']:
            currPlane = np.nansum(self.data[pix1:pix2], axis = 0)
        elif meth == 'median':
            currPlane = np.nanmedian(self.data[pix1:pix2], axis = 0)
        elif meth == 'error':
            currPlane = np.nansum(self.erro[pix1:pix2]**2, axis = 0)**0.5

        if sC == 1:
            logger.info( 'Subtracting continuum using lambda < %.1f and lambda > %.1f' \
                %(cont1, cont2))
            currPlane = self.subtractCont(currPlane, pix1, pix2,
                                               cpix1, cpix2)

        if meth in ['sum', 'int']:
            currPlane = currPlane * self.wlinc
        return currPlane



    def extrSpec(self, ra=None, dec=None, x=None, y=None, radius=None,
                 method='sum', total=False, ell=None, extrmap=None):

        if ra != None and dec != None:
            try:
                posx, posy = self.skytopix(ra, dec)
            except TypeError:
                posx, posy = self.sexatopix(ra, dec)
        elif ell != None:
            posx, posy, a, b, theta = ell
        elif total == False:
            posx, posy = x + 1, y + 1

        if radius == None and total == False and ell==None:
            logger.info( 'Extracting pixel %i, %i' %(posx, posy))
            spec = np.array(self.data[:,posy-1,posx-1])
            err  = np.array(self.erro[:,posy-1,posx-1])
            return self.wave, spec, err

        if total == False and radius != None:
            logger.info( 'Creating extraction mask radius')
            radpix = radius / self.pixsky
            x, y = np.indices(self.data.shape[0:2])
            
            exmask = np.round(((x - posy)**2  +  (y - posx)**2)**0.5)
            exmask[exmask <= radpix] = 1
            exmask[exmask > radpix] = 0
        
        elif total == False and ell != None:
            # Ellipse is in pixel coordinates
            logger.info( 'Creating extraction ellipse')
            x, y = np.indices(self.data.shape[0:2])
            ell = ((x - posx) * np.cos(theta) + (y-posy) *np.sin(theta))**2 / a**2 \
                 +((x - posx) * np.sin(theta) - (y-posy) *np.cos(theta))**2 / b**2
            exmask = np.round(ell)
            exmask[exmask <= 1] = 1
            exmask[exmask > 1] = 0

        elif total in [True, 1, 'Y', 'y']:
            if self.objmask == None:
                logger.info('Extracting full cube (minus edges)')
                exmask = np.zeros(self.data.shape[1:3])
                exmask[20:-20, 20:-20] = 1
            else:
                logger.info('Using object mask')
                exmask = self.objmask

        spectra, errors, nspec = [], [], 0.
        t1 = time.time()
        for y in range(exmask.shape[0]):
            for x in range(exmask.shape[1]):
                if exmask[y, x] == 1:
                    spectra.append(self.data[:, y, x])
                    errors.append(self.erro[:, y, x])
                    nspec += 1

        spectra = np.array(spectra)
        errors = np.array(errors)
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


        logger.info('Extracting spectra took %.1f s' %(time.time()-t1))
        return self.wave, spec, err


    def astro(self, starras, stardecs, ras, decs):
        """Correct MUSE astrometry: Starra and stardec are lists of the original
        coordinates of a source in the MUSE cube with actual coordinates ra, dec"""
        dra, ddec = np.array([]), np.array([])
        for starra, stardec, ra, dec in zip(starras, stardecs, ras, decs):
            starra, stardec = sexa2deg(starra, stardec)
            ra, dec = sexa2deg(ra, dec)
            dra = np.append(dra, starra - ra)
            ddec = np.append(ddec, stardec - dec)
        dram, ddecm = np.average(dra), np.average(ddec)
        logger.info('Changing astrometry by %.1f" %.1f"' %(dram*3600, ddec*3600))
        logger.info('RMS astrometry %.1f" %.1f"' %(np.std(dra)*3600, np.std(ddec)*3600))
        self.head['CRVAL1'] -= dram
        self.head['CRVAL2'] -= ddecm



    def wltopix(self, wl):
        pix = ((wl - self.wave[0]) / self.wlinc) + 1
        return max(0, int(round(pix)))


    def pixtowl(self, pix):
        return self.wave[pix-1]


    def pixtosky(self, x, y):
        dx = x - self.head['CRPIX1']
        dy = y - self.head['CRPIX2']
        decdeg = self.head['CRVAL2'] + self.head['CD2_2'] * dy
        radeg = self.head['CRVAL1'] + (self.head['CD1_1'] * dx) /\
            np.cos( decdeg * np.pi/180.)
        return radeg, decdeg


    def skytopix(self, ra, dec):
        y = (dec - self.head['CRVAL2']) / self.head['CD2_2'] + self.head['CRPIX2']
        x = ((ra - self.head['CRVAL1']) / self.head['CD1_1']) *\
            np.cos( dec * np.pi/180.) + self.head['CRPIX1']
        return x, y


    def pixtosexa(self, x, y):
        ra, dec = self.pixtosky(x,y)
        x, y = deg2sexa(ra, dec)
        return (x, y)



    def sexatopix(self, ra, dec):
        ra, dec = sexa2deg(ra, dec)
        x, y = self.skytopix(ra,dec)
        return (int(round(x)), int(round(y)))



    def velMap(self, line='ha', dv=250):
        logger.info('Calculating valocity map')
        if line in ['Halpha', 'Ha', 'ha']:
            wlline = RESTWL['ha'] * (1 + self.z)
            minwl = wlline - 2 * dv/c * wlline
            maxwl = wlline + 2 * dv/c * wlline
        fitcube, subwl = self.subCube(minwl, maxwl)
        meanmap, sigmamap = [], []
        meanmape, sigmamape = [], []
        snmap = []
        t1 = time.time()
        for y in range(self.data.shape[1]):
            result = Parallel(n_jobs = 1, max_nbytes='1G',)\
            (delayed(gaussfit)(subwl, fitcube[:,y,i]) for i in range(self.data.shape[2]))
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

        if self.objmask != None:
            logger.info( 'Limiting range to objectmask')
            wlmean = np.nanmedian(meanmap[self.objmask == 1])
        else:
            wlmean = np.nansum(meanmap / meanmape**2) / np.nansum(1./meanmape**2)
        
        velmap = (meanmap - wlmean) / wlmean * spc.c/1E3
        logger.info('Velocity map took %.1f s' %(time.time() - t1))
        velmap[snmap < 2] = np.nan
        return np.array(velmap)



    def createaxis(self):
        minra, mindec = self.pixtosexa(self.head['NAXIS1'], 0)
        maxra, maxdec = self.pixtosexa(0, self.head['NAXIS2'])
        prera = '%s:%s:' %tuple(minra.split(':')[:2])
        minh, minm, mins = mindec.split(':')
        while True:
            mins = int(float(mins))-1
            if mins%10==0: break
        decs = [sexa2deg('00:00:00', '%s:%s:%s' %(minh, minm, mins))[1]]
        maxdec = sexa2deg('00:00:00', maxdec)[1]
        for ds in np.arange(20, 400, 20):
            if decs[0]+ds/3600. < maxdec:
                decs.append(decs[0]+ds/3600. + 1E-7)

        ras = [ int(minra.split(':')[2][:2]) + i  for i in np.arange(7,0,-1) ]
        ras = [ ra for ra in ras if ra <= float(maxra.split(':')[2][:2]) ]
        ras = [ prera + str(ra) for ra in ras]
        raspx = [self.sexatopix(ra, mindec)[0] for ra in ras ]
        ras = [r'$%s^{\rm{h}}%s^{\rm{m}}%s^{\rm{s}}$'%tuple(ra.split(':')) for ra in ras]

        decs = [ deg2sexa(0., dec)[1][:-3] for dec in decs]
        decspx = [self.sexatopix(minra, dec)[1] for dec in decs ]
        decs = [r"$%s^\circ%s'\,%s''$"%tuple(dec.split(':')) for dec in decs]
        return [raspx, ras], [decspx, decs]



    def plotxy(self, x, y, xerr=None, yerr=None, ylabel=None, xlabel=None,
                 name=''):
        fig = plt.figure(figsize = (7,4))
        fig.subplots_adjust(bottom=0.13, top=0.97, left=0.13, right=0.97)
        ax = fig.add_subplot(1, 1, 1)
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter(r'$%.1f$'))
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter(r'$%.1f$'))
        ax.errorbar(x, y, xerr = xerr, yerr=yerr, fmt = 'o', 
                    color = 'blue', alpha = 0.2, ms = 0.5,
                    # rasterized = raster,
                    mew = 2, zorder = 1)
#        ax.set_ylabel(r'$F_{\lambda}\,\rm{(10^{-17}\,erg\,s^{-1}\,cm^{-2}\, \AA^{-1})}$')
#        ax.set_xlabel(r'$\rm{Observed\,wavelength\, (\AA)}$')
#        ax.set_ylim(0)
#        ax.set_xlim(self.wave[0], self.wave[-1])
        plt.savefig('%s_%s_%s.pdf' %(self.inst, self.target, name))
        plt.close(fig)



    def plotspec(self, x, y, err=None, name=''):
        fig = plt.figure(figsize = (7,4))
        fig.subplots_adjust(bottom=0.13, top=0.97, left=0.13, right=0.97)
        ax = fig.add_subplot(1, 1, 1)
#        ax.yaxis.set_major_formatter(plt.FormatStrFormatter(r'$%f$'))
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter(r'$%i$'))
        ax.plot(x, y/1E3, color = 'grey', alpha = 1.0, # rasterized = raster,
                    drawstyle = 'steps-mid',  lw = 0.6, zorder = 1)
        ax.set_ylabel(r'$F_{\lambda}\,\rm{(10^{-17}\,erg\,s^{-1}\,cm^{-2}\, \AA^{-1})}$')
        ax.set_xlabel(r'$\rm{Observed\,wavelength\, (\AA)}$')
        ax.set_ylim(0)
        ax.set_xlim(self.wave[0], self.wave[-1])
        plt.savefig('%s_%s_%sspec.pdf' %(self.inst, self.target, name))
        plt.close(fig)


    def hiidetect(self, plane, thresh=10, median=4):
        logger.info('HII region segregation with EW threshold %i A' %(thresh))
        plane = scipy.ndimage.filters.median_filter(plane, median)
        logger.info('Median filtering input plane')
        maxdist = int(.5/self.AngD/self.pixsky)
        minpix = int(round(max(10, (0.05/self.AngD/self.pixsky)**2 * np.pi)))
        logger.info('Maximum distance from brightest region in px: %i' %maxdist)
        logger.info('Minimum area of HII region in px: %i' %minpix)
        segmap, h2count = plane * 0, 20
        h2inf = {}
        while True:
            # Highest count pixel in EW map
            h2indx = np.where(plane == np.nanmax(plane))
            h2indx = h2indx[0][0], h2indx[1][0]
            h2save = []
            # Highest count pixel below threshold we break
            if plane[h2indx] < thresh:
                break

            # Go radially around until maxdist
            for r in np.arange(maxdist):
               samereg = np.array([])
               for j in np.arange(-r, r+1, 1):
                    radflux = 0
                    for i in np.arange(-r, r+1, 1):
                        if r-1 < (j**2 + i**2)**0.5 <= r:
                            posy, posx = h2indx[0] + i, h2indx[1] + j
                            # Check pixel value at positions
                            if plane[posy, posx] > thresh:
                                h2save.append([posy, posx])
                                samereg = np.append(samereg, 1)
                            elif plane[posy, posx] < -10:
                                samereg = np.append(samereg, -2)
                            else:
                                samereg = np.append(samereg, 0)
               
               if np.mean(samereg) < 0.6:
                   break
            
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



    def pdfout(self, plane, smoothx=0, smoothy=0, name='', source='',
               label=None, vmin=None, vmax=None, ra=None, dec=None, median=None,
               psf=None):

        myeffect = withStroke(foreground="w", linewidth=2)
        kwargs = dict(path_effects=[myeffect])
#        hfont = {'fontname':'Helvetica'}

        if smoothx > 0:
            plane = blur_image(plane, smoothx, smoothy)
        if median != None:
            plane = sp.ndimage.filters.median_filter(plane, median)
        if ra != None and dec != None:
            try:
                posx, posy = self.skytopix(ra, dec)
            except TypeError:
                posx, posy = self.sexatopix(ra, dec)

        if plane.ndim == 2:
            fig = plt.figure(figsize = (11,9.5))
            fig.subplots_adjust(bottom=0.16, top=0.99, left=0.13, right=0.99)

        else:
            fig = plt.figure(figsize = (9,9))
            fig.subplots_adjust(bottom=0.18, top=0.99, left=0.18, right=0.99)
        ax = fig.add_subplot(1, 1, 1)
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter(r'$%i$'))
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter(r'$%i$'))

        ax.set_ylim(10,  self.leny-10)
        ax.set_xlim(10,  self.lenx-10)

        plt.imshow(plane, vmin=vmin, vmax=vmax)#, aspect="auto")#, cmap='Greys')

        if psf != None:
            psfrad = psf/2.3538/0.2
            psfsize = plt.Circle((30,30), psfrad, color='grey',
                                 alpha=0.7, **kwargs)
            ax.add_patch(psfsize)
            plt.text(30, 38, r'PSF',
               fontsize = 16, ha = 'center', va = 'center',  **kwargs)

        if ra != None and dec != None:
            psfrad = psf/2.3538/0.2
            psfsize = plt.Circle((posx,posy), psfrad, lw=3, fill=False,
                                 color='white', **kwargs)
            ax.add_patch(psfsize)
            psfsize = plt.Circle((posx,posy), psfrad, lw=1.5, fill=False,
                                 color='black', **kwargs)
            ax.add_patch(psfsize)            
            plt.text(posx, posy-12, source,
               fontsize = 20, ha = 'center', va = 'center',  **kwargs)

        if plane.ndim == 2:
            bar = plt.colorbar(shrink = 0.9)
    #        bar.formatter  = plt.FormatStrFormatter(r'$%.2f$')
            if not label == None:
                bar.set_label(label, size = 16)
                bar.ax.tick_params(labelsize=16)
            bar.update_ticks()
    
        [xticks, xlabels], [yticks, ylabels] = self.createaxis()
        plt.xticks(rotation=50)
        plt.yticks(rotation=50)

        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels)
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels)
        ax.set_xlabel(r'Right Ascension (J2000)', size = 16)
        ax.set_ylabel(r'Declination (J2000)', size = 16)

        plt.savefig('%s_%s_%s.pdf' %(self.inst, self.target, name))
        plt.close(fig)



    def rgb(self, planes, minval=None, maxval=None, scale='lin'):
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



    def scaleCube(self, deg=1):
      if self.scale != []:
          sfac = np.array(self.scale)[:,1]
          wls = np.array(self.scale)[:,0]

          b = np.polyfit(x=wls, y=sfac, deg = deg)
          logger.info('Scaled spectrum and error by ploynomial of degree '\
                    + '%i to %i photometric points' %(deg, len(sfac)))
          logger.info('Linear term %.e' %(b[0]))
          p = np.poly1d(b)
          corrf = p(self.wave)
          self.data *= corrf[:,np.newaxis, np.newaxis]
          self.erro *= corrf[:,np.newaxis, np.newaxis]
          fig1 = plt.figure(figsize = (6,4.2))
          fig1.subplots_adjust(bottom=0.15, top=0.97, left=0.13, right=0.96)
          ax1 = fig1.add_subplot(1, 1, 1)
          ax1.errorbar(wls, sfac, ms = 8, fmt='o', color ='firebrick')
          ax1.plot(self.wave, corrf, '-', color ='black')
          ax1.plot(self.wave, np.ones(len(corrf)), '--', color ='black')
          ax1.set_xlabel(r'$\rm{Observed\,wavelength\,(\AA)}$', fontsize=18)
          ax1.set_ylabel(r'$\rm{Correction\, factor}$', fontsize=18)
          ax1.set_xlim(4650, 9300)
          fig1.savefig('%s_%s_photcorr.pdf' %(self.inst, self.target))
          plt.close(fig1)
      
      
    def fitsout(self, plane, smoothx=0, smoothy=0, name=''):
        planeout = '%s_%s.fits' %(self.output, name)

        if smoothx > 0:
            plane = blur_image(plane, smoothx, smoothy)

        if os.path.isfile(planeout):
            os.remove(planeout)
        hdu = pyfits.HDUList()
        headimg = self.head.copy()
        headimg['NAXIS'] = 2
        for delhead in ['NAXIS3', 'CD3_3', 'CD1_3', 'CD2_3', 'CD3_1', 'CD3_2',
                        'CRPIX3', 'CRVAL3', 'CTYPE3', 'CUNIT3']:
            del headimg[delhead],
        hdu.append(pyfits.PrimaryHDU(header = self.headprim))
        hdu.append(pyfits.ImageHDU(data = plane, header = headimg))
        hdu.writeto(planeout)
        
        
    def fitsin(self, fits):
        return pyfits.getdata(fits)