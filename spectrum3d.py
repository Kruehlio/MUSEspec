#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Spectrum class for 3d-spectra. Particularly MUSE."""

import matplotlib
matplotlib.use('Agg')

import pyfits
import numpy as np
import matplotlib.pyplot as plt
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
from .astro import (LDMP, Avlaws, airtovac, ergJy,
                       abflux, getebv)

from .functions import (deg2sexa, sexa2deg, ccmred)
from .fitter import onedgaussfit
from .starlight import StarLight
from .io import asciiout, cubeout, pdfout


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
        getEBV: Calculates EB-V maps from Balmer decrement
        BPT: Spaxels in the Baldwich-Philips-Terlevich diagram
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
        hiidetect: HII region detection algorithm (work in progress)
        rgb: Provided three planes, creates an RGP image
        scaleCube: Scale cube by polynomial of given degree (default 1)
    """


    def __init__(self, filen=None, inst='MUSE', target='', verbose=0):
        self.inst = inst
        self.z = None
        self.datfile = ''
        self.target = target
        self.mask = None
        self.skymask = None
        self.maskpix = None
        self.ebvmap = None
        self.objmask = None
        self.verbose = verbose
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
        self.output = self.headprim['OBJECT']
        self.base, self.ext = filen.split('.fits')[0], '.fits'
        logger.info( 'Fits cube loaded %s' %(filen))
        logger.info( 'Wavelength range %.1f - %.1f (vacuum)' %(self.wave[0], self.wave[-1]))
        self.starcube = np.zeros(self.data.shape, dtype='>f4')



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
            if self.verbose > 0:
                logger.info('Star at: %s, %s' %(ra, dec))
            wl, spec, err = self.extrSpec(ra = ra, dec = dec, radius = radius)
        else:
            wl, spec, err = self.extrSpec(total=True)

        bandsel = (wl > wls[band][0]) * (wl < wls[band][1])
        avgflux = np.nanmedian(spec[bandsel])*1E-20
        avgwl = np.nanmedian(wl[bandsel])
        fluxspec = ergJy(avgflux, avgwl)
        fluxref = abflux(mag)
        logger.info('Scale factor from spectrum to photometry for %s-band: %.3f' \
          %(band, fluxref/fluxspec))
        self.scale.append([avgwl, fluxref/fluxspec])



    def scaleCube(self, deg=1):
        """ Fits a polynomial of degree deg to the previously calculated scale-
        factors at a given wavelength, and modifies the data with the derived
        correction curve. Returns nothing, but modifies data and error instance
        attribute in place.

        Parameters
        ----------
        deg : int
            default 1, required degree of fit
        """

        if self.scale != []:
            sfac = np.array(self.scale)[:,1]
            wls = np.array(self.scale)[:,0]

            b = np.polyfit(x=wls, y=sfac, deg=deg)
            logger.info('Scaling spectrum by ploynomial of degree '\
                       + '%i to %i photometric points' %(deg, len(sfac)))
            logger.info('Linear term %.e' %(b[0]))
            p = np.poly1d(b)
            corrf = p(self.wave)
            self.data *= corrf[:,np.newaxis, np.newaxis]
            self.erro *= corrf[:,np.newaxis, np.newaxis]
            fig1 = plt.figure(figsize = (6,4.2))
            fig1.subplots_adjust(bottom=0.15, top=0.97, left=0.13, right=0.96)
            ax1 = fig1.add_subplot(1, 1, 1)
            ax1.errorbar(wls, sfac, ms=8, fmt='o', color='firebrick')
            ax1.plot(self.wave, corrf, '-', color ='black')
            ax1.plot(self.wave, np.ones(len(corrf)), '--', color='black')
            ax1.set_xlabel(r'$\rm{Observed\,wavelength\,(\AA)}$', fontsize=18)
            ax1.set_ylabel(r'$\rm{Correction\, factor}$', fontsize=18)
            ax1.set_xlim(4650, 9300)
            fig1.savefig('%s_%s_photcorr.pdf' %(self.inst, self.target))
            plt.close(fig1)
        else:
            logger.warning("No scaling performed")
            logger.warning("Calculate scaling first with checkPhot")




    def subtractCont(self, plane, pix1, pix2, cpix1, cpix2, dx=10):
        cont1 = np.nanmedian(self.data[pix1-dx:pix1], axis=0)
        cont2 = np.nanmedian(self.data[pix2:pix2+dx], axis=0)
        cont = np.nanmean(np.array([cont1,cont2]), axis=0)
        return plane - cont * (pix2 - pix1)


    def getCont(self, pix1, pix2, dx=15):
        cont1 = np.nanmedian(self.data[pix1-dx:pix1], axis=0)
        cont2 = np.nanmedian(self.data[pix2:pix2+dx], axis=0)
        return np.nanmean(np.array([cont1,cont2]), axis=0)


    def getDens(self):
        """ Derive electron density map, using the [SII] doublet and based on 
        the model of O'Dell et al. 2013 using Osterbrock & Ferland 2006
        """

        siia = self.extractPlane(line='SIIa', sC=1, meth='sum')
        siib = self.extractPlane(line='SIIb', sC=1, meth='sum')
        logne = 4.705 - 1.9875*siia/siib
        return 10**logne 
        
    def getSFR(self):
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
                     D02N2: Denicolo et. al. 2002 N2
        Returns
        -------
            ohmap : np.array
                Contains the values of 12 + log(O/H) for the given method
        """

        logger.info( 'Calculating oxygen abundance map')
        ha = self.extractPlane(line='Ha', sC=1, meth='sum')
        hb = self.extractPlane(line='Hb', sC=1, meth='sum')
        oiii = self.extractPlane(line='OIII', sC=1, meth='sum')
        nii = self.extractPlane(line='NII', sC=1, meth='sum')
        siia = self.extractPlane(line='SIIa', sC=1, meth='sum')
        siib = self.extractPlane(line='SIIb', sC=1, meth='sum')

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



    def getIon(self):
        """ Uses the ratio between a collisionally excited line ([OIII]5007)
        and the recombination line Hbeta as a tracer of ionization/excitation

        Returns
        -------
        ionmap : np.array
            Contains the values of [OIII]/Hbeta
        """

        logger.info( 'Calculating [OIII]/Hbeta map')
        hbflux = self.extractPlane(line='Hb', sC=1, meth='sum')
        oiiiflux = self.extractPlane(line='OIII', sC=1, meth='sum')
        ionmap = oiiiflux/hbflux
        return ionmap



    def getEW(self, line, dv=100):
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
        flux = self.extractPlane(line=line, sC=1, meth='sum')
        # Set continuum range, make sure no other emission line lies within
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
        # Calculate emission line rest-frame equivalent width
        ewmap = flux/cont/(1+self.z)
        return ewmap



    def BPT(self, snf=5, snb=5):
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
        ha = self.extractPlane(line='ha', sC=1)
        hae = self.extractPlane(line='ha', meth = 'error')
        oiii = self.extractPlane(line='oiiib', sC=1)
        oiiie = self.extractPlane(line='oiiib', meth = 'error')
        nii = self.extractPlane(line='nii', sC=1)
        niie = self.extractPlane(line='nii', meth = 'error')
        hb = self.extractPlane(line='hb', sC=1)
        hbe = self.extractPlane(line='hb', meth = 'error')
        sn1, sn2, sn3, sn4 = nii/niie, ha/hae, oiii/oiiie, hb/hbe
        sel = (sn1 > snf) * (sn2 > snb) * (sn3 > snb) * (sn4 > snf)

        niiha = np.log10(nii[sel].flatten()/ha[sel].flatten())
        oiiihb = np.log10(oiii[sel].flatten()/hb[sel].flatten())

        # The rest is just for the plot
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

        plt.savefig('%s_%s_BPT.pdf' %(self.inst, self.target))
        plt.close(fig)



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



    def subCube(self, wl1=None, wl2=None):
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



    def extractPlane(self, wl1='', wl2='', z=None, line=None, dv=100,
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
            if self.verbose > 0:
                logger.info( 'Subtracting continuum using lambda < %.1f and lambda > %.1f' \
                    %(cont1, cont2))
            currPlane = self.subtractCont(currPlane, pix1, pix2,
                                               cpix1, cpix2)

        if meth in ['sum', 'int']:
            currPlane = currPlane * self.wlinc
        return currPlane



    def extrSpec(self, ra=None, dec=None, x=None, y=None, 
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
                posx, posy = self.skytopix(ra, dec)
            except TypeError:
                posx, posy = self.sexatopix(ra, dec)
        elif ell != None:
            posx, posy, a, b, theta = ell
        elif total == False:
            posx, posy = x, y

        if radius==None and size==None and total==False and ell==None:
            if verbose == 1:
                logger.info('Extracting pixel %i, %i' %(posx, posy))
            spec = np.array(self.data[:,posy, posx])
            err  = np.array(self.erro[:,posy, posx])
            return self.wave, spec, err

        if total==False and radius!=None:
            if verbose == 1:
                logger.info('Creating extraction mask with radius %i arcsec' %radius)
            radpix = radius / self.pixsky
            x, y = np.indices(self.data.shape[0:2])

            exmask = np.round(((x - posy)**2  +  (y - posx)**2)**0.5)
            exmask[exmask <= radpix] = 1
            exmask[exmask > radpix] = 0

        elif total==False and size!=None:
            if verbose == 1:
                logger.info('Extracting spectrum with size %ix%i pixel' \
                %(2*size+1, 2*size+1))
            miny = max(0, posy-size)
            maxy = min(self.leny-1, posy+size+1)
            minx = max(0, posx-size)
            maxx = min(self.lenx-1, posx+size+1)

            spec = np.array(self.data[:, miny:maxy, minx:maxx])
            err  = np.array(self.erro[:, miny:maxy, minx:maxx])

            rspec = np.nansum(np.nansum(spec, axis = 1), axis=1)
            rerr = np.nansum(np.nansum(spec**2, axis = 1), axis=1)**0.5

            return self.wave, rspec, rerr

        elif total==False and ell!=None:
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
        if self.verbose > 0:
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
        if self.verbose > 0:
            logger.info('Extracting spectra took %.1f s' %(time.time()-t1))
        if pexmask == True:
            logger.info('Plotting extraction map')
            pdfout(self, exmask, name='exmask', cmap = 'gist_gray')
        return self.wave, spec, err


    def astro(self, starras, stardecs, ras, decs):
        """Correct MUSE astrometry: Starra and stardec are lists of the original
        coordinates of a source in the MUSE cube with actual coordinates ra, dec.
        Returns nothing, but changes the header keywords CRVAL1 and CRVAL2 in
        the instance attribute head. Can only correct a translation mismath,
        no rotation, plate scale changes (should be low). Length of
        starras, stardecs, ras, decs must of course be equal for the code to
        make sense.

        Parameters
        ----------
            starras : list
                List of right ascension positions in cube of reference stars
            stardecs : list
                List of declination positions in cube of reference stars
            ras : list
                List of right ascension true positions of reference stars
            decs : list
                List of declinations true positions of reference stars
        """

        if len(starras) != len(stardecs) or len(ras) != len(decs) or \
           len(starras) != len(ras):
            logger.error('Input lists must be of equal length')

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
        """ Converts wavelength as input into nearest integer pixel value """
        pix = ((wl - self.wave[0]) / self.wlinc) + 1
        return max(0, int(round(pix)))


    def pixtowl(self, pix):
        """ Converts pixel into wavelength """
        return self.wave[pix-1]


    def pixtosky(self, x, y):
        """ Converts x, y positions into ra, dec in degree """

        dx = x - self.head['CRPIX1']
        dy = y - self.head['CRPIX2']
        decdeg = self.head['CRVAL2'] + self.head['CD2_2'] * dy
        radeg = self.head['CRVAL1'] + (self.head['CD1_1'] * dx) /\
            np.cos( decdeg * np.pi/180.)
        return radeg, decdeg


    def skytopix(self, ra, dec):
        """ Converts ra, dec positions in degrees into x, y """

        y = (dec - self.head['CRVAL2']) / self.head['CD2_2'] + self.head['CRPIX2']
        x = ((ra - self.head['CRVAL1']) / self.head['CD1_1']) *\
            np.cos( dec * np.pi/180.) + self.head['CRPIX1']
        return x, y


    def pixtosexa(self, x, y):
        """ Converts x, y positions into ra, dec in sexagesimal """

        ra, dec = self.pixtosky(x,y)
        x, y = deg2sexa(ra, dec)
        return (x, y)



    def sexatopix(self, ra, dec):
        """ Converts ra, dec positions in sexagesimal into x, y """

        ra, dec = sexa2deg(ra, dec)
        x, y = self.skytopix(ra,dec)
        return (int(round(x)), int(round(y)))



    def velMap(self, line='ha', dv=250, R=2500):
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
            logger.info('Limiting range to objectmask')
            wlmean = np.nanmedian(meanmap[self.objmask == 1])
        else:
            wlmean = np.nansum(meanmap / meanmape**2) / np.nansum(1./meanmape**2)

        velmap = (meanmap - wlmean) / wlmean * spc.c/1E3
        sigmamap = (sigmamap / meanmap) * spc.c/1E3
        logger.info('Velocity map took %.1f s' %(time.time() - t1))
        velmap[snmap < 2] = np.nan
        sigmamap[snmap < 2] = np.nan
        Rsig = spc.c/(1E3 * R * 2 * (2*np.log(2))**0.5)

        return np.array(velmap), np.array(sigmamap), Rsig



    def hiidetect(self, plane, thresh=15, median=5):
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



    def starlight(self, ascii, plot=1, verbose=1):
        """ Convinience function to run starlight on an ascii file returning its
        spectral fit and bring it into original rest-frame wavelength scale again
        
        Parameters
        ----------
            ascii : str
                Filename of spectrum in Format WL SPEC ERR FLAG

        Returns
        ----------
            data : np.array (array of zeros if starlight not sucessfull)
                Original data (resampled twice, to check for accuracy)
                
            star : np.array (array of zeros if starlight not sucessfull)
                Starlight fit

            success : int
                Flag whether starlight was executed successully
        """
        
        if verbose == 1:
            logger.info('Starting starlight')
        t1 = time.time()
        sl = StarLight(filen=ascii)
        datawl, data, stars, norm, success =  sl.modOut(plot=0)
        zerospec = np.zeros(self.wave.shape)

        if success == 1:
            if verbose == 1:
                logger.info('Running starlight took %.2f s' %(time.time() - t1))
            s = sp.interpolate.InterpolatedUnivariateSpline(datawl*(1+self.z), 
                                                        data*1E3*norm/(1+self.z))
            t = sp.interpolate.InterpolatedUnivariateSpline(datawl*(1+self.z), 
                                                        stars*1E3*norm/(1+self.z))
            return s(self.wave), t(self.wave), success
        else:
            if verbose ==1:
                logger.info('Starlight failed in %.2f s' %(time.time() - t1))
            return zerospec, zerospec, success



    def substarlight(self, x, y, size=0, verbose=1):
        """ Convinience function to subtract a starlight fit based on a single
        spectrum from many spaxels
        
        Parameters
        ----------
            x : integer
                x-Index of region center
            y : integer
                y-Index of region center        
            size : integer
                Size of square around center (x,y +/- size)
        """
        
        wl, spec, err = self.extrSpec(x=x, y=y, size=size, verbose=0)
        ascii = asciiout(s3d=self, wl=wl, spec=spec, err=err, 
                              name='%s_%s_%s' %(x, y, size))
                          
        data, stars, success = self.starlight(ascii=ascii, verbose=0)
        os.remove(ascii)
        miny, maxy = max(0, y-size), min(self.leny-1, y+size+1)
        minx, maxx = max(0, x-size), min(self.lenx-1, x+size+1)
        
        xindizes=np.arange(minx, maxx, 1)
        yindizes=np.arange(miny, maxy, 1)
        zerospec = np.zeros(self.wave.shape)
        if success == 1:
#            rs = data/spec
#            logger.info('Resampling accuracy %.3f +/- %.3f' \
#                %(np.nanmedian(rs), np.nanstd(rs[1:-1])))

            for xindx in xindizes:
                for yindx in yindizes:
                    wl, spec, err = self.extrSpec(x=xindx, y=yindx, verbose=verbose)
                    # Renormalize to actual spectrum
                    substars = np.nanmedian(spec/data)*stars
                    # Overwrite starcube with fitted values
                    self.starcube[:, yindx, xindx] = substars
        else:
            for xindx in xindizes:
                for yindx in yindizes:
                    # Np sucess
                    self.starcube[:, yindx, xindx] = zerospec
        return




    def suball(self, dx=2, nc=None):
        """ Convinience function to subtract starlight fits on the full cube
        """
        
        logger.info("Starting starlight on full cube with %i cores" %self.ncores)
        logger.info("This might take a bit")
        t1 = time.time()
        xindizes = np.arange(dx, self.lenx, 2*dx+1)
        yindizes = np.arange(dx, self.leny, 2*dx+1)
#        xindizes = np.array([260, 265, 270, 275, 280])
#        yindizes = np.array([240, 245, 250, 255, 260])

        for xindx in xindizes:
            for yindx in yindizes:
                self.substarlight(xindx, yindx, dx, verbose=0)
                
        cubeout(self, self.starcube, name='star')
        cubeout(self.data-self.starcube, name='gas')
        logger.info("This took %.2f h" %((time.time()-t1)/3600.))
