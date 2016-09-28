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

import multiprocessing
import logging
import sys

from .astro import (LDMP, Avlaws, airtovac, ergJy,
                       abflux, getebv)

from .functions import (deg2sexa, sexa2deg, ccmred)
from .starlight import runStar, subStars, suballStars
from .io import pdfout, fitsout, asciiout, cubeout, plotspec
from .maps import (getDens, getSFR, getOH, getIon, getEW, getBPT, getEBV,
                   getVel, getSeg, getRGB, getTemp, getOHT)

from .extract import (extract1d, extract2d, extract3d, subtractCont,
                      getGalcen, cutCube, extractCont, RESTWL)
from .analysis import (metGrad, voronoi_bin, anaSpec)
#from .voronoi import voronoi

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
        getOH: Calculates oxygen abundance map based on strong line diagnostics
        getIon: Calculates [OIII]/Hbeta map as ionization/excitation proxy
        getEW: Calculates equivalent width maps of given line
        getEBV: Calculates EB-V maps from Balmer decrement
        BPT: Spaxels in the Baldwich-Philips-Terlevich diagram
        extractCube: Extracts cube cut in wavelength
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
        getGalcen: Get x and y index of center of galaxy (star light)

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
        self.ebvGalCorr = 0
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
        # Read in data
        self.data = pyfits.getdata(filen, 1)
        try:
            self.headerro = pyfits.getheader(filen, 2)
            # Read in variance and turn into stdev
            self.erro = pyfits.getdata(filen, 2)**0.5
        except IndexError:
            pass
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
        try:
            self.erro *= ebvcorr[:,np.newaxis, np.newaxis]
        except AttributeError:
            pass
        self.ebvGalCorr = ebv


    def ebvCor(self, line, rv=3.08, redlaw='mw', ebv=None):
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
        WL = RESTWL[line.lower()]/10.

        if ebv != None:
            ebvcalc = ebv
            ebvcorr = 1./np.exp(-1./1.086*ebvcalc * rv * Avlaws(WL, redlaw))

        elif len(self.ebvmap) != None:
            ebvcalc = self.ebvmap
            ebvcorr = 1./np.exp(-1./1.086*ebvcalc * rv * Avlaws(WL, redlaw))
            ebvcorr[np.isnan(ebvcorr)] = 1
            ebvcorr[ebvcorr < 1] = 1
            
        else:
            logger.error( 'Need an ebv or EBV-map / create via getEBV !!!')
            raise SystemExit

        return ebvcorr


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
        logger.info('Changing astrometry by %.1f" %.1f"' %(dram*3600, ddecm*3600))
        logger.info('RMS astrometry %.3f" %.3f"' %(np.std(dra)*3600, np.std(ddec)*3600))
        self.head['CRVAL1'] -= dram
        self.head['CRVAL2'] -= ddecm


    def getCont(self, pix1, pix2, dx=15):
        cont1 = np.nanmedian(self.data[pix1-dx:pix1], axis=0)
        cont2 = np.nanmedian(self.data[pix2:pix2+dx], axis=0)
        return np.nanmean(np.array([cont1,cont2]), axis=0)

    def getDens(self, **kwargs):
        return getDens(self, **kwargs)
    
    def anaSpec(self, **kwargs):
        return anaSpec(self, **kwargs)
        
    def metGrad(self, **kwargs):
        metGrad(self, **kwargs)

    def getSFR(self, **kwargs):
        return getSFR(self, **kwargs)

    def getOH(self, **kwargs):
        return getOH(self, **kwargs)

    def getOHT(self, toiii, toii, siii, **kwargs):
        return getOHT(self, toiii, toii, siii, **kwargs)

    def getIon(self, meth='S', **kwargs):
        return getIon(self)

    def getGalcen(self, **kwargs):
        return getGalcen(self, **kwargs)

    def extractCont(self, line, **kwargs):
        return extractCont(self, line, **kwargs)

    def getTemp(self, meth='SIII', **kwargs):
        return getTemp(self, meth=meth, **kwargs)

    def getEW(self, line, **kwargs):
        return getEW(self, line, **kwargs)

    def getEBV(self, **kwargs):
        return getEBV(self, **kwargs)

    def BPT(self, **kwargs):
        getBPT(self)

    def velMap(self, **kwargs):
        return getVel(self, **kwargs)

    def hiidetect(self, plane, **kwargs):
        return getSeg(self, plane, **kwargs)

    def rgb(self, planes, **kwargs):
        return getRGB(planes, **kwargs)

    def starlight(self, ascii, **kwargs):
        return runStar(self, ascii)

    def substarlight(self, x, y, **kwargs):
        subStars(self, x, y, **kwargs)

    def suball(self, **kwargs):
        suballStars(self, **kwargs)

    def pdfout(self, plane, **kwargs):
        pdfout(self, plane, **kwargs)

    def fitsout(self, plane, **kwargs):
        fitsout(self, plane, **kwargs)
        
    def cubeout(self, cube, **kwargs):
        cubeout(self, cube, **kwargs)

    def asciiout(self, wl, spec, **kwargs):
        return asciiout(self, wl, spec, **kwargs)

    def plotspec(self, wl, spec, **kwargs):
        return plotspec(self, wl, spec, **kwargs)

    def subCube(self, **kwargs):
        return extract3d(self, **kwargs)

    def cutCube(self, **kwargs):
        return cutCube(self, **kwargs)

    def extractCube(self, **kwargs):
        return  extract3d(self, **kwargs)

    def extractPlane(self, **kwargs):
        return extract2d(self, **kwargs)

    def extrSpec(self, **kwargs):
        return extract1d(self, **kwargs)

    def voronoi_bin(self, **kwargs):
        return voronoi_bin(**kwargs)


    def subtractCont(self, plane, pix1, pix2, cpix1, cpix2, dx=10):
        return  subtractCont(self, plane, pix1, pix2, cpix1, cpix2, dx=10)

    def wltopix(self, wl):
        """ Converts wavelength as input into nearest integer pixel value """
        pix = ((wl - self.wave[0]) / self.wlinc)
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
        """ Converts ra, dec positions in degrees into x, y 
        x, y in python format, starts with 0        
        """

        y = (dec - self.head['CRVAL2']) / self.head['CD2_2'] + self.head['CRPIX2']
        x = ((ra - self.head['CRVAL1']) / self.head['CD1_1']) *\
            np.cos( dec * np.pi/180.) + self.head['CRPIX1']
        return (int(round(x-1)), int(round(y-1)))


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