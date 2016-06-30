#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Spectrum class for 3d-spectra. Particularly MUSE.
"""

__version__ = '0.1'
__author__ = 'Thomas Kruehler'

import matplotlib
matplotlib.use('Agg')

import logging
import sys
import pyfits
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import os
import zap
import time
import subprocess

from MUSEspec.astro import airtovac
from MUSEspec.functions import smooth, checkExec
from MUSEspec.fitter import onedgaussfit

logfmt = '%(levelname)s [%(asctime)s]: %(message)s'
datefmt= '%Y-%m-%d %H:%M:%S'
formatter = logging.Formatter(fmt=logfmt,datefmt=datefmt)
logger = logging.getLogger('__main__')

import signal
def signal_handler(signal, frame):
    sys.exit("CTRL+C detected, stopping execution")
signal.signal(signal.SIGINT, signal_handler)


class MuseSpec:
    """ MUSE class for data manipulation, ZAPing, Skysubtractions.
   
   Arguments:
        inst: Instrument that produced the spectrum (optional, default=xs)
        filen: Filename of fits data cube
        
    Methods:
        setFiles: Sets fits files
        makemask: Create an object mask
        cubezap: Runs ZAP (Zurich Athmospheric Package) on files
        fpack: fpacks the output file
        writeFiles: writes the output into a fits file
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

    def setFiles(self, filen, fluxmult=1, dAxis=3, mult=1):
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
        # Read in variance
        self.erro = pyfits.getdata(filen, 2)**0.5
        if self.target == '':
            self.target = self.headprim['OBJECT']
        self.fluxunit = self.head['BUNIT']
        self.output = filen.split('.fits')[0]
        self.base, self.ext = filen.split('.fits')[0], '.fits'
        logger.info( 'Fits cube loaded %s' %(filen))
        logger.info( 'Wavelength range %.1f - %.1f (vacuum)' %(self.wave[0], self.wave[-1]))



    def makemask(self, sky=None, line=None, continuum=True,
                 wl1=None, wl2=None, clobber=True, sigma=2.5,
                 iterate=10, writefits=True):
        """ Creates an object mask by medianing a data cube between wl1 and wl2,
        pixels with Objects are denoted with 1, sky pixels with 0. If wl1 or wl2
        are not given, uses the full cube
        Optional inputs:
        -sky Skycube
        -wl1 Wavelength start
        -wl2 Wavelength end
        -sigma for source detection
        -iterate iteration for source detection
        """
        
        if not sky:
            logger.info( 'Creating the object mask with sigma %s in %i iterations' \
                %(sigma, iterate))

            self.mask = '%s_mask%s' %(self.base, self.ext)
            maskdata = self.data
            head = self.head.copy()
            mask = self.mask

        else:
            logger.info( 'Creating the sky mask with sigma %s in %i iterations' \
                %(sigma, iterate))
            self.skymask = '%s_mask_sky%s' %(self.base, self.ext)

            maskdata = pyfits.getdata(sky, 1)
            head = pyfits.getheader(sky, 1)
            mask = self.skymask

        if os.path.isfile(mask) and clobber != False:
            os.remove(mask)

        if not os.path.isfile(mask):

            img1 = np.median(maskdata[10:-10], axis = 0)
            bpimg = np.array(img1)

            if continuum == True:
                logger.info( 'Using continuum for object mask')
                newimg = img1[~np.isnan(img1)]
                stdev = np.nanstd(newimg)
                medimg = np.nanmedian(newimg)
                newimg = newimg[newimg > (medimg -7 * stdev)]
                for i in range(iterate):
                    newimg = newimg[newimg < (medimg + sigma*stdev)]
                    newimg = newimg[newimg > (medimg -7 * stdev)]
                    stdev = np.nanstd(newimg)
                    medimg = np.nanmedian(newimg)
                contsel = img1 <= medimg + sigma*stdev
                bpimg[img1 < medimg - 7*stdev] = 1E5

            img2 = None
            if wl1 and wl2:
                pix1 = self.wltopix(wl1)
                pix2 = max(pix1+1, self.wltopix(wl2))
                img2 = np.sum(maskdata[pix1:pix2], axis = 0)

            elif line and self.z:
                logger.info( 'Using %s line for object mask' %line)
                img2 = self.extractPlane(line=line, sC = 1, meth = 'sum')

            if img2 != None:
                newimg = img2[~np.isnan(img2)]
                linestdev = np.nanstd(newimg)
                linemedimg = np.nanmedian(newimg)
                newimg = newimg[newimg > (linemedimg -7 * linestdev)]
                for i in range(iterate):
                    newimg = newimg[newimg < (linemedimg + (sigma-1)*linestdev)]
                    newimg = newimg[newimg > (linemedimg -7 * linestdev)]
                    linestdev = np.nanstd(newimg)
                    linemedimg = np.nanmedian(newimg)
                linesel = img2 <= linemedimg + (sigma)*linestdev
                bpimg[img2 < linemedimg - 7*linestdev] = 1E5

            if continuum == True and img2 != None:
                sel = linesel * contsel
            elif continuum == True and img2 == None:
                sel = contsel
            elif continuum == False and img2 != None:
                sel = linesel

            bpimg[np.isnan(bpimg)] = 1E5
            bpimg[sel] = 0
            bpimg[bpimg != 0] = 1
            self.objmask = bpimg
            self.maskpix = bpimg[bpimg==1].size/float(bpimg.size)
            logger.info('Masking %.2f of all spaxels' \
                %(self.maskpix))

            if writefits == True:
                hdu = pyfits.HDUList()
                head['NAXIS'] = 2
                for delhead in ['NAXIS3', 'CD3_3', 'CD1_3', 'CD2_3', 'CD3_1', 'CD3_2',
                                'CRPIX3', 'CRVAL3', 'CTYPE3', 'CUNIT3']:
                    del head[delhead],
                hdu.append(pyfits.PrimaryHDU(header = self.headprim))
                hdu.append(pyfits.ImageHDU(data = bpimg, header = head))
                hdu.writeto(mask)



    def cubezap(self, cube = None, skycube = None, mask = None, out = None,
                cfwidthSP = 20, skymask = None, nevals=[]):
        if nevals != []:
            optimizeType = 'none'
        else:
            optimizeType = 'normal'
        if os.path.isfile('ZAP_SVD.fits'):
            os.remove('ZAP_SVD.fits')
        if not out:
            out = '%s_zap%s' %(self.base, self.ext)
            if os.path.isfile(out):
                os.remove(out)
        if not cube:
            inc = self.base + self.ext
        if not mask:
            mask = self.mask
        if not skymask:
            skymask = self.skymask

        if skycube:
            svdfn = '%s_svd%s' %(self.base, self.ext)
            if os.path.isfile(svdfn):
                os.remove(svdfn)
            zap.SVDoutput(skycube, svdoutputfits = svdfn, mask=skymask)
            zap.process(inc, outcubefits=out, extSVD = svdfn, #mask=mask,
                        cfwidthSP = cfwidthSP, nevals=nevals,
                        optimizeType = optimizeType)
        else:
            zap.process(inc, outcubefits=out, mask = mask, nevals=nevals,
                        cfwidthSP = cfwidthSP,
                        optimizeType = optimizeType)
        self.output = out

    def subtractSky(self, skypix=None, order = 3, ff=0.03, plot = 0):
        if skypix == None:
            skypix = 1-self.maskpix
        if skypix == None:
            skypix = 0.85
        t1, back = time.time(), np.array([])
        for image in self.data:
            lenx = len(image[0])
            leny = len(image[1])
            minpix, maxpix = int(ff*lenx), int((1-ff)*lenx)
            workimg = image[minpix:maxpix, minpix:maxpix]
            workimg = np.sort(image[~np.isnan(image)]) [ : int(skypix*np.size(image))]

            skyregs = np.array([])
            for j in np.arange(15):
                try:
                    skyindex = np.random.randint(low = 0, high = max(3000, len(workimg)/10))
                    skymedreg = np.where(image == workimg[skyindex])
                    skyreg = image[max(0, skymedreg[0]-15) : min(skymedreg[0]+15, lenx-1),
                                   max(0, skymedreg[1]-15) : min(skymedreg[1]+15, leny-1)]
                    skyregs = np.append(skyregs, skyreg)
                except (ValueError, IndexError):
                    pass
            skyregs = skyregs[~np.isnan(skyregs)]

            yfit, bins = np.histogram(skyregs, bins=30)
            xfit = (bins[1]-bins[0])/2. + bins
            params = onedgaussfit(xfit[:-1], yfit, #err = yfit**0.5,
                                  params=[0, 1, -1, 10],
                                  fixed=[1, 0, 0, 0],
                                  minpars=[0, 0, -20, 0],
                                  limitedmin=[1, 1, 0, 1])

            back = np.append(back, params[0][2])

        backsmooth, backrms = smooth(back[~np.isnan(back)], 20, window='median', rms=1)
        a = sp.polyfit(self.pix[~np.isnan(back)], backsmooth, order, w=1/backrms)
        backfit = sp.polyval(a, self.pix)
        i = 0
        for image in self.data:
            self.data[i] -= backfit[i]
            i+=1
        self.output = '%s_skysub%s' %(self.base, self.ext)

        fig = plt.figure(figsize = (8,6))
        ax = fig.add_subplot(1, 1, 1)
#        ax.yaxis.set_major_formatter(plt.FormatStrFormatter(r'$%i$'))
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter(r'$%i$'))
        ax.set_xlim(self.wave[0],  self.wave[-1])
        ax.plot(self.wave[~np.isnan(back)], back[~np.isnan(back)], 'o', mec = 'grey', ms = 2)
        ax.plot(self.wave, backfit, lw=2, color = 'firebrick')
        ax.plot(self.wave[~np.isnan(back)], backsmooth, 'o', color = 'olive', ms=1)

        ax.set_ylim(min(backfit)-abs(max(backfit)), max(backfit)+abs(max(backfit)))
        ax.set_xlabel(r'$\rm{Wavelength\,(\AA)}$')
        ax.set_ylabel(r'$\rm{Flux\,(10^{-20}\,erg\,cm^{-2}\,s^{-1}\,\AA^{-1})}$')
        plt.savefig('%s_%s_back.pdf' %(self.inst, self.target))
        plt.close()
        logger.info( 'Skysubtraction took %.0f s' %(time.time()-t1))



    def fpack(self):
        """ Run fpack on output file using fpack -q 8 -D -Y"""
        if os.path.isfile(self.output):
            fpack = checkExec(['fpack'])
            if fpack:
                logger.info( 'Fpacking %s' % self.output)
                proc = subprocess.Popen([fpack, '-q', '8', '-D', '-Y', '%s' %self.output],
                    stdout = subprocess.PIPE, stderr = subprocess.PIPE)
                proc.wait()



    def writeFiles(self):
        """  Writes the data variable into a fits file """
        if os.path.isfile(self.output):
            os.remove(self.output)
        hdu = pyfits.HDUList()
        hdu.append(pyfits.PrimaryHDU(header = self.headprim))
        hdu.append(pyfits.ImageHDU(data = self.data, header = self.head))
        hdu.append(pyfits.ImageHDU(data = self.erro, header = self.headerro))
        hdu.writeto(self.output)