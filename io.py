# -*- coding: utf-8 -*-

""" 
IO operations for 3d fits files
"""


import pyfits
import os
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.patheffects import withStroke

from .functions import blur_image


"""
Functions:
    pdfout: Plots a 2d-map as pdf
    createaxis: Helper function for plot
    plotxy: Simple x vs. y scatter plot
    plotspec: Simple flux vs. wavelength line plot
    fitsout: write a given plane into a fits file
    fitsin: read a given fits file into a plane
"""


def fitsin(fits):
    """Read in a fits file and retun the data

    Parameters
    ----------
    fits : str
        fits file name to read

    Returns
    ----------
    data : np.array
        data arry of input fits file
    """

    data = pyfits.getdata(fits)
    return data


def fitsout(s3d, plane, smoothx=0, smoothy=0, name=''):
    """ Write the given plane into a fits file. Uses header of the original
    data. Returns nothing, but write a fits file.

    Parameters
    ----------
    plane : np.array
        data to store in fits file
    smoothx : int
        possible Gaussian smoothing length in x-coordinate (default 0)
    smoothy : int
        possible Gaussian smoothing length in y-coordinate (default 0)
    name : str
        Name to use in fits file name
    """

    planeout = '%s_%s.fits' %(s3d.output, name)

    if smoothx > 0:
        plane = blur_image(plane, smoothx, smoothy)

    if os.path.isfile(planeout):
        os.remove(planeout)
    hdu = pyfits.HDUList()
    headimg = s3d.head.copy()
    headimg['NAXIS'] = 2
    for delhead in ['NAXIS3', 'CD3_3', 'CD1_3', 'CD2_3', 'CD3_1', 'CD3_2',
                    'CRPIX3', 'CRVAL3', 'CTYPE3', 'CUNIT3']:
        del headimg[delhead],
    hdu.append(pyfits.PrimaryHDU(header = s3d.headprim))
    hdu.append(pyfits.ImageHDU(data = plane, header = headimg))
    hdu.writeto(planeout)
    
    
def asciiout(s3d, wl, spec, err=None, resample=1, name=''):
    """ Write the given spectrum into a ascii file. 
    Returns name of ascii file, writes ascii file.

    Parameters
    ----------
    wl : np.array
        wavelength array
    spec : np.array
        spectrum array
    err : np.array
        possible error array (default None)
    resample : int
        wavelength step in AA to resample
    name : str
        Name to use in fits file name
    """
    asciiout = '%s_%s.txt' %(s3d.output, name)
    if s3d.z != None:
#            logger.info('Moving to restframe')
        wls = wl / (1+s3d.z)
        spec = spec * (1+s3d.z) * 1E-3
        if err != None:
            err = err * (1+s3d.z) * 1E-3
    else:
        spec *= 1E-20
        if err != None:
            err = err * 1E-3
    outwls = np.arange(int(wls[0]), int(wls[-1]), resample)

    s = sp.interpolate.InterpolatedUnivariateSpline(wls, spec)
    outspec = s(outwls)

    if err != None:
        t = sp.interpolate.InterpolatedUnivariateSpline(wls, err)
        outerr = t(outwls)

    f = open(asciiout, 'w')
    for i in range(len(outwls)):
        if err != None:
            f.write('%.1f %.3f %.3f 0\n' %(outwls[i], outspec[i], outerr[i]))
        if err == None:
            f.write('%.1f %.3f\n' %(outwls[i], outspec[i]))            
    f.close()
#        logger.info('Writing ascii file took %.2f s' %(time.time() - t1))
    return asciiout    
    
    

def cubeout(s3d, cube, name='', err=False):
    """ Writes a 3d cube in a fits file. Maintains original header. Removes
    file if already existing
    """
    
    cubeout = '%s_%s_cube.fits' %(s3d.output, name)
    if os.path.isfile(cubeout):
        os.remove(cubeout)
    hdu = pyfits.HDUList()
    hdu.append(pyfits.PrimaryHDU(header = s3d.headprim))
    hdu.append(pyfits.ImageHDU(data = cube, header = s3d.head))
    if err == True:
        hdu.append(pyfits.ImageHDU(data = s3d.erro**2, header = s3d.headerro))
    hdu.writeto(cubeout)
 

def pdfout(s3d, plane, smoothx=0, smoothy=0, name='', source='',
           label=None, vmin=None, vmax=None, ra=None, dec=None, median=None,
           psf=None, cmap='viridis'):
               
    """ Simple 2d-image plot function """

    myeffect = withStroke(foreground="w", linewidth=2)
    kwargs = dict(path_effects=[myeffect])
#        hfont = {'fontname':'Helvetica'}

    if smoothx > 0:
        plane = blur_image(plane, smoothx, smoothy)
    if median != None:
        plane = sp.ndimage.filters.median_filter(plane, median)
    if ra != None and dec != None:
        try:
            posx, posy = s3d.skytopix(ra, dec)
        except TypeError:
            posx, posy = s3d.sexatopix(ra, dec)

    if plane.ndim == 2:
        fig = plt.figure(figsize = (11,9.5))
        fig.subplots_adjust(bottom=0.16, top=0.99, left=0.13, right=0.99)

    else:
        fig = plt.figure(figsize = (9,9))
        fig.subplots_adjust(bottom=0.18, top=0.99, left=0.18, right=0.99)
    ax = fig.add_subplot(1, 1, 1)

    ax.set_ylim(10,  s3d.leny-10)
    ax.set_xlim(10,  s3d.lenx-10)

    plt.imshow(plane, vmin=vmin, vmax=vmax, cmap=cmap)#, aspect="auto")#, cmap='Greys')

    if psf != None:
        psfrad = psf/2.3538/0.2
        psfsize = plt.Circle((30,30), psfrad, color='grey',
                             alpha=0.7, **kwargs)
        ax.add_patch(psfsize)
        plt.text(30, 44, r'PSF',
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

    [xticks, xlabels], [yticks, ylabels] = createaxis(s3d)
    plt.xticks(rotation=50)
    plt.yticks(rotation=50)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, size=16)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, size=16)
    ax.set_xlabel(r'Right Ascension (J2000)', size = 20)
    ax.set_ylabel(r'Declination (J2000)', size = 20)

    plt.savefig('%s_%s_%s.pdf' %(s3d.inst, s3d.target, name))
    plt.close(fig)   
    
    
    
    
def createaxis(s3d):
    """ WCS axis helper method """

    if True:
        minra, mindec = s3d.pixtosexa(s3d.head['NAXIS1']-20, 20)
        maxra, maxdec = s3d.pixtosexa(20, s3d.head['NAXIS2']-20)

        if s3d.head['NAXIS1'] > 500:
            dx = 30
        else:
            dx = 20
        if s3d.head['NAXIS2'] > 500:
            dy = 2
        else:
            dy = 1

        minram = int(minra.split(':')[1])
        minrah = int(minra.split(':')[0])
        minras = np.ceil(float(minra.split(':')[-1]))

        axpx, axlab, aypx, aylab = [], [], [], []

        def az(numb):
            if 0 <= numb < 10:
                return '0%i' %numb
            elif -10 < numb < 0:
                return '-0%i' %np.abs(numb)
            else:
                return '%i' %numb

        while True:
            if minras >= 60:
                minram += 1
                minras -= 60
            if minram >= 60:
                minrah += 1
                minram -= 60
            if minrah >= 24:
                minrah -= 24
            axra = '%s:%s:%s' %(az(minrah), az(minram), az(minras))
            xpx = s3d.sexatopix(axra, mindec)[0]
            if xpx > s3d.head['NAXIS1'] or xpx < 0:
                break
            else:
                axpx.append(xpx)
                axlab.append(r'$%s^{\rm{h}}%s^{\rm{m}}%s^{\rm{s}}$'%tuple(axra.split(':')) )
            minras += dy
        
        if s3d.headprim['DEC'] > 0:
            maxdec = mindec
        maxdem = int(maxdec.split(':')[1])
        maxdeh = int(maxdec.split(':')[0])
        maxdes = np.round(float(maxdec.split(':')[-1]) + 10, -1)

        while True:
            if maxdes >= 60:
                maxdem += 1
                maxdes -= 60
            if maxdem >= 60:
                maxdeh += 1
                maxdem -= 60

            axdec = '%s:%s:%s' %(az(maxdeh), az(maxdem), az(maxdes))
            ypx = s3d.sexatopix(minra, axdec)[1]
            if ypx > s3d.head['NAXIS2'] or ypx < 0:
                break
            else:
                aypx.append(ypx)
                aylab.append(r"$%s^\circ%s'\,%s''$"%tuple(axdec.split(':')))
            maxdes += dx

        return [axpx, axlab], [aypx, aylab]
 


   
def plotspec(s3d, x, y, err=None, name='', div=1E3):
    """ Simple spectrum bar plot convinience method """

    fig = plt.figure(figsize = (7,4))
    fig.subplots_adjust(bottom=0.13, top=0.97, left=0.13, right=0.97)
    ax = fig.add_subplot(1, 1, 1)
#        ax.yaxis.set_major_formatter(plt.FormatStrFormatter(r'$%f$'))
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter(r'$%i$'))
    ax.plot(x, y/div, color = 'black', alpha = 1.0, # rasterized = raster,
                drawstyle = 'steps-mid',  lw = 0.8, zorder = 1)
    if err != None:
        ax.plot(x, err/div, color = 'grey', alpha = 1.0, # rasterized = raster,
                drawstyle = 'steps-mid',  lw = 0.6, zorder = 1)
    ax.set_ylabel(r'$F_{\lambda}\,\rm{(10^{-17}\,erg\,s^{-1}\,cm^{-2}\, \AA^{-1})}$')
    ax.set_xlabel(r'$\rm{Observed\,wavelength\, (\AA)}$')
    if div == 1E3:
        ax.set_ylim(min(min(y/div)*1.2, 0))
    ax.set_xlim(s3d.wave[0], s3d.wave[-1])
    plt.savefig('%s_%s_%sspec.pdf' %(s3d.inst, s3d.target, name))
    plt.close(fig)
   
   
   
   
def plotxy(s3d, x, y, xerr=None, yerr=None, ylabel=None, xlabel=None,
             name=''):
    """ Simple error bar plot convinience method """


    fig = plt.figure(figsize = (7,4))
    fig.subplots_adjust(bottom=0.13, top=0.97, left=0.13, right=0.97)
    ax = fig.add_subplot(1, 1, 1)
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter(r'$%.1f$'))
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter(r'$%.1f$'))
    ax.errorbar(x, y, xerr = xerr, yerr=yerr, fmt = 'o',
                color = 'blue', alpha = 0.2, ms = 0.5,
                # rasterized = raster,
                mew = 2, zorder = 1)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(ylabel)
    plt.savefig('%s_%s_%s.pdf' %(s3d.inst, s3d.target, name))
    plt.close(fig)
    