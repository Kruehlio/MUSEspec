# -*- coding: utf-8 -*-

"""
IO operations for 3d fits files
    fitsin: read a given fits file into a plane
    fitsout: write a given plane into a fits file
    asciiout : Write a spectrum into a ascii file
    cubeout : Writes a 3d cube in a fits file
    pdfout: Plots a 2d-map as pdf
    createaxis: Helper function for plot
    plotxy: Simple x vs. y scatter plot
    plotspec: Simple flux vs. wavelength line plot
"""


import pyfits
import os
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.patheffects import withStroke
from matplotlib.colors import LogNorm
from ..analysis.functions import blur_image


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


def asciiin(s3d, ascii):
    """Read in an ascii file and retun the data

    Parameters
    ----------
    fits : str
        ascii file name to read

    Returns
    ----------
    wave : np.array
        wave arry of input ascii file
    spec : np.array
        data arry of input ascii file
    error : np.array
        error arry of input ascii file
    """

    f = open(ascii, 'r')
    lines = [g for g in f.readlines() if not g.startswith('#')]
    f.close()
    error = False
    wave, spec, err = np.array([]), np.array([]), np.array([])
    for line in lines:
        line = line.split()
        if len(line) >= 3:
            error = True
            wave = np.append(wave, float(line[0]))
            spec = np.append(spec, float(line[1]))
            err = np.append(err, float(line[2]))
        if len(line) == 2:
            error = False
            wave = np.append(wave, float(line[0]))
            spec = np.append(spec, float(line[1]))
    if error:
        return wave, spec, err
    else:
        return wave, spec


def fitsout(s3d, plane, smoothx=0, smoothy=0, name='', unit=''):
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

    planeout = '%s_%s_%s.fits' % (s3d.inst, s3d.output, name)

    if smoothx > 0:
        plane = blur_image(plane, smoothx, smoothy)

    if os.path.isfile(planeout):
        os.remove(planeout)
    hdu = pyfits.HDUList()
    headimg = s3d.head.copy()
    headimgprim = s3d.headprim.copy()

    headimg['NAXIS'] = 2
    for delhead in ['NAXIS3', 'CD3_3', 'CD1_3', 'CD2_3', 'CD3_1', 'CD3_2',
                    'CRPIX3', 'CRVAL3', 'CTYPE3', 'CUNIT3']:
        del headimg[delhead],
    if unit != '':
        headimg['BUNIT'] = unit
        headimgprim['BUNIT'] = unit
    hdu.append(pyfits.PrimaryHDU(header=headimgprim))
    hdu.append(pyfits.ImageHDU(data=plane, header=headimg))
    hdu.writeto(planeout)


def asciiout(s3d, wl, spec, err=None, resample=0, name='', div=3,
             frame='obs', fmt='spec'):
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
    asciiout = '%s_%s_%s.%s' % (s3d.inst, s3d.output, name, fmt)
    if s3d.z is not None and frame == 'rest':
        # logger.info('Moving to restframe')
        wls = wl / (1+s3d.z)
        divspec = spec * (1+s3d.z) / 10**div
        if err is not None:
            diverr = err * (1+s3d.z) / 10**div
    else:
        divspec = spec / 10**div
        if err is not None:
            diverr = err / 10**div

    if resample not in [False, 0, 'False']:
        outwls = np.arange(int(wls[0]), int(wls[-1]), resample)
        s = sp.interpolate.InterpolatedUnivariateSpline(wls, divspec)
        outspec = s(outwls)
        if err != None:
            t = sp.interpolate.InterpolatedUnivariateSpline(wls, diverr)
            outerr = t(outwls)
        fmt = '%.1f %.3f %.3f 0\n'
    else:
        outwls, outspec, outerr = \
            np.copy(wl), np.copy(spec) / 10**div, np.copy(err) / 10**div
        fmt = '%.3f %.3e %.3e \n'

    f = open(asciiout, 'w')
    if fmt == 'spec':
        f.write('#Fluxes in [10**-%s erg/cm**2/s/AA] \n' %(-20+div))
        f.write('#Wavelength is in vacuum and in a heliocentric reference\n')
        if s3d.ebvGalCorr != 0:
            f.write('#Fluxes are corrected for Galactic foreground\n')

    for i in range(len(outwls)):
        if err != None:
            f.write(fmt %(outwls[i], outspec[i], outerr[i]))
        if err == None:
            f.write('%.2f %.3f\n' %(outwls[i], outspec[i]))
    f.close()
#        logger.info('Writing ascii file took %.2f s' %(time.time() - t1))
    return asciiout



def cubeout(s3d, cube, name='', err=[]):
    """ Writes a 3d cube in a fits file. Maintains original header. Removes
    file if already existing

    Parameters:
    ----------
    cube : np.array
        3-d array which to write into a fits file
    name : str
        Name to use in fits file name
    err : np.array
        Write variance into second extinsion
    """

    cubeout = '%s_%s_cube.fits' %(s3d.output, name)
    if os.path.isfile(cubeout):
        os.remove(cubeout)
    hdu = pyfits.HDUList()
    hdu.append(pyfits.PrimaryHDU(header = s3d.headprim))
    hdu.append(pyfits.ImageHDU(data = cube, header = s3d.head))
    if err!=[]:
        hdu.append(pyfits.ImageHDU(data = s3d.erro**2, header = s3d.headerro))
    hdu.writeto(cubeout)



def distout(s3d, plane, minx, maxx, dx,
            plane2=None, plane3=None,
            sel=None, sel2=None, sel3=None,
            logx=True,
            name='', label='', ra=None, dec=None, cumulative=True,
            norm=True):
    """Plot the distribution of spaxel parameters in plane. Highlight the
    parameter at ra, dec if given.

    Parameters:
    ----------
    plane : np.array
        2d plane of properties which to put in histogram
    sel : np.array
        2d plane with which to downselect spaxels in plane
    minx : float
        Minimum of parameter to appear in histogram
    maxx : float
        Maximum of parameter to appear in histogram
    dx : float
        Size of bin
    logx : boolean
        Logarithmic x-axis
    cumulative : boolean
        Cumulative histogram
    norm : boolean
        Normed y-axis
    plane2 : np.array
        2d plane of second properties which to put in histogram
    sel2 : np.array
        2d plane with which to downselect spaxels in 2nd plane
    plane3 : np.array
        2d plane of second properties which to put in histogram
    sel3 : np.array
        2d plane with which to downselect spaxels in 3rd plane
    name : str
        Name to use in output plot
    label : str
        Name to use for x-axis label
    ra : str
        Right ascension of position to highlight in plot
    dec : str
        Declination of position to highlight in plot
    """

    fig = plt.figure(figsize = (6,3.5))
    fig.subplots_adjust(bottom=0.16, top=0.98, left=0.15, right=0.995)
    ax = fig.add_subplot(1, 1, 1)

    x = None
    if ra != None and dec != None:
        try:
            posx, posy = s3d.skytopix(ra, dec)
        except TypeError:
            posx, posy = s3d.sexatopix(ra, dec)
        x = plane[posy, posx]
        if plane2 != None:
            x2 = plane2[posy, posx]
        if plane3 != None:
            x3 = plane3[posy, posx]


    if sel == None:
        hist = plane.flatten()
        hist = hist[hist>minx]
        hist = hist[hist<maxx]
    else:
        hist = plane[sel]

    if plane2 != None:
        hist2 = plane2[sel2]
        hist2 = hist2[~np.isnan(hist2)]

    if plane3 != None:
        hist3 = plane3[sel3]
        hist3 = hist3[~np.isnan(hist3)]

    hist = hist[~np.isnan(hist)]
    if norm == True and x != None:
        y = len(hist[hist<x])/float(len(hist))
        if plane2 != None:
            y2 = len(hist2[hist2<x2])/float(len(hist2))
        if plane3 != None:
            y3 = len(hist3[hist3<x3])/float(len(hist3))
    else:
        y = len(hist[hist<x])

    bins = np.arange(minx, maxx, dx)
    if logx == True:
        bins = np.logspace(np.log10(minx), np.log10(maxx), 100)

    ax.hist(hist, bins = bins, cumulative=cumulative, normed=norm,
            histtype='step', lw=2, color='black')
    if plane2 != None:
        ax.hist(hist2, bins = bins, cumulative=cumulative, normed=norm,
            histtype='step', lw=2, color='navy')
    if plane3 != None:
        ax.hist(hist3, bins = bins, cumulative=cumulative, normed=norm,
            histtype='step', lw=2, color='firebrick')

    if x != None:
        ax.plot(x, y, 'o', mew=2, ms=11, color='black', mec='grey' )
        if plane2 != None:
            ax.plot(x2, y2, 'o', mew=2, ms=11, color='navy', mec='grey' )
        if plane3 != None:
            ax.plot(x3, y3, 'o', mew=2, ms=11, color='firebrick', mec='grey' )

    ax.set_xlim(minx, maxx-dx)
    ax.set_ylim(0, 1.01)
    ax.set_ylabel(r'Fraction', size = 14)
    ax.set_xlabel(label, size = 14)

    if logx == True:
        ax.set_xscale('log', subsx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    plt.savefig('%s_%s_%s.pdf' %(s3d.inst, s3d.target, name))
    plt.close(fig)


def pdfout(s3d, plane, smoothx=0, smoothy=0, name='',
           xmin=0, xmax=-1, ymin=0, ymax=-1, errsize=0.5,
           label=None, vmin=None, vmax=None,
           ra=None, dec=None, source='',
           ra2=None, dec2=None, source2='',
           median=None, axis='WCS', size = (11.5,9.15),
           psf=None, cmap='viridis', dy1=0.95, dy2=0.97,
           twoc=True, norm='lin', fs=24):

    """ Simple 2d-image plot function

    Parameters:
    ----------

    plane : np.array
        Either a 2d array for normal images, or a list of three 2d arrays for
        an RGB image
    smoothx, smoothy : integer
        Integer values of gaussian smoothing for the input plane
    xmin, xmax, ymin, ymax : integer
        Zoom into the specific pixel values of the input plane
    errsize : float
        Size of error radius with which to highlight a specific region
    label : str
        Label of region
    ra, dec : sexagesimal
        Location of label
    label2 : str
        Label of region 2
    ra2, dec2 : sexagesimal
        Location of label 2
    median : integer
        median filter the input plane
    axis : string
        WCS axis or none
    psf : float
        FWHM of the PSF which we draw at the left bottom corner of the output
    cmap : string
        Color map, default virdis
    twoc : boolean
        Write text in image in black and white letters
    norm : string
        How to normalize the RGB image (lin, sqrt, log)
    fs : integer
        Fontsize (default 24)
    """

    if xmax == -1:
        xmax = plane.shape[0]
    if ymax == -1:
        ymax = plane.shape[1]

    plane = plane[ymin:ymax, xmin:xmax]

    if twoc == True:
        myeffect = withStroke(foreground="w", linewidth=4)
        kwargs = dict(path_effects=[myeffect])
    else:
        kwargs = {}
#        hfont = {'fontname':'Helvetica'}

    if smoothx > 0:
        plane = blur_image(plane, smoothx, smoothy)
    if median != None:
        plane = sp.ndimage.filters.median_filter(plane, median, mode='constant')

    if ra != None and dec != None:
        try:
            posx, posy = s3d.skytopix(ra, dec)
        except TypeError:
            posx, posy = s3d.sexatopix(ra, dec)
        if xmin !=0:
            posx -= xmin
        if ymin !=0:
            posy -= ymin

    if ra2 != None and dec2 != None:
        try:
            posx2, posy2 = s3d.skytopix(ra2, dec2)
        except TypeError:
            posx2, posy2 = s3d.sexatopix(ra2, dec2)
        if xmin !=0:
            posx2 -= xmin
        if ymin !=0:
            posy2 -= ymin


    if plane.ndim == 2:
        fig = plt.figure(figsize = size)
        if fs > 20 and axis == 'WCS':
            fig.subplots_adjust(bottom=0.22, top=0.99, left=0.12, right=0.96)
        elif axis == 'WCS':
            fig.subplots_adjust(bottom=0.20, top=0.99, left=0.08, right=0.96)
        else:
            fig.subplots_adjust(bottom=0.005, top=0.995, left=0.005, right=0.94)
    else:
        fig = plt.figure(figsize = (9,9))
        fig.subplots_adjust(bottom=0.18, top=0.99, left=0.19, right=0.99)
    ax = fig.add_subplot(1, 1, 1)


    ax.set_ylim(5,  plane.shape[0]-5)
    ax.set_xlim(5,  plane.shape[1]-5)

    if norm == 'lin':
        plt.imshow(plane, vmin=vmin, vmax=vmax, #extent=[],
               cmap=cmap, interpolation="nearest")#, aspect="auto")#, cmap='Greys')

    elif norm == 'log':
        plt.imshow(plane, vmin=vmin, vmax=vmax, #extent=[],
               cmap=cmap, norm=LogNorm(), interpolation="nearest")#, aspect="auto")#, cmap='Greys')

    if plane.ndim == 2:
        bar = plt.colorbar(shrink = 0.9, pad = 0.01)
        if not label == None:
            bar.set_label(label, size = fs+15, family='serif')
            bar.ax.tick_params(labelsize=max(24,fs-4))
        if norm == 'log':
            bar.formatter  = plt.LogFormatterMathtext()
        labels = [item.get_text() for item in bar.ax.get_yticklabels()]
        newlab = []
        for label in labels:
            if norm == 'log':
                newl = label.replace('mathdefault', 'mathrm', 1)
            else:
                newl = r'$%s$' %label
            newlab.append(newl)
        bar.ax.set_yticklabels(newlab)
#            bar.update_ticks()

    if psf != None:
        psfrad = psf/2.3538/s3d.pixsky
        psfsize = plt.Circle((8*plane.shape[0]/9., plane.shape[1]/9.),
                             psfrad, color='black',
                             alpha=1, **kwargs)
        ax.add_patch(psfsize)
        plt.text(8*plane.shape[0]/9., plane.shape[0]/6.5, r'PSF',
           fontsize = fs, ha = 'center', va = 'center',  **kwargs)

    if ra != None and dec != None:
        psfrad = errsize/2.3538/s3d.pixsky
        psfsize = plt.Circle((posx,posy), psfrad, lw=5, fill=False,
                             color='white', **kwargs)
        ax.add_patch(psfsize)
        psfsize = plt.Circle((posx,posy), psfrad, lw=1.5, fill=False,
                             color='black', **kwargs)
        ax.add_patch(psfsize)
        plt.text(posx, posy*dy1, source,
           fontsize = fs, ha = 'center', va = 'top',  **kwargs)

    if ra2 != None and dec2 != None:
#        psfrad = 0.2*errsize/2.3538/s3d.pixsky
#        psfsize = plt.Circle((posx2,posy2), psfrad, lw=5, fill=False,
#                             color='white', **kwargs)
#        ax.add_patch(psfsize)
#        psfsize = plt.Circle((posx2,posy2), psfrad, lw=1.5, fill=False,
#                             color='black', **kwargs)
#        ax.add_patch(psfsize)
        print posx2, posy2
        ax.plot(posx2+1, posy2+1, 'o', ms=9, mec='black', c='black')
        plt.text(posx2, posy2*dy2, source2,
           fontsize = fs, ha = 'center', va = 'top',  **kwargs)

    if axis == 'WCS':

        [xticks, xlabels], [yticks, ylabels] = _createaxis(s3d, plane)

        sel = (xmin+5 < xticks) * (xmax-5 > xticks)
        xticks = xticks[sel]
        xlabels = xlabels[sel]
        xticks -= xmin

        sel = (ymin+5 < yticks) * (ymax-5 > yticks)
        yticks = yticks[sel]
        ylabels = ylabels[sel]
        yticks -= ymin


        plt.xticks(rotation=50)
        plt.yticks(rotation=50)

        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels, size=fs)
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels, size=fs)
        ax.set_xlabel(r'Right Ascension (J2000)', size = fs)
        ax.set_ylabel(r'Declination (J2000)', size = fs)

    else:
        ax.yaxis.set_major_formatter(plt.NullFormatter())
        ax.xaxis.set_major_formatter(plt.NullFormatter())

    plt.savefig('%s_%s_%s.pdf' %(s3d.inst, s3d.target, name))
    plt.close(fig)



def _createaxis(s3d, plane):
    """ WCS axis helper method """

    minra, mindec = s3d.pixtosexa(s3d.head['NAXIS1']-20, 20)
    maxra, maxdec = s3d.pixtosexa(20, s3d.head['NAXIS2']-20)

    if plane.shape[0] > 1000:
        dy = 20
    elif plane.shape[0] > 500:
        dy = 30
    elif plane.shape[0] > 100:
        dy = 10
    else:
        dy = 3

    if plane.shape[1] > 1000:
        dx = 2
    elif plane.shape[1] > 500:
        dx = 3
    elif plane.shape[1] > 100:
        dx = 1
    else:
        dx = 0.3

    minrah = int(minra.split(':')[0])
    minram = int(minra.split(':')[1])-1
    minras = np.ceil(float(minra.split(':')[-1]))

    axpx, axlab, aypx, aylab = [], [], [], []

    def az(numb, dx=1):
        if 0 <= numb < 10:
            if dx >= 1:
                return '0%i' %numb
            else:
                return '0%.1f' %numb
        elif -10 < numb < 0:
            return '-0%i' %np.abs(numb)
        else:
            if dx >= 1:
                return '%i' %numb
            else:
                return '%.1f' %numb

    while True:
        if minras >= 60:
            minram += 1
            minras -= 60
        if minram >= 60:
            minrah += 1
            minram -= 60
        if minrah >= 24:
            minrah -= 24
        axra = '%s:%s:%s' %(az(minrah), az(minram), az(minras, dx))
        xpx = s3d.sexatopix(axra, mindec)[0]
        if xpx < 0:
            break
        else:
            axpx.append(xpx)
            axlab.append(r'$%s^{\rm{h}}%s^{\rm{m}}%s^{\rm{s}}$'\
                    %tuple(axra.split(':')) )
        minras += dx

    if s3d.headprim['DEC'] > 0:
        maxdec = mindec
        dy = -dy
        maxdem = int(maxdec.split(':')[1])+1
    else:
        maxdem = int(maxdec.split(':')[1])-1

    maxdeh = int(maxdec.split(':')[0])
    maxdes = np.round(float(maxdec.split(':')[-1]) + 10, -1)

    haslabel = 0
    while True:
        if maxdes >= 60:
            maxdem += 1
            maxdes -= 60
        if maxdem >= 60:
            maxdeh += 1
            maxdem -= 60
        if maxdes < 0:
            maxdem -= 1
            maxdes += 60
        if maxdem < 00:
            maxdeh -= 1
            maxdem += 60


        axdec = '%s:%s:%s' %(az(maxdeh), az(maxdem), az(maxdes))
        ypx = s3d.sexatopix(minra, axdec)[1]
        if ypx < 0 and haslabel == 1:
            break
        else:
            aypx.append(ypx)
            aylab.append(r"$%s^\circ%s'\,%s''$"%tuple(axdec.split(':')))
            if ypx > 5:
                haslabel = 1
        maxdes += dy

    return [np.array(axpx), np.array(axlab)], [np.array(aypx), np.array(aylab)]




def plotspec(s3d, x, y, err=None, name='', div=1E3, lines=[]):
    """ Simple spectrum bar plot convinience method """

    fig = plt.figure(figsize = (7,4))
    fig.subplots_adjust(bottom=0.13, top=0.97, left=0.13, right=0.97)
    ax = fig.add_subplot(1, 1, 1)
#        ax.yaxis.set_major_formatter(plt.FormatStrFormatter(r'$%f$'))
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter(r'$%i$'))
    ax.plot(x, y/div, color = 'black', alpha = 1.0, # rasterized = raster,
                drawstyle = 'steps-mid',  lw = 1.8, zorder = 1)
    colors = ['firebrick', 'navy']
    if lines != []:
        for color, line in zip(colors, lines):
            if len(lines) == 2:
                ax.plot(line[0], line[1], '-', color=color, lw=1.2)
    if err != None:
        ax.plot(x, err/div, color = 'grey', alpha = 1.0, # rasterized = raster,
                drawstyle = 'steps-mid',  lw = 0.6, zorder = 1)
    ax.set_ylabel(r'$F_{\lambda}\,\rm{(10^{-17}\,erg\,s^{-1}\,cm^{-2}\, \AA^{-1})}$')
    ax.set_xlabel(r'$\rm{Observed\,wavelength\, (\AA)}$')
#    if div == 1E3:
#        ax.set_ylim(min(min(y/div)*1.2, 0))
    ax.set_xlim(x[0],x[-1])
#    ylim = -2*np.std(y[s3d.wltopix(5500):s3d.wltopix(6500)])
#    ax.set_ylim(ylim/div)

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
