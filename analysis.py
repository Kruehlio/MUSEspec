# -*- coding: utf-8 -*-

"""
Performs analysis on MUSE cubes
    metGrad : Plots the metallicity as a function of galaxy distance 
    voronoi_run : Runs a voronoi tesselation code
    voronoi_bin : Bins a 2d-map using voronoi tesselation
"""

import numpy as np
import matplotlib.pyplot as plt
import logging
import scipy as sp
from matplotlib.backends.backend_pdf import PdfPages

from .extract import getGalcen, RESTWL
from .maps import getOH
from .voronoi import voronoi
from .astro import bootstrap, geterrs
from .fitter import (linfit, onedgaussfit, onedgaussian, onedtwogaussian)
from .formulas import (mcebv, mcohD16, mcTSIII, mcOHOIII, mcSHSIII, 
                mcDens, mcohPP04, mcSFR)


logfmt = '%(levelname)s [%(asctime)s]: %(message)s'
datefmt= '%Y-%m-%d %H:%M:%S'
formatter = logging.Formatter(fmt=logfmt,datefmt=datefmt)
logger = logging.getLogger('__main__')
logging.root.setLevel(logging.DEBUG)
ch = logging.StreamHandler() #console handler
ch.setFormatter(formatter)
logger.handlers = []
logger.addHandler(ch)


def anaSpec(s3d, wl, spec, err, plotLines=False, printFlux=True,
            name='', div = 1E3, hasC=0):
    
    lines = ['ha', 'hb', 'niia', 'niib', 'siia', 'siib', 'oiiia', 'oiiib',
             'siii6312', 'siii', 'ariii7135', 'ariii7751', 'oii7320',
             'oii7331', 'nii5755', 'hei5876']
           
    lp = {}           
           
    if plotLines==True:
        pp = PdfPages('%s_%s_%s_lines.pdf' %(s3d.inst, s3d.target, name))
    f = open('%s_%s_%s_lines.txt' %(s3d.inst, s3d.target, name), 'w')

    sigma = 2
    for line in lines:
        lp[line] = {}

        dx1, dx2 = 12, 12
        if line in ['oii7320', 'niia']:
            dx1 = 8
        if line in ['oii7331', 'siii6312']:
            dx2 = 8
        if line in ['oiiia', 'oiiib']:
            dx1, dx2 = 15, 15

        
        p1 = s3d.wltopix(RESTWL[line]*(1+s3d.z) - dx2)
        p2 = s3d.wltopix(RESTWL[line]*(1+s3d.z) + dx1)
        x = wl[p1:p2]
        y = spec[p1:p2]/div
        e = err[p1:p2]/div

        fixz, fixs = False, False
        if line in ['siii6312', 'ariii7135', 'ariii7751', 'oii7320',
             'oii7331', 'nii5755', 'hei5876']:
                 fixz, fixs = True, True
        
        if hasC == 0:
            gp = onedgaussfit(x, y, err=e,
                  params = [0, np.nanmax(y), RESTWL[line]*(1+s3d.z), sigma],
                  fixed=[True,False,fixz,fixs])
                  
        if hasC == 1:
            gp = onedgaussfit(x, y, err=e,
                  params = [np.nanmedian(y), np.nanmax(y), 
                            RESTWL[line]*(1+s3d.z), 2],
                  fixed=[False,False,fixz,False])          

        if line in ['ha', 'oiiib', 'oiiia']:
            sigma = gp[0][3]

        lineflg = gp[0][1]*gp[0][3]*(3.1416*2)**0.5
        linefleg = ((gp[2][1]/gp[0][1])**2 + (gp[2][3]/gp[0][3])**2)**0.5*lineflg
        lp[line]['FluxG'] = [lineflg, linefleg]

        f.write('\nLine: %s\n' %(line.capitalize()))
        f.write('Gauss Chi2/d.o.f: %.3f, %i\n' %(gp[3][0], gp[3][1]))
        f.write('Gauss amplitude = %.2f +/- %.2f 10^-17 erg/cm^2/s/AA\n' \
            %(gp[0][1], gp[2][1]))
        f.write('Gauss mean = %.2f +/- %.2f AA\n' %(gp[0][2], gp[2][2]))
        f.write('Gauss sigma = %.2f +/- %.2f AA\n' %(gp[0][3], gp[2][3]))
        f.write('Lineflux = %.2f +/- %.2f 10^-17 erg/cm^2/s\n' %(lineflg, linefleg))
        
        linefl, linefle = np.nansum(y[4:-4]-gp[0][0]), np.nansum(e[4:-4]**2)**0.5
        f.write('Lineflux sum = %.2f +/- %.2f 10^-17 erg/cm^2/s\n' \
            %(linefl, linefle))
        
        lp[line]['Flux'] = [linefl, linefle]
           
       
        if hasC == 1:
            f.write('Gauss base = %.2f +/- %.2f 10^-17 erg/cm^2/s/AA\n' \
                %(gp[0][0], gp[2][0]))
            ew = linefl/gp[0][0]
            ewe = ((linefle/linefl)**2 +  (gp[2][0]/gp[0][0])**2)**0.5 *ew
            f.write('EW = %.2f +/- %.2f \AA\n' %(ew, ewe))
          
    
        if plotLines==True:
            fig = plt.figure(figsize = (7,4))
            fig.subplots_adjust(bottom=0.12, top=0.99, left=0.12, right=0.99)
            ax = fig.add_subplot(1, 1, 1)
            ax.xaxis.set_major_formatter(plt.FormatStrFormatter(r'$%i$'))
            ax.plot(x, y, color = 'black', alpha = 1.0, # rasterized = raster,
                        drawstyle = 'steps-mid',  lw = 1.8, zorder = 1)
            ax.plot(gp[-1], gp[1], '-', lw=2)
            # Residuals            
#            fit = onedgaussian(x, gp[0][0], gp[0][1], gp[0][2], gp[0][3])
#            ax.errorbar(x, y-fit, yerr=e, capsize = 0, fmt='o', color='black')
            ax.set_xlim(x[0],x[-1])
#            ax.set_ylim(-0.03, max(0.03, max(gp[1])*1.1))
            ax.set_ylabel(r'$F_{\lambda}\,\rm{(10^{-17}\,erg\,s^{-1}\,cm^{-2}\, \AA^{-1})}$')
            ax.set_xlabel(r'$\rm{Observed\,wavelength\, (\AA)}$')

            text = 'Line:  %s\nWL:    %.2f AA\nFlux:  %.2f +/- %.2f' \
                %(line.capitalize(),gp[0][2], linefl, linefle)

            plt.figtext(0.65, 0.95, text, size=12, rotation=0.,
             ha="left", va="top",
             bbox=dict(boxstyle="round", color='lightgrey' ))
                
            pp.savefig(fig)
            plt.close(fig)       
 
    if plotLines==True:
        pp.close()     
    f.close()

#==============================================================================
#   EBV  
#==============================================================================
    

    if lp.has_key('ha') and lp.has_key('hb'):
        ebv, ebverro = mcebv(lp['ha']['Flux'], lp['hb']['Flux'])
        print '\n\n\tEBV = %.3f +/- %.3f mag' %(ebv, ebverro)
        for line in lp.keys():
            lp[line]['Flux'][0] *= s3d.ebvCor(line, ebv=ebv)
            lp[line]['Flux'][1] *= s3d.ebvCor(line, ebv=ebv)
        
    if lp.has_key('siia') and lp.has_key('siia') and lp.has_key('niib'):
        oh, oherr = mcohD16(lp['siia']['Flux'], lp['siib']['Flux'],
                            lp['ha']['Flux'], lp['niib']['Flux'])
        print '\t12+log(O/H) (D16) = %.3f +/- %.3f' %(oh, oherr)

    if lp.has_key('niia') and lp.has_key('hb') and lp.has_key('ha'):
        oho3n2, oherro3n2, ohn2, oherrn2 = mcohPP04(lp['oiiib']['Flux'], 
                    lp['hb']['Flux'], lp['niib']['Flux'], lp['ha']['Flux'])
        print '\t12+log(O/H) (PP04, O3N2) = %.3f +/- %.3f' %(oho3n2, oherro3n2)
        print '\t12+log(O/H) (PP04, N2) = %.3f +/- %.3f' %(ohn2, oherrn2)

    try:
        if lp.has_key('siii') and lp.has_key('siii6312'):
            tsiii, tsiiierr = mcTSIII(lp['siii']['Flux'], lp['siii6312']['Flux'])
            print '\tT_e(SIII) = %.0f +/- %.0f' %(tsiii, tsiiierr)
            a, b, c = -0.546, 2.645, -1.276-tsiii/1E4
            toiii = ((-b + (b**2 - 4*a*c)**0.5) / (2*a))*1E4
            toii = (-0.744 + toiii/1E4*(2.338 - 0.610*toiii/1E4)) * 1E4
            print '\tT_e(OII) = %.0f' %(toii)
            print '\tT_e(OIII) = %.0f' %(toiii)
        
        if lp.has_key('oiiib') and lp.has_key('oii7320') and lp.has_key('oii7331'):
            ohtoiii, ohtoiiie = mcOHOIII(lp['oiiib']['Flux'], lp['oii7320']['Flux'],
                               lp['oii7331']['Flux'], lp['hb']['Flux'],
                               toiii, toiii*tsiiierr/tsiii)
            print '\t12+log(O/H) T(SIII) = %.3f +/- %.3f' %(ohtoiii, ohtoiiie)
    
        if lp.has_key('siii6312') and lp.has_key('siia') and lp.has_key('siib'):
            shsoiii, shsoiiie = mcSHSIII(lp['siii6312']['Flux'], lp['siia']['Flux'],
                               lp['siib']['Flux'], lp['hb']['Flux'],
                               tsiii, tsiiierr, toiii)
            print '\t12+log(S/H) T(SIII) = %.3f +/- %.3f' %(shsoiii, shsoiiie)

    except ValueError:
       pass


    if lp.has_key('siia') and lp.has_key('siib'):
        n, ne = mcDens(lp['siia']['Flux'], lp['siib']['Flux'])
        print '\tlog Electron density = %.2f +/- %.2f cm^-3' %(n, ne)

    if lp.has_key('ha') and lp.has_key('hb'):
        sfr, sfre = mcSFR(lp['ha']['Flux'], lp['hb']['Flux'], s3d)
        print '\tSFR = %.3e +/- %.3e Msun/yr' %(sfr, sfre)
    
    return {'EBV':[ebv, ebverro], 'ne':[n, ne], 'SFR':[sfr, sfre],
    '12+log(O/H)(D16)': [oh, oherr], 
    '12+log(O/H)(O3N2)':[oho3n2, oherro3n2], '12+log(O/H)(N2)': [ohn2, oherrn2], 
    'T_e(SIII)': [tsiii, tsiiierr], 
    'T_e(OII)':[toii], 'T_e(OIII)':[toiii],
    '12+log(O/H)T(SIII)':[ohtoiii, ohtoiiie], '12+log(S/H)T(SIII)':[shsoiii, shsoiiie]}

def metGrad(s3d, meth='S2', snlim = 1, nbin = 2, reff=None, incl=90, posang=0,
            ylim1=7.95, ylim2=8.6, xlim1=0, xlim2=3.45, r25=None,
            sC=0, xcen=None, ycen=None, nsamp=1000):
                
    """ Obtains and plots the metallicity gradient of a galaxy, using a given
    strong-line diagnostic ratio.

    Parameters
    ----------
        s3d : Spectrum3d class
            Initial spectrum class with the data and error
        meth : str
            Acronym for the strong line diagnostic that will be used. Passed to 
            getOH.
        snlim : float
            Signal-to-noise limit for using spaxels in plot
        nbin : integer
            Number of bins in the resulting plot
        reff : float
            Effective radius in arcsec (used for scaling x-axis)
        incl : float
            Inclination of the galaxy, used for deprojecting the angular scales
        ylim1 : float
            Minimum limit of 12+log(O/H) plotted on y-axis
        ylim2 : float
            Maximum limit of 12+log(O/H) plotted on y-axis
        xlim1 : float
            Minimum limit of distance (kpc) plotted on x-axis
        xlim2 : float
            Maximum limit of distance (kpc) plotted on x-axis
    Returns
    -------
        Nothing, but plots the metallicity gradient in a pdf file
    """    
#    posang -= 90
    if xcen == None or ycen == None:
        x, y = getGalcen(s3d, sC=sC, mask=True)
    else:
        logger.info('Using galaxy center at %.2f, %.2f' %(xcen, ycen))
        x, y = xcen, ycen
    ohmap, ohsnmap = getOH(s3d, meth=meth, sC=sC)
    yindx, xindx = np.indices(ohmap.shape)[0], np.indices(ohmap.shape)[1]
    distmap = ((yindx - y)**2 + (xindx - x)**2) ** 0.5
    angmap = np.arctan((x-xindx) / (yindx-y)) * 180./np.pi 
    angmap[angmap < 0] = angmap[angmap < 0] + 180
    # Distcor is distance correction based on angle to posang
    # 1 means no correction
    maxcor = (1. / np.sin(incl/180. * np.pi)) - 1
    distcor = 1 + np.abs((np.sin((angmap - posang) * np.pi / 180.)) * maxcor)
    distmap *= distcor

    if nbin > 0:
        # Median filter by nbin in each of the directions
        ohmap = sp.ndimage.filters.median_filter(ohmap, nbin)
        # Downsample
        ohmap = ohmap[::nbin, ::nbin]
        distmap = distmap[::nbin, ::nbin]
        # Each bin has now a higher S/N by (nbin*nbin)**0.5
        ohsnmap = ohsnmap[::nbin, ::nbin]*nbin
        
    sel = (ohsnmap > snlim) * (~np.isnan(ohmap))

    if reff != None:
        distmap = (distmap[sel].flatten() * s3d.pixsky) / reff
    elif r25 != None:
        distmap = (distmap[sel].flatten() * s3d.pixsky) / r25
    else:
        distmap = (distmap[sel].flatten() * s3d.pixsky) * s3d.AngD 

    ohmap = ohmap[sel].flatten()

    indizes = distmap.argsort()
    distmap = distmap[indizes]
    ohmap = ohmap[indizes]

    sel = distmap < 1
    distmap = distmap[sel]
    ohmap = ohmap[sel]
    # Bootstrapping fits
    indizes1 = bootstrap(distmap, n_samples=nsamp)
    distmaps = distmap[indizes1]
    ohmaps = ohmap[indizes1]
    distbins = np.arange(-0.1, 2, 0.1)    
    
    slope, inter, metgrads = [],[],[]
    for i in range(nsamp):
        res = linfit(distmaps[i], ohmaps[i], params = [8.6, -0.3])
        slope.append(res[0][1])
        inter.append(res[0][0])
        metgrads.append(res[0][0] + distbins*res[0][1])

    metgrads = np.array(metgrads).transpose()
    minss1, bestss1, maxss1 = 3*[np.array([])]
    for i in range(len(distbins)):
        bestu, minu, maxu = geterrs(metgrads[i], sigma=1.0)
        minss1 = np.append(minss1, bestu-minu)
        bestss1 = np.append(bestss1, bestu)
        maxss1= np.append(maxss1, bestu+minu)
    if r25 != None:
        logger.info('Intercept %.2f +/- %.2f' %(np.nanmedian(inter), np.std(inter)))
        logger.info('Slope %.2f +/- %.2f' %(np.nanmedian(slope), np.std(slope)))
        logger.info('Slope %.3f +/- %.3f' \
            %(np.nanmedian(slope) / r25 / s3d.AngD, np.std(slope) / r25 / s3d.AngD))

#        logger.info('Fit parameters %.3f' %( res[0][1] / r25 / s3d.AngD))

    std, meandist, ddist = [], [], 0.03
    for i in distbins:
        sel = (distmap < i+ddist) * (distmap > i-ddist)
        std.append(np.std(ohmap[sel]))
        meandist.append(np.nanmedian(distmap[sel]))
#    dist, oh = binspec(distmap, ohmap, wl=nbin)

    fig2 = plt.figure(figsize = (7,4.5))
    ax = fig2.add_subplot(1, 1, 1)
    ax.plot(meandist, std, '-', color='black', lw=3, zorder=1)
    ax.set_xlim(xlim1, xlim2)
    fig2.savefig('%s_%s_gradstd.pdf' %(s3d.inst, s3d.target))
   
    fig = plt.figure(figsize = (7,4.5))
    fig.subplots_adjust(bottom=0.12, top=0.88, left=0.13, right=0.99)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(distmap, ohmap, 'o', color ='firebrick', ms=4)
    ax.set_ylim(ylim1, ylim2)
    ax.set_xlim(xlim1, xlim2)
#    ax.fill_between(distbins, minss1, maxss1, color='grey', alpha=0.3)
    ax.plot(distbins, bestss1, '-', color='black', lw=2, zorder=1)
    ax.plot(distbins[distbins<0.25], 8.43 - 0.86 * distbins[distbins<0.25],
            '--', color='black', lw=3, zorder=1)

    if reff != None:
        ax.set_xlabel(r'$\rm{Deprojected\,distance\,(R/R_e)}$', fontsize=16)
    elif r25 != None:
        ax.set_xlabel(r'$\rm{Deprojected\,distance\,(R/R_{25})}$', fontsize=16)
    else:
        ax.set_xlabel(r'$\rm{Deprojected\, distance\,from\,center\,(kpc)}$', fontsize=16)
    ax2=ax.figure.add_axes(ax.get_position(), frameon = False, sharey=ax)
    ax2.xaxis.tick_top()
    ax.xaxis.tick_bottom()
    ax2.xaxis.set_label_position("top")
    if reff != None:
        ax2.set_xlim(xlim1, xlim2 * reff * s3d.AngD)
        ax2.set_xlabel(r'$\rm{Deprojected\,distance\,(kpc)}$', fontsize=16)
    elif r25 != None:
        ax2.set_xlim(xlim1, xlim2 * r25 * s3d.AngD)
        ax2.set_xlabel(r'$\rm{Deprojected\,distance\,(kpc)}$', fontsize=16)

    else:
        ax2.set_xlim(xlim1, xlim2 / s3d.AngD)
        ax2.set_xlabel(r'$\rm{Deprojected\,distance\,(arcsec)}$', fontsize=16)

    ax.set_ylabel(r'$\rm{12+log(O/H)\,(D16)}$', fontsize=18)
    fig.savefig('%s_%s_metgrad.pdf' %(s3d.inst, s3d.target))
    plt.close(fig)




def voronoi_bin(plane, planee=None, planesn=None,
                binplane=None, binplanee=None, binplanesn=None,
                targetsn=10):
                    
    """Bins a 2d-map, based on the voronoi tessalation of a (potentially
    different) map.

    Parameters
    ----------
        plane : np.array (2-dimensional)
            This is the data which we want to voronoi bin
        planee : np.array (2-dimensional)
            For the tesselation, we require either the error or the sn. This is 
            the error.
        planesn : np.array (2-dimensional)
            For the tesselation, we require either the error or the sn. This is 
            the sn.
        binplane : np.array (2-dimensional)
            Alternatively, a different map can be provided where the binning is 
            derived from .
        binplanee : np.array (2-dimensional)
            For the tesselation, we require either the error or the sn. This is 
            the error of the map that defines the binning.
        binplanesn : np.array (2-dimensional)
            For the tesselation, we require either the error or the sn. This is 
            the sn of the map that defines the binning.
    Returns
    -------
        binnedplane : np.array (2-dimensional)
            Binned plane derived from tesselation of plane
        binnedplane : np.array  (2-dimensional)
            Signal to noise ration from binned plane 
    """

    binnedplane = np.copy(plane)
    if planee != None:
        binnedplanee = np.copy(planee)
    elif planesn != None:
        binnedplanee = np.copy(planesn)
        
    if binplane == None:
        binplane, binplanee, binplanesn = plane, planee, planesn

    voronoi_idx = _voronoi_run(binplane, error=binplanee, snmap=binplanesn,
                              targetsn=targetsn)
    voronoi_idx = voronoi_idx[np.argsort(voronoi_idx[:,2])]

    for voro_bin in range(np.max(voronoi_idx[:,2])):
        sel = (voronoi_idx[:,2] == voro_bin)
        v_pixs = voronoi_idx[sel]
        data_bin, data_err = np.array([]), np.array([])
        for v_pix in v_pixs:
            data_bin = np.append(data_bin, plane[v_pix[1], v_pix[0]])
            if planee != None:
                data_err = np.append(data_err, planee[v_pix[1], v_pix[0]])
            else:
                data_err = np.append(data_err, plane[v_pix[1], v_pix[0]]/\
                                               planesn[v_pix[1], v_pix[0]])
        
        bin_val = np.nansum(data_bin * 1./data_err**2)/np.nansum(1./data_err**2)
        bin_err = 1. / np.nansum(1./data_err**2)**0.5
        for v_pix in v_pixs:
            binnedplane[v_pix[1], v_pix[0]] = bin_val
            binnedplanee[v_pix[1], v_pix[0]] = bin_err
        binnedplanesn = binnedplane/binnedplanee
    return binnedplane, binnedplanesn



def _voronoi_run(plane, error=None, snmap=None, targetsn=10):
    
    """Convenience function that runs the voronoi tesselation code in module
    voronoi
    
    Parameters
    ----------
        plane : np.array (2-dimensional)
            This is the data which we want to voronoi bin
        error : np.array (2-dimensional)
            For the tesselation, we require either the error or the sn. This is 
            the error.
        snmap : np.array (2-dimensional)
            For the tesselation, we require either the error or the sn. This is 
            the sn.
        targetsn : float
            Target Signal-to-noise ratio in the Voronoi bins

    Returns
    -------
        np.column_stack([indx, indy, binNum]) : np.array
            This is a stack where each pixel indizes (indx, indy) are listed 
            with the corresponding bin number
    """

    logger.info('Voronoi tesselation of input map with S/N = %i' %targetsn)
    indx, indy, signal, noise = _voronoi_prep(plane, error, snmap)
    binNum, xNode, yNode, xBar, yBar, sn, nPixels, scale = \
        voronoi(indx, indy, signal, noise, targetsn, logger)
    return np.column_stack([indx, indy, binNum])



def _voronoi_prep(plane, error = None, sn = None):
    
    """ Convenience function that takes a 2d plane and error as input and prepares four vectors to be run
    with voronoi binning

    Parameters
    ----------
        plane : np.array (2-dimensional)
            This is the data which we want to voronoi bin
        error : np.array (2-dimensional)
            For the tesselation, we require either the error or the sn. This is 
            the error.
        snmap : np.array (2-dimensional)
            For the tesselation, we require either the error or the sn. This is 
            the sn.

    Returns
    -------
        indx, indy : np.array 
            x- and y-indices for the individual spaxels
        signal : np.array
            data at spaxel x,y
        noise : np.array
            noise at spaxel x,y
    """
    indx = np.indices(plane.shape)[1].flatten()
    indy = np.indices(plane.shape)[0].flatten()
    signal = plane.flatten()
    if error != None:
        noise = error.flatten()
    elif sn != None:
        noise = (plane/sn).flatten()
    else:
        raise SystemExit
    return indx, indy, signal, noise
