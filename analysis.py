# -*- coding: utf-8 -*-

""" 
Analysis on MUSE cubes
"""

import numpy as np
import matplotlib.pyplot as plt

from .extract import getGalcen
from .maps import getOH, getEW
from .astro import binspec

def metGrad(s3d, meth='S2', ewlim = 5, nbin = 8, reff=None, incl=90):
    
    x, y = getGalcen(s3d)
    ohmap = getOH(s3d, meth=meth)
    distmap = ((np.indices(ohmap.shape)[0] - y)**2 \
        + (np.indices(ohmap.shape)[1] - x)**2) ** 0.5
    hamap = getEW(s3d, 'ha')
    oiiimap = getEW(s3d, 'oiii')

    sel = (hamap > ewlim) * (oiiimap > ewlim)
    
    if reff != None:
        distmap = (distmap[sel].flatten() * s3d.pixsky) / reff \
            / np.sin(incl/180. * np.pi)
    else:
        distmap = (distmap[sel].flatten() * s3d.pixsky) * s3d.AngD \
            / np.sin(incl/180. * np.pi)

    ohmap = ohmap[sel].flatten()
    
    indizes = distmap.argsort()
    distmap = distmap[indizes]
    ohmap = ohmap[indizes]

    dist, oh = binspec(distmap, ohmap, wl=nbin)
    
    fig = plt.figure(figsize = (7,5.5))
    fig.subplots_adjust(bottom=0.12, top=0.88, left=0.13, right=0.99)   
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(dist, oh, 'o', color ='firebrick', ms=4)
    ax.set_ylim(7.95, 8.49)
    xlim2 = 3
    ax.set_xlim(0, xlim2)
    if reff != None:
        ax.set_xlabel(r'$\rm{Deprojected\,distance\,(R/R_e)}$', fontsize=16)
    else:
        ax.set_xlabel(r'$\rm{Distance\,from\,center\,(kpc)}$', fontsize=16)
    ax2=ax.figure.add_axes(ax.get_position(), frameon = False, sharey=ax)
    ax2.xaxis.tick_top()
    ax.xaxis.tick_bottom()
    ax2.xaxis.set_label_position("top")
    if reff != None:
        ax2.set_xlim(0, xlim2 * reff * s3d.AngD)
        ax2.set_xlabel(r'$\rm{Deprojected\,distance\,(kpc)}$', fontsize=16)
    else:
        ax2.set_xlim(0, xlim2 / s3d.AngD)
        ax2.set_xlabel(r'$\rm{Deprojected\,distance\,(arcsec)}$', fontsize=16)

    ax.set_ylabel(r'$\rm{12+log(O/H)\,(D16)}$', fontsize=18)
    fig.savefig('%s_%s_metgrad.pdf' %(s3d.inst, s3d.target))
    plt.close(fig)
    
    
#def getEffrad(s3d, xcen, ycen, mask = True, line='ha', wlim1=4800, wlim2=5200):
#    """ 
#    Gets the effective Radius of the galaxy
#    """
#    hamap = getEW(s3d, line)
#    mask = np.zeros(hamap.shape)
#    mask[20:-20, 20:-20] = 1
#    if mask == True:
#        mask[hamap < 10] == 0
#
#    wl, spec, err = s3d.extrSpec(exmask=mask)
#    
#            
#    ycen = np.nansum(ysum[ysum > 0] * np.arange(s3d.leny)[ysum > 0])\
#        /np.nansum(ysum[ysum > 0])
#    xcen = np.nansum(xsum[xsum > 0] * np.arange(s3d.lenx)[xsum > 0])\
#        /np.nansum(xsum[xsum > 0])
##    print xcen, ycen, bpixx, bpixy
#    return xcen, ycen    