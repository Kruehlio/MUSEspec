# -*- coding: utf-8 -*-

""" Spectrum class for running starlight on spectra. Particularly for 
 MUSE cubes
 """

import matplotlib
matplotlib.use('Agg')

import os
import numpy as np
import scipy as sp

import shutil
import time
import platform
import matplotlib.pyplot as plt
import logging

from .io import asciiout, cubeout

logfmt = '%(levelname)s [%(asctime)s]: %(message)s'
datefmt= '%Y-%m-%d %H:%M:%S'
formatter = logging.Formatter(fmt=logfmt,datefmt=datefmt)
logger = logging.getLogger('__main__')
logging.root.setLevel(logging.DEBUG)
ch = logging.StreamHandler() #console handler
ch.setFormatter(formatter)
logger.handlers = []
logger.addHandler(ch)


SL_BASE = os.path.join(os.path.dirname(__file__), "etc/Base.BC03.M")
SL_CONFIG = os.path.join(os.path.dirname(__file__), "etc/MUSE_SLv01.config")
SL_MASK = os.path.join(os.path.dirname(__file__), "etc/Masks.EmLines.SDSS.gm")
SL_BASES = os.path.join(os.path.dirname(__file__), "etc/bases")
if platform.platform().startswith('Linux'):
    SL_EXE = os.path.join(os.path.dirname(__file__), "etc/starlight")
else:
    SL_EXE = os.path.join(os.path.dirname(__file__), "etc/starlight_mac")



class StarLight:
    """ StarLight class for fitting """

    def __init__(self, filen, verbose=0, minwl=3500, maxwl=9400,
                 run=1):
        self.specfile = filen
        self.minwl=minwl
        self.maxwl=maxwl
        root, ext = os.path.splitext(filen)
        self.output = root+'_sl_out'+ext
        self.sllog = root+'_sl_log'+ext
        self.seed = np.random.randint(1E6, 9E6)
        self.cwd = os.getcwd()
        self.inst = 'MUSE'
        shutil.copy(SL_BASE, self.cwd)
        shutil.copy(SL_CONFIG, self.cwd)
        if not os.path.isdir(os.path.join(self.cwd, 'bases')):
            shutil.copytree(SL_BASES, os.path.join(self.cwd, 'bases'))

        if not os.path.isfile(SL_EXE):
            print 'ERROR: STARLIGHT executable not found'
            raise SystemExit
            
        if run == 1:
            self.makeGrid()
            self.runGrid()


    def makeGrid(self, name='muse_grid.in'):

        headkey = ['[Number of fits to run]',
               '[base_dir]', '[obs_dir]', '[mask_dir]', '[out_dir]',
               '[seed]', '[llow_SN]', '[lupp_SN]', '[Olsyn_ini]',
               '[Olsyn_fin]', '[Odlsyn]', '[fscale_chi2]', '[FIT/FXK]', 
               '[IsErrSpecAvailable]', '[IsFlagSpecAvailable]']
        speckey = ['spectrum', 'config', 'bases', 'masks', 'red', 'v0_start',
                  'vd_start', 'output']
       
        header = {'[Number of fits to run]': '1',
               '[base_dir]': self.cwd+'/bases/',
               '[obs_dir]' :self.cwd+'/', 
               '[mask_dir]' : os.path.split(SL_MASK)[0]+'/', 
               '[out_dir]': self.cwd+'/',
               '[seed]': self.seed, 
               '[llow_SN]': 5200, 
               '[lupp_SN]': 5400, 
               '[Olsyn_ini]': self.minwl,
               '[Olsyn_fin]': self.maxwl, 
               '[Odlsyn]':1.0, 
               '[fscale_chi2]':1.0, 
               '[FIT/FXK]': 'FIT',
               '[IsErrSpecAvailable]':'1', 
               '[IsFlagSpecAvailable]':'1'}

        specline = {'spectrum': self.specfile, 
            'config': os.path.split(SL_CONFIG)[-1], 
            'bases': os.path.split(SL_BASE)[-1], 
            'masks': os.path.split(SL_MASK)[-1], 
            'red' : 'CAL', 
            'v0_start': 0,
            'vd_start': 150, 
            'output': self.output}
            
        f = open(name, 'w')
        for head in headkey:
            f.write('%s  %s\n' %(header[head], head))
        for spec in speckey:
            f.write('%s   ' %(specline[spec]))
        f.write('\n')
        self.grid = name
        
    def runGrid(self, cleanup=True):
        t1 = time.time()
        slarg = [SL_EXE, '<', self.grid, '>', self.sllog]
        os.system(' '.join(slarg))
        # Cleanup
        if cleanup == True:
            shutil.rmtree('bases')
            os.remove(os.path.join(self.cwd, os.path.split(SL_BASE)[-1]))
            os.remove(os.path.join(self.cwd, os.path.split(SL_CONFIG)[-1]))
#            os.remove(self.grid)
        return time.time()-t1

       
    def modOut(self, plot=1, minwl=4750, maxwl=5150):
        starwl, starfit = np.array([]), np.array([])
        datawl, data, gas, stars = 4*[np.array([])]
        success, run, norm = 0, 0, 1

        try:
            f = open(self.output)
            output = f.readlines()
            f.close()
            os.remove(self.sllog)
            os.remove(self.output)
            run = 1
        except IOError:
            pass
        
        if run == 1:
            for out in output:
              outsplit =   out.split()
              if outsplit[1:] == ['[fobs_norm', '(in', 'input', 'units)]']:
                  norm = float(outsplit[0])
                  success = 1
              if outsplit[1:] == ['Run', 'aborted:(']:
                  break
              if len(outsplit) == 4:
                try:  
                  outsplit = [float(a) for a in outsplit]  
                  if float(outsplit[0]) >= self.minwl:
                      starfit = np.append(starfit, outsplit[2])
                      starwl = np.append(starwl, outsplit[0])
                      if outsplit[3] != -2:
                          data = np.append(data, outsplit[1])
                          gas = np.append(gas, outsplit[1]-outsplit[2] )
                          stars = np.append(stars, outsplit[2])
                          datawl = np.append(datawl, outsplit[0])                    
                except ValueError:
                    pass
        
            if plot == 1:
                sel1 = (datawl > minwl) * (datawl < maxwl)
                sel2 = (datawl > 6500) * (datawl < 6750)
                
                fig1 = plt.figure(figsize = (6,8.4))
                fig1.subplots_adjust(bottom=0.15, top=0.97, left=0.13, right=0.96)
                ax1 = fig1.add_subplot(2, 1, 1)
                ax2 = fig1.add_subplot(2, 1, 2)
                for ax in [ax1, ax2]:
                    ax.plot(datawl, 0*datawl, '--', color ='grey')
                    ax.plot(datawl, gas, '-', color ='black')
                    ax.plot(datawl, data, '-', color ='firebrick', lw=2)
                    ax.plot(starwl, starfit, '-', color ='green')
                    ax.set_ylabel(r'$F_{\lambda}\,\rm{(10^{-17}\,erg\,s^{-1}\,cm^{-2}\, \AA^{-1})}$',
                               fontsize=18)
                ax2.set_xlabel(r'Restframe wavelength $(\AA)$', fontsize=18)
                ax1.set_xlim(minwl, maxwl)
                ax2.set_xlim(6500, 6750)
                ax1.set_ylim(np.min(gas[sel1]), np.max(data[sel1])*1.05)
                ax2.set_ylim(np.min(gas[sel2]), np.max(data[sel2])*1.05)
                fig1.savefig('%s_starlight.pdf' %(self.inst))
                plt.close(fig1)
                
        return datawl, data, stars, norm, success
        
        
        
def runStar(s3d, ascii, plot=1, verbose=1):
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
    zerospec = np.zeros(s3d.wave.shape)

    if success == 1:
        if verbose == 1:
            logger.info('Running starlight took %.2f s' %(time.time() - t1))
        s = sp.interpolate.InterpolatedUnivariateSpline(datawl*(1+s3d.z), 
                                                    data*1E3*norm/(1+s3d.z))
        t = sp.interpolate.InterpolatedUnivariateSpline(datawl*(1+s3d.z), 
                                                    stars*1E3*norm/(1+s3d.z))
        return s(s3d.wave), t(s3d.wave), success
    else:
        if verbose ==1:
            logger.info('Starlight failed in %.2f s' %(time.time() - t1))
        return zerospec, zerospec, success
        

def subStars(s3d, x, y, size=0, verbose=1):
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
    
    wl, spec, err = s3d.extrSpec(x=x, y=y, size=size, verbose=0)
    ascii = asciiout(s3d=s3d, wl=wl, spec=spec, err=err, 
                          name='%s_%s_%s' %(x, y, size))
                      
    data, stars, success = runStar(s3d, ascii, verbose=0)
    os.remove(ascii)
    miny, maxy = max(0, y-size), min(s3d.leny-1, y+size+1)
    minx, maxx = max(0, x-size), min(s3d.lenx-1, x+size+1)
    
    xindizes=np.arange(minx, maxx, 1)
    yindizes=np.arange(miny, maxy, 1)
    zerospec = np.zeros(s3d.wave.shape)
    if success == 1:
#            rs = data/spec
#            logger.info('Resampling accuracy %.3f +/- %.3f' \
#                %(np.nanmedian(rs), np.nanstd(rs[1:-1])))

        for xindx in xindizes:
            for yindx in yindizes:
                wl, spec, err = s3d.extrSpec(x=xindx, y=yindx, verbose=verbose)
                
                # Renormalize to actual spectrum
                substars = np.nanmedian(spec/data)*stars
                
                # Overwrite starcube with fitted values
                s3d.starcube[:, yindx, xindx] = substars
    else:
        for xindx in xindizes:
            for yindx in yindizes:
                # Np sucess
                s3d.starcube[:, yindx, xindx] = zerospec
    return
    
    
def suballStars(s3d, dx=2, nc=None):
    """ 
    Convinience function to subtract starlight fits on the full cube
    """
    
    logger.info("Starting starlight on full cube with %i cores" %s3d.ncores)
    logger.info("This might take a bit")
    t1 = time.time()
    xindizes = np.arange(dx, s3d.lenx, 2*dx+1)
    yindizes = np.arange(dx, s3d.leny, 2*dx+1)

    for xindx in xindizes:
        for yindx in yindizes:
            subStars(s3d, xindx, yindx, dx, verbose=0)
            
    cubeout(s3d, s3d.starcube, name='star')
    cubeout(s3d, s3d.data-s3d.starcube, name='gas')
    logger.info("This took %.2f h" %((time.time()-t1)/3600.))
