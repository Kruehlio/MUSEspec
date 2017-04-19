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

from ..io.io import asciiout, cubeout

logfmt = '%(levelname)s [%(asctime)s]: %(message)s'
datefmt= '%Y-%m-%d %H:%M:%S'
formatter = logging.Formatter(fmt=logfmt,datefmt=datefmt)
logger = logging.getLogger('__main__')
logging.root.setLevel(logging.DEBUG)
ch = logging.StreamHandler() #console handler
ch.setFormatter(formatter)
logger.handlers = []
logger.addHandler(ch)


SL_BASE_ALL = os.path.join(os.path.dirname(__file__), "../etc/Base.BC03.S")
SL_BASE_FEW = os.path.join(os.path.dirname(__file__), "../etc/Base.BC03.N")
SL_BASE_BB = os.path.join(os.path.dirname(__file__), "../etc/Base.BC03.15lh")

SL_CONFIG = os.path.join(os.path.dirname(__file__), "../etc/MUSE_SLv01.config")
SL_MASK = os.path.join(os.path.dirname(__file__), "../etc/Masks.EmLines.SDSS.gm")
SL_BASES = os.path.join(os.path.dirname(__file__), "../etc/bases")
if platform.platform().startswith('Linux'):
    SL_EXE = os.path.join(os.path.dirname(__file__), "../etc/starlight")
else:
    SL_EXE = os.path.join(os.path.dirname(__file__), "../etc/starlight_mac")



class StarLight:
    """ StarLight class for fitting """

    def __init__(self, filen, verbose=0, minwl=None, maxwl=None,
                 run=1, bases='FEW', inst='MUSE', red='CAL'):
        self.specfile = filen
        if minwl == None:
            self.minwl=3330
        else:
            self.minwl=minwl
        
        if maxwl == None:
            self.maxwl=9400
        else:
            self.maxwl=maxwl
            
        self.cwd = os.getcwd()
        root, ext = os.path.splitext(filen)

        self.output = os.path.join(root+'_sl_out'+ext)
        self.sllog = root+'_sl_log'+ext
        self.seed = np.random.randint(1E6, 9E6)
        self.inst = inst
        self.red = red

        basewdir = os.path.join(self.cwd, 'bases')
        if not os.path.isdir(basewdir):
            os.makedirs(basewdir)
        
        if bases == 'FEW':
            shutil.copy(SL_BASE_FEW, self.cwd)
            self.bases = SL_BASE_FEW
        elif bases == 'ALL':
            shutil.copy(SL_BASE_ALL, self.cwd)
            self.bases = SL_BASE_ALL
        elif bases == 'BB':
            shutil.copy(SL_BASE_BB, self.cwd)
            self.bases = SL_BASE_BB
            
        shutil.copy(SL_CONFIG, self.cwd)
        f = open(self.bases)
        basescps = [g for g in f.readlines() if not g.startswith('#')]
        f.close()
        for basescp in basescps:
            baseraw = os.path.join(SL_BASES, basescp.split()[0])
            if os.path.isfile(baseraw):
                shutil.copy(baseraw, basewdir)



        if not os.path.isfile(SL_EXE):
            print ('ERROR: STARLIGHT executable not found')
            raise SystemExit
            
        if run == 1:
            self._makeGrid()
            self._runGrid()


    def _makeGrid(self, name='muse_grid.in'):

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
            'bases': os.path.split(self.bases)[-1], 
            'masks': os.path.split(SL_MASK)[-1], 
            'red' : self.red, 
            'v0_start': 0,
            'vd_start': 50, 
            'output': self.output}
            
        f = open(name, 'w')
        for head in headkey:
            f.write('%s  %s\n' %(header[head], head))
        for spec in speckey:
            f.write('%s   ' %(specline[spec]))
        f.write('\n')
        self.grid = name
        
    def _runGrid(self, cleanup=True):
        t1 = time.time()
        slarg = [SL_EXE, '<', self.grid, '>', self.sllog]
        os.system(' '.join(slarg))
        # Cleanup
        if cleanup == True:
            shutil.rmtree('bases')
            os.remove(os.path.join(self.cwd, os.path.split(self.bases)[-1]))
            os.remove(os.path.join(self.cwd, os.path.split(SL_CONFIG)[-1]))
        return time.time()-t1

       
    def modOut(self, plot=0, minwl=3860, maxwl=4470,
               rm=True):
        
        starwl, starfit = np.array([]), np.array([])
        datawl, data, gas, stars = 4*[np.array([])]
        success, run, norm, v0, vd, av = 0, 0, 1, -1, -1, -1

        try:
            f = open(self.output)
            output = f.readlines()
            f.close()
            if rm == True:
                os.remove(self.sllog)
                slpath = os.path.join(self.cwd, 'sl_fits')
                if not os.path.isdir(slpath):
                    os.makedirs(slpath)
                slout = os.path.join(slpath, self.output)
                if os.path.isfile(slout):
                    os.remove(slout)
                shutil.move(self.output, os.path.join(self.cwd, 'sl_fits'))
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
                
              if len(outsplit) == 3:
                 if outsplit[1] == '[v0_min':
                    v0 = float(outsplit[0])
                 if outsplit[1] == '[vd_min':
                    vd = float(outsplit[0])       
                 if outsplit[1] == '[AV_min':
                    av = float(outsplit[0])       

            if plot == 1:
                sel0 = (datawl > minwl) * (datawl < maxwl)
                sel1 = (datawl > 3860) * (datawl < 4630)
                sel2 = (datawl > 4730) * (datawl < 5230)
                sel3 = (datawl > 6420) * (datawl < 7020)

               
                fig1 = plt.figure(figsize = (5,8.4))
                fig1.subplots_adjust(bottom=0.10, top=0.99, left=0.15, right=0.98)
                ax1 = fig1.add_subplot(3, 1, 1)
                ax2 = fig1.add_subplot(3, 1, 2)
                ax3 = fig1.add_subplot(3, 1, 3)
                for ax in [ax1, ax2, ax3]:
                    ax.plot(datawl, 0*datawl, '--', color ='grey')
                    ax.plot(datawl, norm*gas, '-', color ='black')
                    ax.plot(datawl, norm*data, '-', color ='firebrick', lw=2)
                    ax.plot(starwl, norm*starfit, '-', color ='green')
                    ax.set_ylabel(r'$F_{\lambda}\,\rm{(10^{-17}\,erg\,s^{-1}\,cm^{-2}\, \AA^{-1})}$',
                               fontsize=16)
                
                ax3.set_xlabel(r'Restframe wavelength $(\AA)$', fontsize=16)
                ax1.set_xlim(3860, 4630)
                ax3.set_xlim(6420, 6780)
                ax2.set_xlim(4750, 5230)
                
                ax1.set_ylim(norm*np.min(gas[sel1]), norm*np.max(data[sel1])*1.05)
                ax2.set_ylim(norm*np.min(gas[sel2]), norm*np.max(data[sel2])*1.05)
                ax3.set_ylim(norm*np.min(gas[sel3]), norm*np.max(data[sel3])*1.05)

                fig1.savefig('%s_starlight.pdf' %(self.inst))
                plt.close(fig1)

                fig2 = plt.figure(figsize = (8,5))
                fig2.subplots_adjust(bottom=0.14, top=0.99, left=0.12, right=0.98)
                ax = fig2.add_subplot(1, 1, 1)
                ax.plot(datawl, 0*datawl, '--', color ='grey')
                ax.plot(datawl, norm*gas, '-', color ='black')
                ax.plot(datawl, norm*data, '-', color ='firebrick', lw=2)
                ax.plot(starwl, norm*starfit, '-', color ='green')
                ax.set_ylabel(r'$F_{\lambda}\,\rm{(10^{-17}\,erg\,s^{-1}\,cm^{-2}\, \AA^{-1})}$',
                               fontsize=16)
                
                ax.set_xlabel(r'Restframe wavelength $(\AA)$', fontsize=16)
                ax.set_xlim(np.min(datawl[sel0]), np.max(datawl[sel0]))
                ax.set_ylim(norm*np.min(gas[sel0]), norm*np.max(data[sel0])*1.05)

                fig2.savefig('%s_starlight_all.pdf' %(self.inst))
                plt.close(fig2)
                
        return datawl, data, stars, norm, success, v0, vd, av
        
        
        
def runStar(s3d, ascii, starres = None, minwl=None, maxwl=None,
            plot=0, verbose=1, rm=True, bases='ALL'):
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
        
    if starres == None:
        starres =  '%s_star_res.txt' %(s3d.inst) 
        if os.path.isfile(starres):
            os.remove(starres)    
            
    t1 = time.time()
    sl = StarLight(filen=ascii, bases=bases, minwl=minwl, maxwl=maxwl)
    datawl, data, stars, norm, success, v0, vd, av =\
        sl.modOut(plot=plot, rm=rm, minwl=minwl, maxwl=maxwl)
    zerospec = np.zeros(s3d.wave.shape)

    if success == 1:
        if verbose == 1:
            logger.info('Running starlight took %.2f s' %(time.time() - t1))
        s = sp.interpolate.InterpolatedUnivariateSpline(datawl*(1+s3d.z), 
                                                    data*1E3*norm/(1+s3d.z))
        t = sp.interpolate.InterpolatedUnivariateSpline(datawl*(1+s3d.z), 
                                                    stars*1E3*norm/(1+s3d.z))
        return s(s3d.wave), t(s3d.wave), success, v0, vd, av
    
    else:
        if verbose ==1:
            logger.info('Starlight failed in %.2f s' %(time.time() - t1))
        return zerospec, zerospec, success, v0, vd, av
        

def subStars(s3d, x, y, size=0, verbose=1, 
             inst='MUSE', bases='ALL', starres=None):
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
    if starres == None:
        starres = '%s_x%i_y%i_star_res.txt' %(s3d.inst, x, y)
        if os.path.isfile(starres):
            os.remove(starres)    
            
    wl, spec, err = s3d.extrSpec(x=x, y=y, size=size, verbose=0)
    ascii = asciiout(s3d=s3d, wl=wl, spec=spec, err=err, frame='rest',
                     resample = 1, name='%s_%s_%s' %(x, y, size), fmt='txt')
                      
    data, stars, success, v0, vd, av = runStar(s3d, ascii, bases=bases, verbose=0)
    f = open(starres, 'a')
    f.write('%i\t%i\t%.1f\t%.1f\t%.3f\n' %(x, y, v0, vd, av))
    f.close()
    os.remove(ascii)

    miny, maxy = max(0, y-size), min(s3d.leny-1, y+size+1)
    minx, maxx = max(0, x-size), min(s3d.lenx-1, x+size+1)
    
    xindizes = np.arange(minx, maxx, 1)
    yindizes = np.arange(miny, maxy, 1)
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
                # No sucess
                s3d.starcube[:, yindx, xindx] = zerospec
    return
    

def subAllStars(s3d, dx=2, nc=None, x1=None, x2=None, y1=None, y2=None,
                bases = 'FEW'):
    """ 
    Convinience function to subtract starlight fits on the full cube. Can work
    with subcubes defined by x1, x2, y1, y2. Resamples by a factor of 2*dx+1.
    """
    
    logger.info("Starting starlight on full cube with %i cores" %s3d.ncores)
    logger.info("This might take a bit")
    t1 = time.time()
    
    if x1 != None and x2!= None:
        logger.info("X-range: %i to %i" %(x1, x2))
        xindizes = np.arange(x1, x2, 2*dx+1)
    else:
        xindizes = np.arange(dx, s3d.lenx, 2*dx+1)

    if y1 != None and y2!= None:
        logger.info("Y-range: %i to %i" %(y1, y2))
        yindizes = np.arange(y1, y2, 2*dx+1)
    else:
        yindizes = np.arange(dx, s3d.leny, 2*dx+1)

    starres = '%s_x%i_%i_y%i_%i_star_res.txt' \
        %(s3d.inst, xindizes[0], xindizes[-1], yindizes[0], xindizes[-1])
    if os.path.isfile(starres):
        os.remove(starres)

    for xindx in xindizes:
        for yindx in yindizes:
            subStars(s3d, xindx, yindx, dx, 
                     bases=bases, verbose=0, starres=starres)
            
    cubeout(s3d, s3d.starcube, err=s3d.erro, name='star')
    cubeout(s3d, s3d.data-s3d.starcube, err=s3d.erro, name='gas')
    logger.info("This took %.2f h" %((time.time()-t1)/3600.))
