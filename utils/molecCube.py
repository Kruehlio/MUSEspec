# -*- coding: utf-8 -*-

""" Correcting for telluric absorption. Primarily used for MUSE spectra,
and in combination with the spectrum3d class.
 """


import os
import subprocess
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

molecBase = os.path.expanduser('~/bin/molecfit')
molecCall = os.path.join(molecBase, 'bin/molecfit')
transCall = os.path.join(molecBase, 'bin/calctrans')

class molecCubeFit():

    ''' Sets up and runs a molecfit to spectral data - requires molecfit in its
    version 1.2.0 or greater. 
    See http://www.eso.org/sci/software/pipelines/skytools/molecfit
    '''
    
    def __init__(self, s3d, header = '', output='.'):
        ''' Reads in the required information to set up the parameter files'''
        
        self.s3d = s3d
        self.molecparfile = {}
        
        self.params = {'basedir': molecBase,
           'listname': 'none', 'trans' : 1,
           'columns': 'LAMBDA FLUX ERR NULL',
           'default_error': 0.01, 'wlgtomicron' : 0.0001,
           'vac_air': 'vac', 'list_molec': [], 'fit_molec': [],
           'wrange_exclude': 'none',
           'output_dir': os.path.abspath(os.path.expanduser(output)),
           'plot_creation' : 'P', 'plot_range': 1,
           'ftol': 0.01, 'xtol': 0.01, 
           'relcol': [], 'flux_unit': 2,
           'fit_back': 0, 'telback': 0.1, 'fit_cont': 1, 'cont_n': 4,
           'cont_const': 1.0, 'fit_wlc': 1, 'wlc_n': 1, 'wlc_const': 0.0,
           'fit_res_box': 0, 'relres_box': 0.0, 'kernmode': 0, 
           'fit_res_gauss': 1, 'res_gauss': 4.0, 
           'fit_res_lorentz': 0, 'res_lorentz': 0.5, 
           'kernfac': 30.0, 'varkern': 1, 'kernel_file': 'none'}
        
        self.headpars = {'utc': 'UTC', 'telalt': 'HIERARCH ESO TEL ALT', 
             'rhum' : 'HIERARCH ESO TEL AMBI RHUM',
             'obsdate' : 'MJD-OBS',
             'temp' : 'HIERARCH ESO TEL AMBI TEMP',
             'm1temp' : 'HIERARCH ESO TEL TH M1 TEMP',
             'geoelev': 'HIERARCH ESO TEL GEOELEV',
             'longitude': 'HIERARCH ESO TEL GEOLON',
             'latitude': 'HIERARCH ESO TEL GEOLAT',
             'pixsc' : 'HIERARCH ESO OCS IPS PIXSCALE' } 
        
        self.atmpars = {'ref_atm': 'equ.atm', 
            'gdas_dir': os.path.join(molecBase, 'data/profiles/grib'),
            'gdas_prof': 'auto', 'layers': 1, 'emix': 5.0, 'pwv': -1.} 

    def updateParams(self, arm, paramdic, headpars):
        ''' Sets Parameters for molecfit execution'''
        
        for key in paramdic.keys():
            print '\t\tMolecfit: Setting parameter %s to %s' %(key,  paramdic[key])
            if key in self.params.keys():
                self.params[key] = paramdic[key]
            elif key in self.headpars.keys():
                self.headpars[key] = paramdic[key]
            elif key in self.atmpars.keys():
                self.atmpars[key] = paramdic[key]
            else:
                print '\t\tWarning: Parameter %s not known to molecfit'
                self.params[key] = paramdic[key]
        
    def setParams(self, specfile):
        ''' Writes Molecfit parameters into file '''

        wlinc = [[0.686, 0.694], [0.758, 0.762], [0.762, 0.770]] 
#                    [0.894, 0.904], [0.920, 0.933]]

        wrange = '%s_molecfit_muse_inc.dat' %(self.s3d.target)
        f = open(wrange, 'w')
        for wls in wlinc:
            f.write('%.4f %.4f\n' %(wls[0], wls[1]))
        f.close

        prange = os.path.join(molecBase, 'examples/config/exclude_muse.dat')
        self.params['wrange_include'] =  os.path.abspath(wrange)
        self.params['prange_exclude'] =  prange

        self.params['list_molec'] = ['H2O', 'O2']
        self.params['fit_molec'] = [1, 1]
        self.params['relcol'] = [1.0, 1.0]
        self.params['wlc_n'] = 0

        print '\tPreparing Molecfit params file'
        self.molecparfile = os.path.abspath('./%s_molecfit_muse.par' % (self.s3d.target))

        f = open(self.molecparfile, 'w')
        f.write('## INPUT DATA\n')
        f.write('filename: %s\n' % (os.path.abspath(specfile)))
        f.write('output_name: %s_muse_molecfit\n' % (self.s3d.target))

        for param in self.params.keys():
            if hasattr(self.params[param], '__iter__'):
                f.write('%s: %s \n' % (param, ' '.join([str(a) for a in self.params[param]]) ))
            else:
                f.write('%s: %s \n' % (param, self.params[param] ))

        f.write('\n## HEADER PARAMETERS\n')
        f.write('slitw: 1.0\n' )
        for headpar in self.headpars.keys():
            f.write('%s: %s \n' % (headpar,  self.s3d.headprim[self.headpars[headpar]]))

        f.write('\n## ATMOSPHERIC PROFILES\n')
        for atmpar in self.atmpars.keys():
            f.write('%s: %s \n' % (atmpar, self.atmpars[atmpar]))
        
        f.write('\nend\n')
        f.close()
        self.tacfile = os.path.splitext(specfile)[0]+'_TAC.spec'

            
    def runMolec(self):
        t1 = time.time()
        print '\tRunning molecfit'
        print '\t%s' %(' '.join([molecCall, self.molecparfile]))
        runMolec = subprocess.Popen([molecCall, self.molecparfile],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)
                                    
        runMolec.wait()
        runMolecRes = runMolec.stdout.readlines()
        molecpar = os.path.abspath('./%s_molecfit.output' % (self.s3d.target))
        f = open(molecpar, 'w')
        f.write(''.join(runMolecRes))
        f.close()
        
        if runMolecRes[-1].strip() == '[ INFO  ] No errors occurred':
            print '\tMolecfit sucessful in %.0f s' % (time.time()-t1)
        else:
            print runMolecRes[-1].strip()
        
        t1 = time.time()
        print '\tRunning calctrans'
        runTrans = subprocess.Popen([transCall, self.molecparfile],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)
        runTrans.wait()
        runTransRes = runTrans.stdout.readlines()
        runtranspar = os.path.abspath('./%s_calctranss.output' % (self.s3d.target))
        f = open(runtranspar, 'w')
        f.write(''.join(runTransRes))
        f.close()        
        
        if runTransRes[-1].strip() == '[ INFO  ] No errors occurred':
            print '\tCalctrans sucessful in %.0f s' % (time.time()-t1)
        else:
            print runTransRes[-1].strip()
            
    def updateCube(self, tacfile = ''):
        
        ''' Read in Calctrans output und update spectrum3d class with telluric
        correction spectra '''
        print '\tUpdating the cube with telluric-corrected data'

        if tacfile == '':
            tacfile = self.tacfile

        if os.path.isfile(tacfile):
            f = open(tacfile, 'r')
            tacspec = [item for item in f.readlines() if not item.startswith('#')]
            f.close()
            wl, rawspec, rawspece, transm, tcspec, tcspece = 6 * [np.array([])]
            for spec in tacspec:
                column = [float(ent) for ent in spec.split()]
                wl = np.append(wl, column[0])
                rawspec = np.append(rawspec, column[1])
                rawspece = np.append(rawspece, column[2])
                transm = np.append(transm, column[3])
                tcspec = np.append(tcspec, column[4])
                tcspece = np.append(tcspece, column[5])

            self.s3d.data /= transm[:,np.newaxis, np.newaxis]
            self.s3d.erro /= transm[:,np.newaxis, np.newaxis]
            self.s3d.wave = wl
            self.s3d.atmotrans = transm
            self.s3d.head['CRVAL3'] = wl[0]
            self.s3d.head['CD3_3'] = (wl[-1]-wl[0])/(len(wl)-1.)
                
            # Plot the fit regions
            wrange = os.path.abspath('%s_molecfit_muse_inc.dat' %(self.s3d.target))
            g = open(wrange, 'r')
            fitregs = [reg for reg in g.readlines() if not reg.startswith('#')]
            g.close()
            
            pp = PdfPages('%s_tellcor.pdf' % (self.s3d.target) )
            print '\tPlotting telluric-corrected data for arm'

            for fitreg in fitregs:
                mictowl = 1./self.params['wlgtomicron']
                if float(fitreg.split()[1])*mictowl < self.s3d.wave[-1]:
                    x1 = self.wltopix(float(fitreg.split()[0])*mictowl)
                    x2 = self.wltopix(float(fitreg.split()[1])*mictowl)
                    
        
                    fig = plt.figure(figsize = (9.5, 7.5))
                    fig.subplots_adjust(hspace=0.05, wspace=0.0, right=0.97)
                    ax1 = fig.add_subplot(2, 1, 1)
                    wlp, tcspecp, transmp = wl[x1:x2], tcspec[x1:x2], transm[x1:x2] 

                    if len(tcspecp[transmp>0.90]) > 3:
                        cont = np.median(tcspecp[transmp>0.90])/np.median(transmp[transmp>0.90])
                        ax1.plot(wlp, transmp * cont, '-' ,color = 'firebrick', lw = 2)
                    else:
                        ax1.errorbar(wlp, tcspec[x1:x2], tcspece[x1:x2],
                        capsize = 0, color = 'firebrick', fmt = 'o', ms = 4,
                        mec = 'grey', lw = 0.8, mew = 0.5)
                    
                    ax1.errorbar(wlp, rawspec[x1:x2], rawspece[x1:x2],
                        capsize = 0, color = 'black', fmt = 'o', ms = 4,
                        mec = 'grey', lw = 0.8, mew = 0.5)
        
                    ax2 = fig.add_subplot(2, 1, 2)
                    if len(tcspecp[transmp>0.90]) > 3:
                        ax2.errorbar(wlp, rawspec[x1:x2] / (transmp * cont), 
                            yerr = rawspece[x1:x2] / (transmp * cont),
                            capsize = 0, color = 'black', fmt = 'o',  ms = 4,
                            mec = 'grey', lw = 0.8, mew = 0.5)
                    else:
                        ax2.plot(wlp, transmp, '-' ,color = 'firebrick', lw = 2)

        
                    for ax in [ax1, ax2]:
                        ax.yaxis.set_major_formatter(plt.FormatStrFormatter(r'$%s$'))
                        ax.set_xlim(float(fitreg.split()[0])*mictowl, float(fitreg.split()[1])*mictowl)
        
                    ax1.xaxis.set_major_formatter(plt.NullFormatter())
                    ax2.set_xlabel(r'$\rm{Observed\,wavelength\, (\AA)}$')
                    ax1.set_ylabel(r'$F_{\lambda}\,\rm{(10^{-17}\,erg\,s^{-1}\,cm^{-2}\, \AA^{-1})}$')
                    
                    if len(tcspecp[transmp>0.90]) > 3:
                        ax2.set_ylim(ymax = max(rawspec[x1:x2] / (transmp * cont))*1.05)
                        ax2.set_ylabel(r'Ratio')
                    else:
                        ax2.set_ylim(ymax=1.05)
                        ax2.set_ylabel(r'Transmission')
                    ax1.set_ylim(ymax = max(tcspec[x1:x2])*1.05)
        
                    pp.savefig(fig)
                    plt.close(fig)  
            pp.close()      
        return self.s3d
        
    def wltopix(self, wl):
        """ Converts wavelength as input into nearest integer pixel value """
        pix = ((wl - self.s3d.wave[0]) / self.s3d.wlinc)
        return max(0, int(round(pix)))