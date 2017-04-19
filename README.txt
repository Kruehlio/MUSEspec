""" 
Various tools for processing, handling, and analysing astronomical IFU spectral data

Usage in python:

# Import class
import numpy as np
from MUSEspec.spectrum3d import Spectrum3d

# Create instance, load header info files, convert wavelengths to vacuum (MUSE pipeline provides air)
s = Spectrum3d(filen = ''filename'')

# Set Redshift
s.setRedshift(0.0086)

# Correct for galactic foreground using RA, DEC from header
s.ebvGal()

# Get EW map for Halpha
haew, haewsn, cont = s.getEW(line='Ha')

#haew = EW map of Halpha
#haewsn = S/N map of haew
#cont = Continuum flux around Halpha

# Apply S/N thresholding

haew[sn < 5] = np.nan

# Plot Ha EW map
s.pdfout(np.log10(haew), name = 'EW', label = r'$\log10(\mathrm{EW}(H\alpha))$',
        vmin =-1.0, vmax=2.7,
        psf=0.9, source = r'98bw', cmap='jet',
        ra = '19:35:03.315', dec='-52:50:44.99')

"""
