"""
PyKernel, a Python package to analyze imaging data with the kernel-phase
technique using Frantz Martinache's XARA library
(https://github.com/fmartinache/xara). The PyKernel library is maintained on
GitHub at https://github.com/kammerje/PyKernel.

Author: Jens Kammerer
Version: 3.0.0
Last edited: 15.01.19
"""


# PREAMBLE
#==============================================================================
# Requires XARA (https://github.com/fmartinache/xara) and opticstools
# (https://github.com/mikeireland/opticstools.git)
import sys
#sys.path.append('F:\\Python\\Development\\NIRC2\\xara')
sys.path.append('/home/kjens/Python/Development/NIRC2/xara')
#sys.path.append('F:\\Python\\Packages\\opticstools\\opticstools\\opticstools')
sys.path.append('/home/kjens/Python/Packages/opticstools/opticstools/opticstools')
import xara
import opticstools as ot

import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import numpy as np

import pyempirical


# MAIN
#==============================================================================
idir = '161109_pbfiles_sub/'
odir = '161109_emp/'
src_pbfile = 'UZ Tau A.fits'
#cal_pbfiles = ['2MASS J04321540.fits', 'CX Tau.fits', 'DO Tau.fits', 'FP Tau.fits', 'GO Tau.fits', 'JH 223.fits']
#cal_pbfiles = ['2MASS J04221675.fits', '2MASS J04354093.fits', '2MASS J05052286.fits', 'DH Tau.fits', 'GM Aur.fits', 'HN Tau A.fits', 'JH 223.fits', 'LkHa 358.fits']
cal_pbfiles = ['2MASS J04154278.fits', '2MASS J05052286.fits', 'CX Tau.fits', 'FP Tau.fits', 'FT Tau.fits', 'GO Tau.fits', 'HK Tau.fits', 'HN Tau A.fits', 'IRAS 04108+2910.fits', 'JH 223.fits', 'V409 Tau.fits']

PyEmpirical = pyempirical.PyEmpirical(idir,
                                      odir,
                                      src_pbfile,
                                      cal_pbfiles)
PyEmpirical.cs_sdev_empirical()
