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

import os

import pybinary


# MAIN
#==============================================================================
#idir = '161108_kpfiles_sub3\\'
#odir = ''

# GO Tau + JH 223/JH 223
#src_kpfiles = ['cube257_kernel.fits']
#cal_kpfiles = ['cube248_kernel.fits', 'cube263_kernel.fits']

# DH Tau/DH Tau/DH Tau + IRAS 04187+1927/LkHa 358/2MASS J04221675
#src_kpfiles = ['cube119_kernel.fits', 'cube390_kernel.fits', 'cube414_kernel.fits']
#cal_kpfiles = ['cube104_kernel.fits', 'cube396_kernel.fits', 'cube408_kernel.fits']

idir = '161108_kpfiles/'
odir = 'test/'

#FNUM_161107 = ['cube244.fits', 'cube252.fits', 'cube258.fits', 'cube272.fits', 'cube278.fits', 'cube420.fits', 'cube426.fits', 'cube438.fits', 'cube444.fits', 'cube456.fits', 'cube482.fits', 'cube550.fits', 'cube556.fits', 'cube568.fits']
#FNUM_161107_art = ['cube238.fits', 'cube264.fits', 'cube377.fits', 'cube389.fits', 'cube396.fits', 'cube414.fits', 'cube432.fits', 'cube450.fits', 'cube462.fits', 'cube472.fits', 'cube508.fits', 'cube514.fits', 'cube524.fits', 'cube574.fits']
#FNUM_161108 = ['cube119.fits', 'cube141.fits', 'cube193.fits', 'cube248.fits', 'cube263.fits', 'cube420.fits', 'cube444.fits', 'cube452.fits']
#FNUM_161108_art = ['cube4.fits', 'cube9.fits', 'cube28.fits', 'cube34.fits', 'cube52.fits', 'cube104.fits', 'cube131.fits', 'cube162.fits', 'cube169.fits', 'cube175.fits', 'cube181.fits', 'cube187.fits', 'cube199.fits', 'cube212.fits', 'cube218.fits', 'cube224.fits', 'cube230.fits', 'cube236.fits', 'cube242.fits', 'cube332.fits', 'cube338.fits', 'cube350.fits', 'cube356.fits', 'cube362.fits', 'cube376.fits', 'cube390.fits', 'cube396.fits', 'cube402.fits', 'cube408.fits', 'cube414.fits', 'cube426.fits', 'cube432.fits', 'cube438.fits']
#FNUM_161109 = ['cube111.fits', 'cube123.fits', 'cube129.fits', 'cube147.fits', 'cube153.fits', 'cube260.fits', 'cube305.fits', 'cube311.fits', 'cube323.fits', 'cube329.fits', 'cube341.fits', 'cube347.fits', 'cube353.fits', 'cube389.fits', 'cube419.fits']
#FNUM_161109_art = ['cube117.fits', 'cube135.fits', 'cube159.fits', 'cube176.fits', 'cube194.fits', 'cube212.fits', 'cube218.fits', 'cube224.fits', 'cube242.fits', 'cube248.fits', 'cube254.fits', 'cube290.fits', 'cube295.fits', 'cube299.fits', 'cube335.fits', 'cube359.fits', 'cube365.fits', 'cube371.fits', 'cube377.fits', 'cube383.fits', 'cube425.fits']

#dir1 = '161107_kpfiles_sub3/'
#dir2 = '161108_kpfiles_sub3/'
#dir3 = '161109_kpfiles_sub3/'
#
#for i in range(len(FNUM_161107)):
#    FNUM_161107[i] = dir1+FNUM_161107[i]
#for i in range(len(FNUM_161107_art)):
#    FNUM_161107_art[i] = dir1+FNUM_161107_art[i]
#for i in range(len(FNUM_161108)):
#    FNUM_161108[i] = dir2+FNUM_161108[i]
#for i in range(len(FNUM_161108_art)):
#    FNUM_161108_art[i] = dir2+FNUM_161108_art[i]
#for i in range(len(FNUM_161109)):
#    FNUM_161109[i] = dir3+FNUM_161109[i]
#for i in range(len(FNUM_161109_art)):
#    FNUM_161109_art[i] = dir3+FNUM_161109_art[i]

#FNUM = FNUM_161107+FNUM_161107_art+FNUM_161108+FNUM_161108_art+FNUM_161109+FNUM_161109_art
#FNUM = FNUM_161108+FNUM_161108_art

FNUM = [f for f in os.listdir(idir) if f.endswith('_kernel.fits')]
TARGNAME = []
for i in range(len(FNUM)):
#    FNUM[i] = FNUM[i][:-5]+'_kernel.fits'
    TARGNAME += [pyfits.getheader(idir+FNUM[i])['TARGNAME']]

# Exclude HD targets
FNUM_temp = []
TARGNAME_temp = []
for i in range(len(FNUM)):
    if ('HD' not in TARGNAME[i]):
        FNUM_temp += [FNUM[i]]
        TARGNAME_temp += [TARGNAME[i]]
FNUM = FNUM_temp
TARGNAME = TARGNAME_temp

TARGS = np.unique(TARGNAME)

# Reject bad frames based on Fourier power
imgs = []
radiis = []
azavgs = []
for i in range(len(FNUM)):
    
    # Compute the maximum spatial frequency supported by the primary mirror
    if (i == 0):
        xsz = pyfits.getheader(idir+FNUM[i])['NAXIS1']
        cwave = pyfits.getheader(idir+FNUM[i])['CENWAVE']*1E-6
        pscale = 10.0
        m2pix = xara.core.mas2rad(pscale)*xsz/cwave
        sfmax = 8.*m2pix # Primary mirror diameter = 8 meters
    
    # Load data cube
    imgs += [pyfits.getdata(idir+FNUM[i])]
    nimgs = imgs[-1].shape[0]
    
    # Compute Fourier power of data cube
    fpow = np.zeros(imgs[-1].shape)
    radii = []
    azavg = []
    for j in range(nimgs):
        fpow[j] = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(imgs[-1][j]))))**2
        fpow[j] /= np.max(fpow[j])
        
        # Compute azimuthal average using opticstools
        r, a = ot.azimuthalAverage(fpow[j], returnradii=True, binsize=1.)
        radii += [r[r < sfmax]]
        azavg += [a[r < sfmax]]
    radiis += [radii]
    azavgs += [azavg]
radiis = np.concatenate(radiis)
azavgs = np.concatenate(azavgs)

#
fpowmax = np.max(azavgs, axis=0) # Maximum Fourier power
fpowstd = np.std(azavgs, axis=0) # Standard deviation of Fourier power
cuts = np.repeat(fpowmax-3.*fpowstd[np.newaxis, :], azavgs.shape[0], axis=0) # Everything below max-3*std is bad
flag = np.sum(azavgs < cuts, axis=1) < azavgs.shape[1]*0.50 # Reject frame if more than 50% of azimuthal averages are bad

# Reorder flags
flags = []
nframe = 0
for i in range(len(FNUM)):
    nimgs = imgs[i].shape[0]
    flags += [flag[nframe:nframe+nimgs]]
    nframe += nimgs
    
#    print(flags[i])
#    f, axarr = plt.subplots(1, nimgs)
#    for j in range(nimgs):
#        axarr[j].imshow(np.angle(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(imgs[i][j])))))
#        axarr[j].axes.get_xaxis().set_visible(False)
#        axarr[j].axes.get_yaxis().set_visible(False)
#    plt.show(block=True)
flags = np.array(flags)

#for i in range(TARGS.shape[0]):
#    print(str(i)+'\t'+str(TARGNAME.count(TARGS[i]))+'\t'+TARGS[i])
#ii = input('Enter source target index: ')
#FNUM = np.array(FNUM)
#TARGNAME = np.array(TARGNAME)
#ws = np.where(TARGNAME == TARGS[ii])
#wc = np.where(TARGNAME != TARGS[ii])
#src_kpfiles = list(FNUM[ws])
#cal_kpfiles = list(FNUM[wc])

FNUM = np.array(FNUM)
TARGNAME = np.array(TARGNAME)
for i in range(TARGS.shape[0]):
    print(i)
    ws = np.where(TARGNAME == TARGS[i])
    wc = np.where(TARGNAME != TARGS[i])
    src_kpfiles = list(FNUM[ws])
#    src_flags = None
    src_flags = flags[ws]
    cal_kpfiles = list(FNUM[wc])
#    cal_flags = None
    cal_flags = flags[wc]
    
    # If there is no good source frame jump to the next target, otherwise
    # perform PyBinary analysis
    if (np.sum(np.concatenate(src_flags)) < 1):
        continue
    else:
        PyBinary = pybinary.PyBinary(idir,
                                     odir,
                                     src_kpfiles,
                                     cal_kpfiles,
                                     src_flags=src_flags,
                                     cal_flags=cal_flags,
                                     name='auto',
                                     K_klip=0)
#        PyBinary.gridsearch(grid_size=300,
#                            sampling='pscale/2',
#                            method='sim',
#                            multiklip=False)
#        PyBinary.leastsquares(p0=None,
#                              rho_min=40)
#        PyBinary.multiklip(K_klip=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
#                           grid_size=500,
#                           sampling='pscale/2',
#                           method='sim',
#                           rho_min=40)
        PyBinary.multiklip_sub(K_klip=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                               grid_size=500,
                               sampling='pscale/2',
                               method='sim',
                               rho_min=40)
