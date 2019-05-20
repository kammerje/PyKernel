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

import pykernel

#idir = 'E:\\NIRC2\\192x192\\161107_cubes\\'
idir = '/priv/mulga2/kjens/NIRC2/192x192/161107_cubes/'
odir = '161107_kpfiles/'
tel = 'Keck'
pupil_rot = 'auto' # Extracts pupil rotation from the fits header
#pupil_rot = 44. # degrees
grid_size = 1024 # pixels; Not required if advanced Keck pupil model is used
sampling = 0.4 # meters; Not required if advanced Keck pupil model is used
bmax = None # meters

#bdir = 'E:\\NIRC2\\161107_blockinfo.txt'
bdir = '/priv/mulga2/kjens/NIRC2/161107_blockinfo.txt'
FNUM = []
TARGNAME = []
binf = open(bdir, 'r')
for line in binf:
    line = line.split()
    if (line[2] == 'Lp' and line[4] == 'clear'):
        FNUM += ['cube'+str(int(line[0]))+'.fits']
        if (len(line) == 13):
            TARGNAME += [line[6]]
        else:
            TARGNAME += [line[7]+' '+line[8]]
import pdb; pdb.set_trace()


# MAIN
#==============================================================================
phase_std = []
kernel_phase_std = []

for i, fitsfile in enumerate(FNUM):
    
#        data = pyfits.getdata(idir+fitsfile)
#        header = pyfits.getheader(idir+fitsfile)
#        plt.figure()
#        plt.imshow(np.median(data, axis=0))
#        plt.show()
#        plt.close()
#        import pdb; pdb.set_trace()
    try:
        # Pupil rotation according to Mike's pynrm pipeline
        if (pupil_rot == 'auto'):
            header = pyfits.getheader(idir+fitsfile) # Get fits header
            if (tel == 'VLT'):
                altstart = 90.-(180./np.pi)*np.arccos(1./header['ESO TEL AIRM START'])
                altend = 90.-(180./np.pi)*np.arccos(1./header['ESO TEL AIRM END'])
                rot = (header['ESO ADA ABSROT START']+header['ESO ADA ABSROT END'])/2.
                rot += (altstart+altend)/2.
                rot = -rot
            elif (tel == 'Keck'):
                rot = -(header['ROTPPOSN']-header['EL']-header['INSTANGL'])
            else:
                raise UserWarning(str(tel)+' is not a known telescope')
        else:
            rot = pupil_rot
        print('Pupil rotation is %.1f degrees' % (-rot))
        
        PyKernel = pykernel.PyKernel(idir=idir,
                                     odir=odir,
                                     tel=tel,
                                     pupil_rot=rot,
                                     make_pupil=True,
                                     grid_size=grid_size,
                                     sampling=sampling,
                                     bmax=bmax)
        
        PyKernel.extract_kerphase(fitsfiles=[fitsfile],
                                  recenter=True,
                                  wrad=50,
                                  method='LDFT1')
        PyKernel.compute_covariance()
        PyKernel.save_as_fits()
    
    except:
        pass
