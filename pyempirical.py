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

import ast
from scipy.interpolate import interp1d


# MAIN
#==============================================================================
class PyEmpirical():
    
    def __init__(self,
                 idir,
                 odir,
                 src_pbfile,
                 cal_pbfiles):
        """
        TODO
        """
        
        # Read input parameters
        self.idir = idir
        self.odir = odir
        self.src_pbfile = src_pbfile
        self.cal_pbfiles = cal_pbfiles
        
        pass
    
    def azavg(self,
              img):
        """
        TODO
        """
        
        # Compute azimuthal average using opticstools
        radii, azavg = ot.azimuthalAverage(img, returnradii=True, binsize=1.)
        
        # Return azimuthal average, the first value is always inf (this is
        # because the grid search finds an infinite contrast for the central
        # pixel where the binary model is not defined)
        return radii[1:], azavg[1:]
    
    def __f_corrs_from_cals(self):
        """
        TODO
        """
        
        # Compute correction factors for determining an empirical detection
        # limit, one for each K_klip
        f_corrs = []
        for i in range(len(self.cal_pbfiles)):
            
            # Check whether the source and the calibrator PyBinary files
            # contain data for the same K_klips
            hdul = pyfits.open(self.idir+self.cal_pbfiles[i])
            temp = hdul['MULTIKLIP'].header['KKLIP']
            K_klip = np.array(ast.literal_eval(temp)) # array of K_klips
            if (np.array_equal(self.K_klip, K_klip) == False):
                hdul.close()
                raise UserWarning('Source PyBinary file does not contain data for the same K_klips as '+str(self.cal_pbfiles[i]))
            
            # Check whether the source and the calibrator PyBinary files
            # were computed on the same grid
            seps = hdul['MULTIKLIP'].data[4, 0]
            if (np.array_equal(self.seps, seps) == False):
                hdul.close()
                raise UserWarning('Source PyBinary file was not computed on the same grid as '+str(self.cal_pbfiles[i]))
            
            # Load (1) best fit contrast before and (2) after subtracting the
            # best fit binary model directly from the kernel-phase data
            cs_mean = hdul['MULTIKLIP'].data[0]
            cs_mean_sub = hdul['MULTIKLIP_SUB'].data[0]
            hdul.close()
            
            #
            f_corr = []
            for j in range(len(self.K_klip)):
                
                # Compute azimuthal average of (1) and (2), therefor use
                # absolute of best fit contrast because the map is symmetric
                # wrt the origin and the azimuthal average would be zero
                # otherwise
                _, azavg = self.azavg(np.abs(cs_mean[j]))
                _, azavg_sub = self.azavg(np.abs(cs_mean_sub[j]))
                
                # The correction factor is the ratio of the azimuthal average
                # of (1) and (2) and can later be multiplied by the azimuthal
                # average of (2) from the source in order to obtain an
                # empirical detection limit
                f_corr += [np.true_divide(azavg, azavg_sub)]
            f_corr = np.array(f_corr)
            f_corrs += [f_corr]
        f_corrs = np.array(f_corrs)
        
        # Average over all calibrators
        self.f_corrs = np.mean(f_corrs, axis=0)
        
        pass
    
    def cs_sdev_empirical(self):
        """
        TODO
        """
        
        print('--> Computing empirical detection limits')
        
        # Load K_klips and grid (in order to check whether the source and
        # calibrator PyBinary files are compatible)
        hdul = pyfits.open(self.idir+self.src_pbfile)
        temp = hdul['MULTIKLIP'].header['KKLIP']
        self.K_klip = np.array(ast.literal_eval(temp)) # array of K_klips
        self.seps = hdul['MULTIKLIP'].data[4, 0]
        self.pas = hdul['MULTIKLIP'].data[5, 0]
        
        # Load (3) best fit contrast after subtracting the best fit binary
        # model directly from the kernel-phase data
        cs_mean_sub = hdul['MULTIKLIP_SUB'].data[0]
        hdul.close()
        
        # Compute correction factors for determining an empirical detection
        # limit
        self.__f_corrs_from_cals()
        
        #
        cs_sdev = []
        rs = []
        cs_curv = []
        for i in range(len(self.K_klip)):
            
            # Compute azimuthal average of (3), therefor use absolute of best
            # fit contrast because the map is symmetric wrt the origin and the
            # azimuthal average would be zero otherwise
            radii, azavg = self.azavg(np.abs(cs_mean_sub[i]))
            
            #
            ramp = self.seps[0]*np.cos(np.radians(self.pas[0]))
            step = (ramp[-1]-ramp[0])/(ramp.shape[0]-1.)
            radii *= step
            
            # The empirical detection limit is the azimuthal average of (3)
            # multiplied by the correction factor
            detlim = np.multiply(azavg, self.f_corrs[i])
            
            # Disregard separations which do not lie within the range
            # characterized by the azimuthal average
            if (i == 0):
                r_min = radii.min()
                r_max = radii.max()
                s_min = self.seps.min()
                s_max = self.seps.max()
                if (s_min < r_min):
                    print('Min grid distance (%.1f mas) < r_min (%.1f mas)' % (s_min, r_min))
                if (s_max > r_max):
                    print('Max grid distance (%.1f mas) > r_max (%.1f mas)' % (s_max, r_max))
                self.seps[self.seps < r_min] = np.nan
                self.seps[self.seps > r_max] = np.nan
            
            # Interpolate the empirical detection limit with a linear function
            # in log-space, so that it can be evaluated at the separations of
            # the grid
            f_log = interp1d(radii, np.log(detlim), kind='linear')
            cs_sdev += [np.exp(f_log(self.seps))]
            
            # Compute contrast curve
            rs += [np.linspace(r_min, r_max, 1024)]
            cs_curv += [np.exp(f_log(rs[-1]))]
        
        # Prepare empirical detection limit HDU
        self.cs_sdev = np.array(cs_sdev)
        
        # Save empirical detection limit HDU to PyBinary file
        edl_hdu = pyfits.ImageHDU(self.cs_sdev)
        edl_hdu.header['EXTNAME'] = 'EMPIRICAL'
        edl_hdu.header['KKLIP'] = str(self.K_klip.tolist())
        for i in range(len(self.cal_pbfiles)):
            edl_hdu.header['HISTORY'] = 'Cal: '+self.cal_pbfiles[i]
        hdul = pyfits.open(self.idir+self.src_pbfile)
        try:
            test = hdul['EMPIRICAL']
            hdul.pop(hdul.index_of('EMPIRICAL'))
        except:
            pass
        hdul.append(edl_hdu)
        hdul.writeto(self.odir+self.src_pbfile, clobber=True, output_verify='fix') # FIXME
        hdul.close()
        
        # Plot contrast curve
        plt.figure()
        for i in range(len(self.K_klip)):
            plt.plot(rs[i], cs_curv[i], label=str(self.K_klip[i]))
        plt.yscale('log')
        plt.xlabel('Separation [mas]')
        plt.ylabel('Contrast')
        plt.legend()
        plt.show(block=True)
        plt.close()
        
        print('--------------------')
        
        pass
