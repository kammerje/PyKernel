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
sys.path.append('F:\\Python\\Development\\NIRC2\\xara')
#sys.path.append('/home/kjens/Python/Development/NIRC2/xara')
sys.path.append('F:\\Python\\Packages\\opticstools\\opticstools\\opticstools')
#sys.path.append('/home/kjens/Python/Packages/opticstools/opticstools/opticstools')
import xara
import opticstools as ot

import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import numpy as np

#from scipy.optimize import leastsq
#from scipy.optimize import least_squares
from scipy.optimize import minimize

import kl_calibration


# MAIN
#==============================================================================
class PyBinary():
    
    def __init__(self,
                 idir,
                 odir,
                 src_kpfiles,
                 cal_kpfiles,
                 src_flags=None,
                 cal_flags=None,
                 name='auto',
                 K_klip=4):
        """
        TODO
        """
        
        # Read input parameters
        self.idir = idir
        self.odir = odir
        self.src_kpfiles = src_kpfiles
        self.src_flags = src_flags
        self.cal_kpfiles = cal_kpfiles
        self.cal_flags = cal_flags
        
        # Read source and calibrator data
        self.src_from_fits(self.idir,
                           self.src_kpfiles,
                           self.src_flags)
        self.cal_kpfiles = self.cal_from_fits(self.idir,
                                              self.cal_kpfiles,
                                              self.cal_flags)
        
        # Connect KL calibration
        self.__connect_KL(K_klip)
        
        # Compute inverse source kernel-phase covariances
        self.src_inverse(self.src_cvs,
                         self.src_cvs_KL)
        
        # Make a PyBinary file with the specified name
        if (name == 'auto'):
            self.name = self.__get_name(self.idir,
                                        self.src_kpfiles,
                                        name)
        elif (name.endswith('.fits')):
            self.name = name
        else:
            self.name = name+'.fits'
        self.__make_pybinary_file(self.idir,
                                  self.src_kpfiles,
                                  self.cal_kpfiles)
        
        pass
    
    def src_from_fits(self,
                      idir,
                      kpfiles,
                      flags):
        """
        TODO
        """
        
        # Read source data
        src_kps = []
        src_cvs = []
        src_pas = []
        for i in range(len(kpfiles)):
            hdul = pyfits.open(idir+kpfiles[i], memmap=False)
            if (flags is not None):
                src_kps += [hdul['KP-DATA'].data[flags[i]]]
                src_cvs += [hdul['KP-SIGM'].data[flags[i]]]
                try:
                    src_pas += [np.mean(hdul['TEL'].data['DETPA'])] # Position
                                                                    # angles from
                                                                    # PyConica
                                                                    # pipeline
                except:
                    src_pas += [hdul['TEL'].data['pa'][flags[i]]]
            else:
                src_kps += [hdul['KP-DATA'].data]
                src_cvs += [hdul['KP-SIGM'].data]
                try:
                    src_pas += [np.mean(hdul['TEL'].data['DETPA'])] # Position
                                                                    # angles from
                                                                    # PyConica
                                                                    # pipeline
                except:
                    src_pas += [hdul['TEL'].data['pa']]
            if (i == 0):
                self.cwave = hdul[0].header['CWAVE'] # Central filter
                                                     # wavelength (meters)
                self.pscale = hdul[0].header['PSCALE'] # Pixel scale
                                                       # (milli-arcseconds)
                self.UUC = hdul['UV-PLANE'].data['UUC'] # Fourier u-coordinates
                self.VVC = hdul['UV-PLANE'].data['VVC'] # Fourier v-coordinates
                self.KPM = hdul['KER-MAT'].data
            hdul.close()
        self.src_kps = np.concatenate(src_kps)
        self.src_cvs = np.concatenate(src_cvs)
        self.src_pas = np.concatenate(src_pas)
        
        pass
    
    def cal_from_fits(self,
                      idir,
                      kpfiles,
                      flags):
        """
        TODO
        """
        
        # Read calibrator data
        cal_kps = []
        cal_cvs = []
        targets = []
        for i in range(len(kpfiles)):
            hdul = pyfits.open(idir+kpfiles[i], memmap=False)
            if (flags is not None):
                cal_kps += [hdul['KP-DATA'].data[flags[i]]]
                cal_cvs += [hdul['KP-SIGM'].data[flags[i]]]
            else:
                cal_kps += [hdul['KP-DATA'].data]
                cal_cvs += [hdul['KP-SIGM'].data]
            targets += [hdul[0].header['TARGNAME']]
            hdul.close()
#        self.cal_kps = np.concatenate(cal_kps)
#        self.cal_cvs = np.concatenate(cal_cvs)
#        
#        print(np.unique(targets))
#        return kpfiles
        
        # Compute the absolute difference between source and calibrator kernel-
        # phase weighted by its uncertainty for each calibrator separately
        src = np.mean(self.src_kps, axis=0)
        kps = []
        cvs = []
        dev = []
        for i in range(len(cal_kps)):
            kps += [np.mean(cal_kps[i], axis=0)]
            cvs += [np.mean(cal_cvs[i], axis=0)]
            dev += [np.true_divide(np.abs(src-kps[-1]), np.sqrt(np.diag(cvs[-1])))]
        dev = np.array(dev)
        
        # Only use the calibrators for which the sum of the squares of this
        # difference is below 3 times its median
        sqd = np.sum(dev**2, axis=1)
        sqdmed = np.median(sqd[np.isnan(sqd) == False])
#        plt.figure()
#        plt.plot(np.sort(sqd))
#        plt.axhline(3.*sqdmed, color='red')
#        plt.xlabel('calibrator data cube')
#        plt.ylabel('$\sum((\\theta_{src}-\\theta_{cal})^2/\sigma_{cal}^2)$')
#        plt.show()
#        import pdb; pdb.set_trace()
        ww = np.where(sqd < 3.*sqdmed)
        print('--> Using a subset of %.0f calibrator data cubes' % ww[0].shape[0])
        cal_kps = [kps[i] for i in ww[0]]
        cal_cvs = [cvs[i] for i in ww[0]]
        self.cal_kps = np.array(cal_kps)
        self.cal_cvs = np.array(cal_cvs)
        
        print(np.unique([targets[i] for i in ww[0]]))
        return [kpfiles[i] for i in ww[0]] # Only return the calibrators which
                                           # are actually used
    
    def __connect_KL(self,
                     K_klip):
        """
        TODO
        """
        
        # Connect KL calibration
        self.KL = kl_calibration.KL(self.cal_kps,
                                    K_klip=K_klip)
        
        # Project source data into the robustly calibrated sub-space
        self.src_kps_KL = self.KL.project(self.src_kps)
        self.src_cvs_KL = self.KL.project(self.src_cvs)
        
        pass
    
    def src_inverse(self,
                    src_cvs,
                    src_cvs_KL):
        """
        TODO
        """
        
        # Compute inverse kernel-phase covariance
        self.src_icvs = np.zeros(src_cvs.shape)
        for i in range(src_cvs.shape[0]):
            self.src_icvs[i] = np.linalg.inv(src_cvs[i])
        self.src_icvs_KL = np.zeros(src_cvs_KL.shape)
        for i in range(src_cvs_KL.shape[0]):
            self.src_icvs_KL[i] = np.linalg.inv(src_cvs_KL[i])
        
        pass
    
    def __get_name(self,
                   idir,
                   kpfiles,
                   name):
        """
        TODO
        """
        
        # Find object name
        auto_name = []
        for i in range(len(kpfiles)):
            hdul = pyfits.open(idir+kpfiles[i], memmap=False)
            auto_name += [hdul[0].header['OBJECT']]
            hdul.close()
        
        # Check whether all kernel-phase files come from the same object
        if (len(np.unique(auto_name)) == 1):
            auto_name = auto_name[0]+'.fits'
        else:
            raise UserWarning('Source kernel-phase files come from different objects')
        
        # Return name for PyBinary file
        return auto_name
    
    def __make_pybinary_file(self,
                             idir,
                             src_kpfiles,
                             cal_kpfiles):
        """
        TODO
        """
        
        # Read data from all source kernel-phase files
        data = []
        bpmap = []
        nrows = []
        for i in range(len(src_kpfiles)):
            data += [pyfits.getdata(idir+src_kpfiles[i], 0)]
            bpmap += [pyfits.getdata(idir+src_kpfiles[i], 'BP-MAP')]
            nrows += [pyfits.getdata(idir+src_kpfiles[i], 'TEL').shape[0]]
        data = np.concatenate(data)
        bpmap = np.concatenate(bpmap)
        hdul = pyfits.open(idir+src_kpfiles[0], memmap=False)
        tel = pyfits.BinTableHDU.from_columns(hdul['TEL'].columns, nrows=np.sum(nrows))
        for colname in hdul['TEL'].columns.names:
            nr = nrows[0]
            for i in range(1, len(src_kpfiles)):
                temp = pyfits.getdata(idir+src_kpfiles[i], 'TEL')[colname]
                tel.data[colname][nr:nr+nrows[i]] = temp
                nr += nrows[i]
        tel.header['EXTNAME'] = 'TEL'
        
        # Make PyBinary file
        hdul[0].data = data
        hdul[0].header['KKLIP'] = int(self.KL.K_klip)
        for i in range(len(src_kpfiles)):
            hdul[0].header['HISTORY'] = 'Src: '+src_kpfiles[i]
        for i in range(len(cal_kpfiles)):
            hdul[0].header['HISTORY'] = 'Cal: '+cal_kpfiles[i]
        hdul['BP-MAP'].data = bpmap
        hdul['TEL'] = tel
        hdul['KP-DATA'].data = self.src_kps_KL
        hdul['KP-DATA'].header['KKLIP'] = int(self.KL.K_klip)
        hdul['KP-DATA'].header.add_comment('Calibrated kernel-phase')
        hdul['KP-SIGM'].data = self.src_cvs_KL
        hdul['KP-SIGM'].header['KKLIP'] = int(self.KL.K_klip)
        hdul['KP-SIGM'].header.add_comment('Calibrated covariance of kernel-phase')
        try:
            hdul.pop(hdul.index_of('KP-SIGM MC'))
        except:
            pass
        kpi_hdu = pyfits.ImageHDU(self.src_icvs_KL)
        kpi_hdu.header['EXTNAME'] = 'KP-SIGM INV'
        kpi_hdu.header['KKLIP'] = int(self.KL.K_klip)
        kpi_hdu.header.add_comment('Calibrated inverse covariance of kernel-phase')
        hdul.append(kpi_hdu)
        klp_hdu = pyfits.ImageHDU(self.KL.P)
        klp_hdu.header['EXTNAME'] = 'KL-PROJ'
        klp_hdu.header['KKLIP'] = int(self.KL.K_klip)
        klp_hdu.header.add_comment('Karhunen-Loeve projection matrix')
        hdul.append(klp_hdu)
        
        # Save data cube
        hdul.writeto(self.odir+self.name, clobber=True, output_verify='fix') # FIXME
        hdul.close()
        
        pass
    
    def kpm(self,
            p0):
        """
        TODO
        """
        
        # Sky coordinates of secondary
        dra = p0[0]/1000./60./60.*np.pi/180.*np.cos(p0[1]*np.pi/180.)
        ddec = p0[0]/1000./60./60.*np.pi/180.*np.sin(p0[1]*np.pi/180.)
        
        # Fluxes of primary and secondary
        l1 = 1.
        l2 = p0[2]
        
        # Fourier plane signal of binary
        ft = l1+l2*np.exp(-2j*np.pi*(dra*self.UUC/self.cwave+ddec*self.VVC/self.cwave))
        
        # Kernel-phase signal of binary
        kpm = self.KPM.dot(np.angle(ft))
        kpm_KL = self.KL.project(kpm)
        
        # Return kernel-phase signal of binary
        return kpm_KL
    
    def chi2(self,
             res,
             icv):
        """
        TODO
        """
        
        # Compute chi-squared
        chi2 = res.dot(icv).dot(res)
        
        # Return chi-squared
        return chi2
    
    def chi2_raw(self,
                 kps,
                 icvs):
        """
        TODO
        """
        
        # Compute chi-squared
        chi2 = 0.
        for i in range(kps.shape[0]):
            chi2 += kps[i].dot(icvs[i]).dot(kps[i])
        
        # Return chi-squared
        return chi2
    
    def __explore(self,
                  p0):
        """
        TODO
        """
        
        nkps = np.prod(self.src_kps_KL.shape)
        npts = 100
        
        dx0 = float(self.pscale)/2.
        xx0 = np.linspace(p0[0]-dx0, p0[0]+dx0, npts)
        yy0 = []
        pp = p0.copy()
        for i in range(npts):
            pp[0] = xx0[i]
            yy = np.sum(self.chi2_leastsquares(pp,
                                               self.src_kps_KL,
                                               self.src_icvs_KL,
                                               self.src_pas))
            yy0 += [yy/nkps]
        
        dx1 = 10.
        xx1 = np.linspace(p0[1]-dx1, p0[1]+dx1, npts)
        yy1 = []
        pp = p0.copy()
        for i in range(npts):
            pp[1] = xx1[i]
            yy = np.sum(self.chi2_leastsquares(pp,
                                               self.src_kps_KL,
                                               self.src_icvs_KL,
                                               self.src_pas))
            yy1 += [yy/nkps]
            
        dx2 = 5E-3
        xx2 = np.linspace(p0[2]-dx2, p0[2]+dx2, npts)
        yy2 = []
        pp = p0.copy()
        for i in range(npts):
            pp[2] = xx2[i]
            yy = np.sum(self.chi2_leastsquares(pp,
                                               self.src_kps_KL,
                                               self.src_icvs_KL,
                                               self.src_pas))
            yy2 += [yy/nkps]
        
        xx = np.linspace(-1, 1, npts)
        plt.figure()
        plt.plot(xx, yy0)
        plt.plot(xx, yy1)
        plt.plot(xx, yy2)
        plt.show(block=True)
        
        pass
    
    def __make_grid(self,
                    grid_size=500,
                    sampling='pscale/2'):
        """
        TODO
        """
        
        print('--> Making grid')
        
        # Compute grid steps
        if (sampling == '2pscale'):
            step = 2.*float(self.pscale)
        elif (sampling == 'pscale'):
            step = float(self.pscale)
        elif (sampling == 'pscale/2'):
            step = float(self.pscale)/2.
        else:
            raise UserWarning(str(sampling)+' is not a known grid sampling')
        nstep = np.ceil(float(grid_size)/step)
        ramp = np.arange(-(nstep*step), (nstep+1)*step, step)
        
        # Compute grid, separations and position angles
        xy = np.meshgrid(ramp, ramp, indexing='ij')
        self.seps = np.sqrt(xy[0]**2+xy[1]**2)
        self.pas = np.degrees(np.arctan2(xy[0], xy[1]))
        
        # Return grid, separations and position angles
        return xy, self.seps, self.pas
    
    def __gridsearch_cell(self,
                          kps_KL,
                          icvs_KL,
                          pas,
                          p0,
                          method='sim'):
        """
        TODO
        """
        
        # Simultaneous fitting method
        if (method == 'sim'):
            chi2 = 0.
            DKPS = []
            MKPS = []
            MKPS_ICVS = []
            for i in range(kps_KL.shape[0]):
                
                # Rotate back position angle
                pp = p0.copy()
                pp[1] -= pas[i]
                
                # Compute kernel-phase signal of binary
                kpm_KL = self.kpm(pp)
                
                # Compute chi-squared of binary
                chi2 += self.chi2(kps_KL[i]-kpm_KL,
                                  icvs_KL[i])
                
                # Stack data and model kernel-phase
                kpm_KL /= p0[2]
                DKPS += [kps_KL[i]]
                MKPS += [kpm_KL]
                MKPS_ICVS += [kpm_KL.T.dot(icvs_KL[i])]
            
            # Compute best fit contrast and its error
            DKPS = np.array(DKPS).flatten()
            MKPS = np.array(MKPS).flatten()
            MKPS_ICVS = np.array(MKPS_ICVS).flatten()
            c_mean = (MKPS_ICVS.dot(DKPS))/(MKPS_ICVS.dot(MKPS))
            c_sdev = 1./np.sqrt(MKPS_ICVS.dot(MKPS))
        
        # Separate fitting method
        elif (method == 'sep'):
            chi2 = 0.
            c = []
            for i in range(kps_KL.shape[0]):
                
                # Rotate back position angle
                pp = p0.copy()
                pp[1] -= pas[i]
                
                # Compute kernel-phase signal of binary
                kpm_KL = self.kpm(pp)
                
                # Compute chi-squared of binary
                chi2 += self.chi2(kps_KL[i]-kpm_KL,
                                  icvs_KL[i])
                
                # Compute best fit contrast for each frame separately
                kpm_KL /= p0[2]
                c += [(kpm_KL.T.dot(icvs_KL[i]).dot(kps_KL[i]))/(kpm_KL.T.dot(icvs_KL[i]).dot(kpm_KL))]
            
            # Compute best fit contrast and its error
            c = np.array(c)
            c_mean = np.mean(c)
            c_sdev = np.sqrt(np.sum(np.abs(c-c_mean)**2)/(c.shape[0]-1.))/np.sqrt(c.shape[0]) # Standard deviation of the mean
        
        # Unknown fitting method
        else:
            raise UserWarning(str(method)+' is not a known method')
        
        # Compute chi-squared for best fit contrast
        chi2_c = 0.
        p0[2] = c_mean
        for i in range(kps_KL.shape[0]):
            
            # Rotate back position angle
            pp = p0.copy()
            pp[1] -= pas[i]
            
            # Compute kernel-phase signal of binary
            kpm_KL = self.kpm(pp)
            
            # Compute chi-squared of binary
            chi2_c += self.chi2(kps_KL[i]-kpm_KL,
                              icvs_KL[i])
        
        # Return results from the fitting
        return c_mean, c_sdev, chi2, chi2_c
    
    def gridsearch(self,
                   grid_size=500,
                   sampling='pscale/2',
                   method='sim',
                   multiklip=False):
        """
        TODO
        """
        
        print('--> Performing gridsearch')
        
        # Compute grid, separations and position angles
        _, seps, pas = self.__make_grid(grid_size=grid_size,
                                        sampling=sampling)
        
        #
        self.c0 = 0.001 # Reference contrast which is small enough to be in the
                        # linear regime
        cs_mean = []
        cs_sdev = []
        chi2s = []
        chi2s_c = []
        counter = 1
        nc = np.prod(seps.shape) # Number of grid cells
        
        #
        for sep, pa in zip(seps.flatten(), pas.flatten()):
            sys.stdout.write('\rsep=%.2f, pa=%.2f, cell %.0f of %.0f' % (sep, pa, counter, nc))
            sys.stdout.flush()
            
            # Compute results from the fitting
            p0 = np.array([sep, pa, self.c0]) # Current grid cell
            c_mean, c_sdev, chi2, chi2_c = self.__gridsearch_cell(self.src_kps_KL,
                                                                  self.src_icvs_KL,
                                                                  self.src_pas,
                                                                  p0,
                                                                  method=method)
            cs_mean += [c_mean]
            cs_sdev += [c_sdev]
            chi2s += [chi2]
            chi2s_c += [chi2_c]
            counter += 1
        print('')
        
        # Reshape results from the fitting into grids
        cs_mean = np.array(cs_mean).reshape(seps.shape)
        cs_sdev = np.array(cs_sdev).reshape(seps.shape)
        chi2s = np.array(chi2s).reshape(seps.shape)
        chi2s_c = np.array(chi2s_c).reshape(seps.shape)
        
        # Compute reduced chi-squaredes
        red_chi2s = chi2s/np.prod(self.src_kps_KL.shape)
        red_chi2s_c = chi2s_c/np.prod(self.src_kps_KL.shape)
        
        # Find best fit grid position
        temp = red_chi2s_c.copy()
        temp[np.isnan(temp)] = np.inf # Bad pixels are not the minimum
        temp[cs_mean < 0.] = np.inf # Pixels with negative contrasts are not
                                    # the minimum
        best = np.unravel_index(np.argmin(temp), temp.shape)
        red_chi2_raw = self.chi2_raw(self.src_kps_KL,
                                     self.src_icvs_KL)/np.prod(self.src_kps_KL.shape)
        red_chi2_bin = red_chi2s_c[best]
        
        # Print best fit parameters
        print('\tBest reduced chi2: %.3f' % red_chi2_bin)
        print('\tBest separation in mas: %.3f' % seps[best])
        print('\tBest position angle in deg: %.3f' % pas[best])
        print('\tBest contrast secondary/primary: %.6f' % cs_mean[best])
        
        # Prepare grid search HDU
        self.gds = [cs_mean, cs_sdev, red_chi2s, red_chi2s_c, seps, pas]
        self.gds = np.array(self.gds)
        self.p0 = np.array([seps[best], pas[best], cs_mean[best]])
        self.p0 = np.append(self.p0, np.array([red_chi2_raw, red_chi2_bin]))
        
        # Save grid search HDU to PyBinary file
        if (multiklip == False):
            gds_hdu = pyfits.ImageHDU(self.gds)
            gds_hdu.header['EXTNAME'] = 'GRIDSEARCH'
            gds_hdu.header['KKLIP'] = int(self.KL.K_klip)
            gds_hdu.header['C0'] = float(self.c0)
            gds_hdu.header.add_comment('Frame 0: best fit contrasts')
            gds_hdu.header.add_comment('Frame 1: uncertainty of best fit contrasts')
            gds_hdu.header.add_comment('Frame 2: reduced chi-squareds, c = '+str(self.c0))
            gds_hdu.header.add_comment('Frame 3: reduced chi-squareds, c = best fit contrast')
            gds_hdu.header.add_comment('Frame 4: separations of grid (mas)')
            gds_hdu.header.add_comment('Frame 5: position angles of grid (deg)')
            p0f_hdu = pyfits.ImageHDU(self.p0)
            p0f_hdu.header['EXTNAME'] = 'P0'
            p0f_hdu.header.add_comment('Posterior from grid search')
            hdul = pyfits.open(self.odir+self.name, mode='update')
            try:
                test = hdul['P0']
                hdul.pop(hdul.index_of('GRIDSEARCH'))
                hdul.pop(hdul.index_of('P0'))
            except:
                pass
            hdul.append(gds_hdu)
            hdul.append(p0f_hdu)
            hdul.flush()
            hdul.close()
            
            print('--------------------')
            
            pass
        
        # Return grid search HDU
        else:
            return self.gds, self.p0
    
    def chi2_leastsquares(self,
                          p0,
                          kps_KL,
                          icvs_KL,
                          pas):
        """
        TODO
        """
        
        #
        chi2 = np.zeros(kps_KL.shape[0])
        for i in range(kps_KL.shape[0]):
            
            # Rotate back position angle
            pp = p0.copy()
            pp[1] -= pas[i]
            
            # Compute kernel-phase signal of binary
            kpm_KL = self.kpm(pp)
            
            # Compute chi-squared of binary
            chi2[i] = self.chi2(kps_KL[i]-kpm_KL,
                                icvs_KL[i])
        
        # Return chi-squared of binary
        return chi2
    
    def chi2_leastsquares_bounded(self,
                                  p0,
                                  rho_min,
                                  kps_KL,
                                  icvs_KL,
                                  pas):
        """
        TODO
        """
        
        #
        chi2 = np.zeros(kps_KL.shape[0])
        for i in range(kps_KL.shape[0]):
            
            # Rotate back position angle
            pp = np.array([rho_min, p0[0], p0[1]])
            pp[1] -= pas[i]
            
            # Compute kernel-phase signal of binary
            kpm_KL = self.kpm(pp)
            
            # Compute chi-squared of binary
            chi2[i] = self.chi2(kps_KL[i]-kpm_KL,
                                icvs_KL[i])
        
        # Return chi-squared of binary
        return chi2
    
    def chi2_minimize(self,
                      p0,
                      kps_KL,
                      icvs_KL,
                      pas):
        """
        TODO
        """
        
        #
        chi2 = np.zeros(kps_KL.shape[0])
        for i in range(kps_KL.shape[0]):
            
            # Rotate back position angle
            pp = p0.copy()
            pp[1] -= pas[i]
            
            # Compute kernel-phase signal of binary
            kpm_KL = self.kpm(pp)
            
            # Compute chi-squared of binary
            chi2[i] = self.chi2(kps_KL[i]-kpm_KL,
                                icvs_KL[i])
        
        # Return chi-squared of binary
        return np.sum(chi2)
    
    def leastsquares(self,
                     p0=None,
                     rho_min=None,
                     multiklip=False):
        """
        TODO
        """
        
        print('--> Performing leastsquares')
        
        # Get the prior from the grid search
        if (p0 is None):
            p0 = self.p0[:3]
#        self.__explore(p0)
        
#        # Firstly, perform the least squares without any boundary condition
#        ls = leastsq(self.chi2_leastsquares,
#                     p0,
#                     args=(self.src_kps_KL, self.src_icvs_KL, self.src_pas),
#                     ftol=1E-10,
#                     xtol=1E-10,
#                     full_output=True)
#        self.s0 = ls[0]
#        
#        # Secondly, perform least squares with the boundary condition in case
#        # there is given one and the unbounded solution lies outside it
#        if (rho_min is not None):
#            if (self.s0[0] < rho_min):
#                print('Now fixing the separation to %.0f mas' % rho_min)
#                ls = leastsq(self.chi2_leastsquares_bounded,
#                             np.array([p0[1], p0[2]]),
#                             args=(rho_min, self.src_kps_KL, self.src_icvs_KL, self.src_pas),
#                             ftol=1E-10,
#                             xtol=1E-10,
#                             full_output=True)
#                self.s0 = np.array([rho_min, ls[0][0], ls[0][1]])
        
        # Another least squares algorithm which can handle boundary conditions,
        # but performs very poorly compared to the other one
        if (rho_min is not None):
            bounds = [(rho_min, np.inf), (-180., 180.), (0., 1.)]
        else:
            bounds = [(0., np.inf), (-180., 180.), (0., 1.)]
        ls = minimize(self.chi2_minimize,
                      p0,
                      args=(self.src_kps_KL, self.src_icvs_KL, self.src_pas),
                      bounds=bounds)
        self.s0 = ls['x']
#        self.__explore(self.s0)
        
        # Compute reduced chi-squaredes
        red_chi2_raw = self.chi2_raw(self.src_kps_KL,
                                     self.src_icvs_KL)/np.prod(self.src_kps_KL.shape)
#        red_chi2_bin = np.sum(ls[2]['fvec'])/np.prod(self.src_kps_KL.shape)
        red_chi2_bin = ls['fun']/np.prod(self.src_kps_KL.shape)
        self.s0 = np.append(self.s0, np.array([red_chi2_raw, red_chi2_bin]))
        
        # Print best fit parameters
        print('\tBest reduced chi2: %.3f' % self.s0[4])
        print('\tBest separation in mas: %.3f' % self.s0[0])
        print('\tBest position angle in deg: %.3f' % self.s0[1])
        print('\tBest contrast secondary/primary: %.6f' % self.s0[2])
        
        # Save least squares HDU to PyBinary file
        if (multiklip == False):
            s0f_hdu = pyfits.ImageHDU(self.s0)
            s0f_hdu.header['EXTNAME'] = 'S0'
            s0f_hdu.header.add_comment('Posterior from least squares')
            hdul = pyfits.open(self.odir+self.name, mode='update')
            try:
                test = hdul['S0']
                hdul.pop(hdul.index_of('S0'))
            except:
                pass
            hdul.append(s0f_hdu)
            hdul.flush()
            hdul.close()
            
            print('--------------------')
            
            pass
        
        # Return least squares HDU
        else:
            return self.s0
    
    def multiklip(self,
                  K_klip=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                  grid_size=500,
                  sampling='pscale/2',
                  method='sim',
                  rho_min=None):
        """
        TODO
        """
        
        print('--> Performing multiklip')
        
        # Save the previous K_klip so that it can be restored later
        temp = self.KL.K_klip
        
        #
        nk = len(K_klip)
        cs_mean = []
        cs_sdev = []
        red_chi2s = []
        red_chi2s_c = []
        seps = []
        pas = []
        p0s = []
        s0s = []
        for i in range(nk):
            
            # Apply the current K_klip
            print('--> K_klip = %.0f' % K_klip[i])
            self.__connect_KL(K_klip=K_klip[i])
            self.src_inverse(self.src_cvs,
                             self.src_cvs_KL)
            
            # Perform grid search and least squares for the current K_klip
            gds, p0 = self.gridsearch(grid_size=grid_size,
                                      sampling=sampling,
                                      method=method,
                                      multiklip=True)
            s0 = self.leastsquares(p0=p0[:3],
                                   rho_min=rho_min,
                                   multiklip=True)
            cs_mean += [gds[0]]
            cs_sdev += [gds[1]]
            red_chi2s += [gds[2]]
            red_chi2s_c += [gds[3]]
            seps += [gds[4]]
            pas += [gds[5]]
            p0s += [p0]
            s0s += [s0]
        cs_mean = np.array(cs_mean)
        cs_sdev = np.array(cs_sdev)
        red_chi2s = np.array(red_chi2s)
        red_chi2s_c = np.array(red_chi2s_c)
        seps = np.array(seps)
        pas = np.array(pas)
        p0s = np.array(p0s)
        s0s = np.array(s0s)
        
        # Prepare multiklip HDU
        self.mkl = [cs_mean, cs_sdev, red_chi2s, red_chi2s_c, seps, pas]
        self.mkl = np.array(self.mkl)
        self.p0s = p0s
        self.s0s = s0s
        
        # Save multiklip HDU to PyBinary file
        mkl_hdu = pyfits.ImageHDU(self.mkl)
        mkl_hdu.header['EXTNAME'] = 'MULTIKLIP'
        mkl_hdu.header['KKLIP'] = str(K_klip)
        mkl_hdu.header['C0'] = float(self.c0)
        mkl_hdu.header.add_comment('Frame 0: best fit contrasts')
        mkl_hdu.header.add_comment('Frame 1: uncertainty of best fit contrasts')
        mkl_hdu.header.add_comment('Frame 2: reduced chi-squareds, c = '+str(self.c0))
        mkl_hdu.header.add_comment('Frame 3: reduced chi-squareds, c = best fit contrast')
        mkl_hdu.header.add_comment('Frame 4: separations of grid (mas)')
        mkl_hdu.header.add_comment('Frame 5: position angles of grid (deg)')
        p0s_hdu = pyfits.ImageHDU(self.p0s)
        p0s_hdu.header['EXTNAME'] = 'P0S'
        p0s_hdu.header.add_comment('Posterior from grid search')
        s0s_hdu = pyfits.ImageHDU(self.s0s)
        s0s_hdu.header['EXTNAME'] = 'S0S'
        s0s_hdu.header.add_comment('Posterior from least squares')
        hdul = pyfits.open(self.odir+self.name, mode='update')
        try:
            test = hdul['P0S']
            hdul.pop(hdul.index_of('MULTIKLIP'))
            hdul.pop(hdul.index_of('P0S'))
            hdul.pop(hdul.index_of('S0S'))
        except:
            pass
        hdul.append(mkl_hdu)
        hdul.append(p0s_hdu)
        hdul.append(s0s_hdu)
        hdul.flush()
        hdul.close()
        
        print('--------------------')
        
        # Restore the previous K_klip
        print('--> Restoring the previous K_klip = %.0f' % temp)
        self.__connect_KL(K_klip=temp)
        self.src_inverse(self.src_cvs,
                         self.src_cvs_KL)
        
        pass
    
    def multiklip_sub(self,
                      K_klip=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                      grid_size=500,
                      sampling='pscale/2',
                      method='sim',
                      rho_min=None):
        """
        TODO
        """
        
        self.multiklip(K_klip=K_klip,
                       grid_size=grid_size,
                       sampling=sampling,
                       method=method,
                       rho_min=rho_min)
        
        print('--> Performing multiklip sub')
        
        # Save the previous K_klip so that it can be restored later
        temp = self.KL.K_klip
        
        #
        nk = len(K_klip)
        cs_mean = []
        cs_sdev = []
        red_chi2s = []
        red_chi2s_c = []
        seps = []
        pas = []
        
        # Get the best fit binary model parameters
        hdul = pyfits.open(self.odir+self.name, mode='update')
        s0 = hdul['S0S'].data # from the least squares
        
        for i in range(nk):
            
            # Apply the current K_klip
            print('--> K_klip = %.0f' % K_klip[i])
            self.__connect_KL(K_klip=K_klip[i])
            self.src_inverse(self.src_cvs,
                             self.src_cvs_KL)
            
            # Subtract the best fit binary model from the data, the covariance
            # remains the same
            for j in range(self.src_kps_KL.shape[0]):
                ss = s0[i][:3].copy()
                ss[1] -= self.src_pas[j]
                self.src_kps_KL[j] -= self.kpm(ss)
            
            # Perform grid search for the current K_klip
            gds, _ = self.gridsearch(grid_size=grid_size,
                                     sampling=sampling,
                                     method=method,
                                     multiklip=True)
            cs_mean += [gds[0]]
            cs_sdev += [gds[1]]
            red_chi2s += [gds[2]]
            red_chi2s_c += [gds[3]]
            seps += [gds[4]]
            pas += [gds[5]]
        cs_mean = np.array(cs_mean)
        cs_sdev = np.array(cs_sdev)
        red_chi2s = np.array(red_chi2s)
        red_chi2s_c = np.array(red_chi2s_c)
        seps = np.array(seps)
        pas = np.array(pas)
        
        # Prepare multiklip sub HDU
        self.mkl_sub = [cs_mean, cs_sdev, red_chi2s, red_chi2s_c, seps, pas]
        self.mkl_sub = np.array(self.mkl_sub)
        
        # Save multiklip sub HDU to PyBinary file
        mkl_sub_hdu = pyfits.ImageHDU(self.mkl_sub)
        mkl_sub_hdu.header['EXTNAME'] = 'MULTIKLIP_SUB'
        mkl_sub_hdu.header['KKLIP'] = str(K_klip)
        mkl_sub_hdu.header['C0'] = float(self.c0)
        mkl_sub_hdu.header.add_comment('Frame 0: best fit contrasts')
        mkl_sub_hdu.header.add_comment('Frame 1: uncertainty of best fit contrasts')
        mkl_sub_hdu.header.add_comment('Frame 2: reduced chi-squareds, c = '+str(self.c0))
        mkl_sub_hdu.header.add_comment('Frame 3: reduced chi-squareds, c = best fit contrast')
        mkl_sub_hdu.header.add_comment('Frame 4: separations of grid (mas)')
        mkl_sub_hdu.header.add_comment('Frame 5: position angles of grid (deg)')
        try:
            test = hdul['MULTIKLIP_SUB']
            hdul.pop(hdul.index_of('MULTIKLIP_SUB'))
        except:
            pass
        hdul.append(mkl_sub_hdu)
        hdul.flush()
        hdul.close()
        
        print('--------------------')
        
        # Restore the previous K_klip
        print('--> Restoring the previous K_klip = %.0f' % temp)
        self.__connect_KL(K_klip=temp)
        self.src_inverse(self.src_cvs,
                         self.src_cvs_KL)
        
        pass
