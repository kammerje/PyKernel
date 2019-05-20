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


# MAIN
#==============================================================================
class KL():
    
    def __init__(self,
                 cal_kps,
                 K_klip=4):
        """
        Class for projecting the kernel-phase and its covariance into a
        robustly calibrated sub-space based on a Karhunen-Loeve decomposition.
        
        Parameters
        ----------
        cal_kps : 2D array of calibrator kernel-phases, first axis is number of
                  calibrator observations and second axis is number of kernel-
                  phases.
        K_klip : Integer representing the order of the calibration, i.e. how
                 many principal components of the calibrator kernel-phases
                 shall be accounted for.
        """
        
        # Read input parameters
        self.ncal = cal_kps.shape[0] # Number of calibrator observations
        self.K_klip = int(K_klip)
        if (self.K_klip > self.ncal):
            raise ValueError('K_klip cannot be larger than the number of calibrator observations')
        
        # Compute projection matrix
        self.decompose(cal_kps)
        
        pass
    
    def decompose(self,
                  cal_kps):
        """
        Function for computing the Karhunen-Loeve decomposition and the
        projection matrix.
        
        Parameters
        ----------
        cal_kps : 2D array of calibrator kernel-phases, first axis is number of
                  calibrator observations and second axis is number of kernel-
                  phases.
        """
        
        print('--> Computing projection matrix')
        
        # Compute covariance matrix (equation 6 of Soummer et al. 2012)
#        E_RR = np.zeros((self.ncal, self.ncal))
#        for i in range(self.ncal):
#            for j in range(self.ncal):
#                E_RR[i, j] = cal_kps[i].dot(cal_kps[j])
        E_RR = cal_kps.dot(cal_kps.T)
        
        # Compute eigenvalues and eigenvectors of covariance matrix
#        w, v = np.linalg.eig(E_RR)
#        v_sort = np.zeros(v.shape)
#        temp = np.argsort(w)[::-1]
#        for i in range(len(w)):
#            v_sort[:, i] = v[:, temp[i]]
#        w_sort = np.sort(w)[::-1]
        w, v = np.linalg.eigh(E_RR)
        v_sort = np.zeros(v.shape)
        temp = np.argsort(w)[::-1]
        for i in range(len(w)):
            v_sort[:, i] = v[:, temp[i]]
        w_sort = np.sort(w)[::-1]
        
        # Compute Karhunen-Loeve transform (equation 5 of Soummer et al. 2012)
#        self.Z_KL = np.zeros((cal_kps.shape[1], self.ncal))
#        for n in range(cal_kps.shape[1]):
#            for k in range(self.ncal):
#                for p in range(self.ncal):
#                    self.Z_KL[n, k] += v_sort[p, k]*cal_kps[p, n]
#                self.Z_KL[n, k] *= 1./np.sqrt(w_sort[k])
        v_norm = np.divide(v_sort, np.sqrt(w_sort))
        self.Z_KL = cal_kps.T.dot(v_norm)
        
        # Compute projection matrix
        Z_prime = self.Z_KL[:, :self.K_klip]
        P = np.identity(Z_prime.shape[0])-Z_prime.dot(Z_prime.T)
        
        # Reduce dimension of projection matrix
        w, v = np.linalg.eigh(P)
        self.P = v[np.where(w > 1E-10)[0]].dot(np.diag(w)).dot(v.T)
        
        print('--------------------')
        
        pass
    
    def project(self,
                arr_in):
        """
        Function for projecting the kernel-phase and its covariance into a
        robustly calibrated sub-space.
        
        Parameters
        ----------
        arr_in : Input array containing the data to be calibrated.
                 Possible shapes are 1D/2D (kernel-phase) and 3D (kernel-phase
                 covariance).
        """
        
        #
        ndim = len(arr_in.shape)
        
        # Project input array into the robustly calibrated sub-space
        if (ndim == 1):
            arr_out = self.P.dot(arr_in)
        elif (ndim == 2):
            arr_out = (self.P.dot(arr_in.T)).T
        elif (ndim == 3):
            arr_out = np.zeros((arr_in.shape[0], self.P.shape[0], self.P.shape[0]))
            for i in range(arr_out.shape[0]):
                arr_out[i] = self.P.dot(arr_in[i]).dot(self.P.T)
        else:
            raise UserWarning('Input shape not supported')
        
        # Return array projected into the robustly calibrated sub-space
        return arr_out
