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
class PyKernel():
    
    def __init__(self,
                 idir='',
                 odir='',
                 tel='VLT',
                 pupil_rot=None,
                 make_pupil=True,
                 grid_size=1024,
                 sampling=0.6,
                 bmax=7.):
        """
        Class for extracting kernel-phases and computing kernel-phase
        covariances. These will be appended to the original fits file according
        to the kernel-phase data exchange format.
        
        Parameters
        ----------
        idir : Input directory where the fits files, from which the kernel-
               phase shall be extracted, are located.
        odir : Output directory where the kernel-phase fits files and the pupil
               model will be saved to.
        tel : Telescope from which the data comes.
              Possible options are 'VLT', 'Keck'.
        pupil_rot : Angle (counter-clockwise) in degrees by which the pupil
                    model shall be rotated. If 'None', no rotation will be
                    applied.
        make_pupil : If 'True', generates a new pupil model. If 'False', tries
                     to use a previously computed pupil model located in odir.
        grid_size : Size of the grid in pixels on which the pupil model is
                    computed.
        sampling : Sampling of the grid in meters on which the sub-apertures
                   are distributed.
        bmax : Maximal baseline length in meters which is kept in the pupil
               model.
        """
        
        # Read input parameters
        self.idir = idir
        self.odir = odir
        self.tel = tel
        
        # Make pupil model
        if (make_pupil):
            _, fitspath = self.make_pupil_model(tel=self.tel,
                                                pupil_rot=pupil_rot,
                                                grid_size=grid_size,
                                                sampling=sampling,
                                                bmax=bmax)
        elif (self.tel == 'VLT'):
            _, fitspath = self.odir+'vlt.txt', self.odir+'vlt.fits'
        elif (self.tel == 'Keck'):
            _, fitspath = self.odir+'keck.txt', self.odir+'keck.fits'
        else:
            raise UserWarning(str(self.tel)+' is not a known telescope')
        
        # Connect XARA
        self.__connect_XARA(fitspath,
                            bmax=bmax)
        
        pass
    
    def __connect_XARA(self,
                       fitspath='vlt.fits',
                       bmax=7.):
        """
        Function for connecting the XARA package.
        
        Parameters
        ----------
        fitspath : Path to the fits file of the pupil model (relative to odir)
                   which shall be used.
        bmax : Maximal baseline length in meters which is kept in the pupil
               model.
        """
        
        # Connect KPI and KPO
        self.__connect_KPI(fitspath=fitspath,
                           bmax=bmax)
        self.__connect_KPO(fitspath=fitspath)
        
        pass
    
    def __connect_KPI(self,
                      fitspath='vlt.fits',
                      bmax=7.):
        """
        Function for connecting the KPI class.
        
        Parameters
        ----------
        fitspath : Path to the fits file of the pupil model (relative to odir)
                   which shall be used.
        bmax : Maximal baseline length in meters which is kept in the pupil
               model.
        """
        
        print('--> Connecting KPI')
        
        # Connect KPI
        self.KPI = xara.KPI(fname=fitspath,
                            bmax=bmax)
        
        print('--------------------')
        
        pass
    
    def __connect_KPO(self,
                      fitspath='vlt.fits'):
        """
        Function for connecting the KPO class.
        
        Parameters
        ----------
        fitspath : Path to the fits file of the pupil model (relative to odir)
                   which shall be used.
        """
        
        print('--> Connecting KPO')
        
        # Connect KPO
        self.KPO = xara.KPO(fname=fitspath)
        
        print('--------------------')
        
        pass
    
    def make_pupil_model(self,
                         tel='VLT',
                         pupil_rot=None,
                         grid_size=1024,
                         sampling=0.6,
                         bmax=7.):
        """
        Function for making a pupil model which can be read by the XARA
        package.
        
        Parameters
        ----------
        tel : Telescope from which the data comes.
              Possible options are 'VLT', 'Keck'.
        pupil_rot : Angle (counter-clockwise) in degrees by which the pupil
                    model shall be rotated. If 'None', no rotation will be
                    applied.
        grid_size : Size of the grid in pixels on which the pupil model is
                    computed.
        sampling : Sampling of the grid in meters on which the sub-apertures
                   are distributed.
        bmax : Maximal baseline length in meters which is kept in the pupil
               model.
        """
        
        # Make pupil model
        if (tel == 'VLT'):
            txtpath, fitspath = self.__make_pupil_model_VLT(pupil_rot=pupil_rot,
                                                            grid_size=grid_size,
                                                            sampling=sampling,
                                                            bmax=bmax)
        elif (tel == 'Keck'):
            txtpath, fitspath = self.__make_pupil_model_Keck(pupil_rot=pupil_rot,
                                                             grid_size=grid_size,
                                                             sampling=sampling,
                                                             bmax=bmax)
        else:
            raise UserWarning(str(tel)+' is not a known telescope')
        
        # Return paths to the text file and the fits file of the pupil model
        return txtpath, fitspath
    
    def __make_pupil_model_VLT(self,
                               pupil_rot=None,
                               grid_size=1024,
                               sampling=0.6,
                               bmax=7.):
        """
        Function for making a pupil model for the VLT.
        
        Parameters
        ----------
        pupil_rot : Angle (counter-clockwise) in degrees by which the pupil
                    model shall be rotated. If 'None', no rotation will be
                    applied.
        grid_size : Size of the grid in pixels on which the pupil model is
                    computed.
        sampling : Sampling of the grid in meters on which the sub-apertures
                   are distributed.
        bmax : Maximal baseline length in meters which is kept in the pupil
               model.
        """
        
        print('--> Making VLT pupil model')
        
        # Paths to which the text file and the fits file of the pupil model
        # will be saved
        txtpath = self.odir+'vlt.txt'
        fitspath = self.odir+'vlt.fits'
        
        # Telescope dimensions
        pri_mirror_size = 8.2 # Primary mirror size (meters)
                              # Frantz's xaosim uses 8.00
        sec_mirror_size = 1.2 # Secondary mirror size (meters)
                              # Frantz's xaosim uses 1.12
        
        # 1 meter equals ratio pixels on the grid
        ratio = (grid_size-64)/pri_mirror_size # Leave margin of 32 pixels on
                                               # the edge of the grid
        
        # Make pupil model
        pupil = ot.circle(grid_size, pri_mirror_size*ratio)
        pupil -= ot.circle(grid_size, sec_mirror_size*ratio)
        
        # Open the text file for the pupil model coordinates
        txtfile = open(txtpath, 'w')
        
        # Array for plotting the apertures
        dummy = np.zeros((grid_size, grid_size))
        
        # Aperture dimensions
        apa = np.floor(pri_mirror_size/sampling) # Number of apertures sampled
                                                 # per axis
        asz = int(grid_size/64.) # Aperture size is 1/64 of grid size
        
        # Compute aperture coordinates
        if (pupil_rot is not None):
            theta = np.deg2rad(pupil_rot)
            rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        if (apa % 2 == 0):
            for i in range(-int(apa/2), int(apa/2)):
                for j in range(-int(apa/2), int(apa/2)):
                    aptr = ot.circle(grid_size, asz)
                    aptr = np.roll(aptr, int((i+0.5)*sampling*ratio), axis=0)
                    aptr = np.roll(aptr, int((j+0.5)*sampling*ratio), axis=1)
                    
                    # Check whether the aperture lies inside the pupil model
                    if (np.min(pupil[aptr == 1]) == 1):
                        xaptr = int(i+0.5)*sampling
                        yaptr = int(j+0.5)*sampling
                        
                        # Rotate pupil
                        if (pupil_rot is not None):
                            temp = rot.dot(np.array([xaptr, yaptr]))
                            xaptr = temp[0]
                            yaptr = temp[1]
                        txtfile.write('%+.10f %+.10f 1.0\n' % (xaptr, yaptr))
                        dummy[aptr == 1] = 1
        else:
            for i in range(-int(apa/2), int(apa/2)+1):
                for j in range(-int(apa/2), int(apa/2)+1):
                    aptr = ot.circle(grid_size, asz)
                    aptr = np.roll(aptr, int(i*sampling*ratio), axis=0)
                    aptr = np.roll(aptr, int(j*sampling*ratio), axis=1)
                    
                    # Check whether the aperture lies inside the pupil model
                    if (np.min(pupil[aptr == 1]) == 1):
                        xaptr = int(i)*sampling
                        yaptr = int(j)*sampling
                        
                        # Rotate pupil
                        if (pupil_rot is not None):
                            temp = rot.dot(np.array([xaptr, yaptr]))
                            xaptr = temp[0]
                            yaptr = temp[1]
                        txtfile.write('%+.10f %+.10f 1.0\n' % (xaptr, yaptr))
                        dummy[aptr == 1] = 1
        
        # Close the text file for the pupil model coordinates
        txtfile.close()
        
        # Plot pupil model and apertures
        plt.figure()
        plt.imshow(pupil+dummy)
        plt.show()
        plt.close()
        
        # Load pupil model into XARA and save it as fits file
        KPI = xara.KPI(fname=txtpath,
                       bmax=bmax)
        KPI.package_as_fits(fname=fitspath)
        
        # Plot pupil model and apertures
        f = KPI.plot_pupil_and_uv()
        plt.show()
        plt.close()
        
        print('--------------------')
        
        # Return paths to the text file and the fits file of the pupil model
        return txtpath, fitspath
    
    def __make_pupil_model_Keck(self,
                                pupil_rot=None,
                                grid_size=1024,
                                sampling=0.4,
                                bmax=None):
        """
        Function for making a pupil model for the Keck telescope.
        
        Parameters
        ----------
        pupil_rot : Angle (counter-clockwise) in degrees by which the pupil
                    model shall be rotated. If 'None', no rotation will be
                    applied.
        grid_size : Size of the grid in pixels on which the pupil model is
                    computed.
        sampling : Sampling of the grid in meters on which the sub-apertures
                   are distributed.
        bmax : Maximal baseline length in meters which is kept in the pupil
               model.
        """
        
        print('--> Making Keck pupil model')
        
        # Paths to which the text file and the fits file of the pupil model
        # will be saved
        txtpath = self.odir+'keck.txt'
        fitspath = self.odir+'keck.fits'
        
        # Telescope dimensions
        D_pri = 10.95 # Long diagonal of the primary mirror (meters)
        d_pri = D_pri*np.sqrt(3)/2. # Short diagonal of the primary mirror
                                    # (meters)
        D_sec = 2.00 # Long diagonal of the secondary mirror (meters)
                     # 1.40 according to https://www2.keck.hawaii.edu/optics/teldocs/prescriptions.html
                     # 2.00 guessed for the central obscuration in the old
                     # pynrm code
        d_sec = D_sec*np.sqrt(3)/2. # Short diagonal of the secondary mirror
                                    # (meters)
        D_cen = 2.65 # Diameter of the central obscuration (meters)
                     # According to http://www.oir.caltech.edu/twiki_oir/pub/Keck/NGAO/NotesKeckPSF/KeckPupil_Notes.png
                     # the Keck pupil has a circular central obscuration and
                     # six 25.4 mm wide spiders
        D_hex = 1.8 # Long diagonal of an individual mirror segment
        d_hex = D_hex*np.sqrt(3)/2. # Short diagonal of an individual mirror
                                    # segment
        s_width = 0.0254/2. # Half width of the spiders
        
        # 1 meter equals ratio pixels on the grid
        ratio = (grid_size-64)/D_pri # Leave safety margin of 32 pixels on the
                                     # edge of the grid
        
#        # Make pupil model
#        # Simple model with a hexagonal primary mirror and a hexagonal or
#        # circular central obscuartion
#        pupil = ot.hexagon(grid_size, d_pri*ratio)
#        pupil -= ot.hexagon(grid_size, d_sec*ratio) # Hexagonal central
#                                                    # obscuration
#        pupil -= ot.circle(grid_size, D_cen*ratio) # Circular central
#                                                   # obscuration
#        # Advanced model with a primary mirror consisting of 37 hexagons
#        x_roll = [0, 0, 0, 0, 0, 0, 0, -0.75*D_hex, -0.75*D_hex, -0.75*D_hex, -0.75*D_hex, -0.75*D_hex, -0.75*D_hex, 0.75*D_hex, 0.75*D_hex, 0.75*D_hex, 0.75*D_hex, 0.75*D_hex, 0.75*D_hex, -1.5*D_hex, -1.5*D_hex, -1.5*D_hex, -1.5*D_hex, -1.5*D_hex, 1.5*D_hex, 1.5*D_hex, 1.5*D_hex, 1.5*D_hex, 1.5*D_hex, -2.25*D_hex, -2.25*D_hex, -2.25*D_hex, -2.25*D_hex, 2.25*D_hex, 2.25*D_hex, 2.25*D_hex, 2.25*D_hex]
#        y_roll = [0, -d_hex, d_hex, -2.*d_hex, 2.*d_hex, -3.*d_hex, 3.*d_hex, -0.5*d_hex, 0.5*d_hex, -1.5*d_hex, 1.5*d_hex, -2.5*d_hex, 2.5*d_hex, -0.5*d_hex, 0.5*d_hex, -1.5*d_hex, 1.5*d_hex, -2.5*d_hex, 2.5*d_hex, 0, -d_hex, d_hex, -2.*d_hex, 2.*d_hex, 0, -d_hex, d_hex, -2.*d_hex, 2.*d_hex, -0.5*d_hex, 0.5*d_hex, -1.5*d_hex, 1.5*d_hex, -0.5*d_hex, 0.5*d_hex, -1.5*d_hex, 1.5*d_hex]
#        pupil = np.zeros((grid_size, grid_size))
#        segment = ot.hexagon(grid_size, d_hex*ratio)
#        for i in range(len(x_roll)):
#            pupil += np.roll(np.roll(segment, int(x_roll[i]*ratio), axis=0), int(y_roll[i]*ratio), axis=1)
#        pupil = (pupil > 0.5).astype(float)
#        pupil -= ot.circle(grid_size, D_cen*ratio) # Circular central
#                                                   # obscuration
#        
#        spider = np.zeros((grid_size, grid_size))
#        spider[int(grid_size/2.):, int(grid_size/2.-s_width*ratio):int(grid_size/2.+s_width*ratio)] = 1.
#        for i in range(6):
#            pupil -= rotate(spider, i*60, reshape=False) > 0.5
#        pupil = (pupil > 0.5).astype(float)
#        
#        # Open the text file for the pupil model coordinates
#        txtfile = open(txtpath, 'w')
#        
#        # Array for plotting the apertures
#        dummy = np.zeros((grid_size, grid_size))
#        
#        # Aperture dimensions
#        apa = np.floor(D_pri/sampling) # Number of apertures sampled per axis
#        asz = int(grid_size/64.) # Aperture size is 1/64 of grid size
#        
#        # Compute aperture coordinates
#        if (pupil_rot is not None):
#            theta = pupil_rot*np.pi/180.
#            rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
#        if (apa % 2 == 0):
#            for i in range(-int(apa/2), int(apa/2)):
#                for j in range(-int(apa/2), int(apa/2)):
#                    aptr = ot.circle(grid_size, asz)
#                    aptr = np.roll(aptr, int((i+0.5)*sampling*ratio), axis=0)
#                    aptr = np.roll(aptr, int((j+0.5)*sampling*ratio), axis=1)
#                    
#                    # Check whether the aperture lies inside the pupil model
#                    if (np.mean(pupil[aptr == 1]) > 0.5):
#                        xaptr = int(i+0.5)*sampling
#                        yaptr = int(j+0.5)*sampling
#                        
#                        # Rotate pupil
#                        if (pupil_rot is not None):
#                            temp = rot.dot(np.array([xaptr, yaptr]))
#                            xaptr = temp[0]
#                            yaptr = temp[1]
#                        txtfile.write('%+.10f %+.10f 1.0\n' % (xaptr, yaptr))
#                        dummy[aptr == 1] = 1
#        else:
#            for i in range(-int(apa/2), int(apa/2)+1):
#                for j in range(-int(apa/2), int(apa/2)+1):
#                    aptr = ot.circle(grid_size, asz)
#                    aptr = np.roll(aptr, int(i*sampling*ratio), axis=0)
#                    aptr = np.roll(aptr, int(j*sampling*ratio), axis=1)
#                    
#                    # Check whether the aperture lies inside the pupil model
#                    if (np.mean(pupil[aptr == 1]) > 0.5):
#                        xaptr = int(i)*sampling
#                        yaptr = int(j)*sampling
#                        
#                        # Rotate pupil
#                        if (pupil_rot is not None):
#                            temp = rot.dot(np.array([xaptr, yaptr]))
#                            xaptr = temp[0]
#                            yaptr = temp[1]
#                        txtfile.write('%+.10f %+.10f 1.0\n' % (xaptr, yaptr))
#                        dummy[aptr == 1] = 1
#        
#        # Close the text file for the pupil model coordinates
#        txtfile.close()
        
#        # Special model with one sub-aperture per individual hexagon
#        txtfile = open(txtpath, 'w')
#        x_roll = [0, 0, 0, 0, 0, 0, -0.75*D_hex, -0.75*D_hex, -0.75*D_hex, -0.75*D_hex, -0.75*D_hex, -0.75*D_hex, 0.75*D_hex, 0.75*D_hex, 0.75*D_hex, 0.75*D_hex, 0.75*D_hex, 0.75*D_hex, -1.5*D_hex, -1.5*D_hex, -1.5*D_hex, -1.5*D_hex, -1.5*D_hex, 1.5*D_hex, 1.5*D_hex, 1.5*D_hex, 1.5*D_hex, 1.5*D_hex, -2.25*D_hex, -2.25*D_hex, -2.25*D_hex, -2.25*D_hex, 2.25*D_hex, 2.25*D_hex, 2.25*D_hex, 2.25*D_hex]
#        y_roll = [-d_hex, d_hex, -2.*d_hex, 2.*d_hex, -3.*d_hex, 3.*d_hex, -0.5*d_hex, 0.5*d_hex, -1.5*d_hex, 1.5*d_hex, -2.5*d_hex, 2.5*d_hex, -0.5*d_hex, 0.5*d_hex, -1.5*d_hex, 1.5*d_hex, -2.5*d_hex, 2.5*d_hex, 0, -d_hex, d_hex, -2.*d_hex, 2.*d_hex, 0, -d_hex, d_hex, -2.*d_hex, 2.*d_hex, -0.5*d_hex, 0.5*d_hex, -1.5*d_hex, 1.5*d_hex, -0.5*d_hex, 0.5*d_hex, -1.5*d_hex, 1.5*d_hex]
#        if (pupil_rot is not None):
#            theta = pupil_rot*np.pi/180.
#            rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
#        for i in range(len(x_roll)):
#            xaptr = x_roll[i]
#            yaptr = y_roll[i]
#            
#            # Rotate pupil
#            if (pupil_rot is not None):
#                temp = rot.dot(np.array([xaptr, yaptr]))
#                xaptr = temp[0]
#                yaptr = temp[1]
#            txtfile.write('%+.10f %+.10f 1.0\n' % (xaptr, yaptr))
#        txtfile.close()
        
#        # Special model with three sub-apertures per individual hexagon
#        txtfile = open(txtpath, 'w')
#        x_roll = [0, 0, 0, 0, 0, 0, -0.75*D_hex, -0.75*D_hex, -0.75*D_hex, -0.75*D_hex, -0.75*D_hex, -0.75*D_hex, 0.75*D_hex, 0.75*D_hex, 0.75*D_hex, 0.75*D_hex, 0.75*D_hex, 0.75*D_hex, -1.5*D_hex, -1.5*D_hex, -1.5*D_hex, -1.5*D_hex, -1.5*D_hex, 1.5*D_hex, 1.5*D_hex, 1.5*D_hex, 1.5*D_hex, 1.5*D_hex, -2.25*D_hex, -2.25*D_hex, -2.25*D_hex, -2.25*D_hex, 2.25*D_hex, 2.25*D_hex, 2.25*D_hex, 2.25*D_hex]
#        y_roll = [-d_hex, d_hex, -2.*d_hex, 2.*d_hex, -3.*d_hex, 3.*d_hex, -0.5*d_hex, 0.5*d_hex, -1.5*d_hex, 1.5*d_hex, -2.5*d_hex, 2.5*d_hex, -0.5*d_hex, 0.5*d_hex, -1.5*d_hex, 1.5*d_hex, -2.5*d_hex, 2.5*d_hex, 0, -d_hex, d_hex, -2.*d_hex, 2.*d_hex, 0, -d_hex, d_hex, -2.*d_hex, 2.*d_hex, -0.5*d_hex, 0.5*d_hex, -1.5*d_hex, 1.5*d_hex, -0.5*d_hex, 0.5*d_hex, -1.5*d_hex, 1.5*d_hex]
#        if (pupil_rot is not None):
#            theta = pupil_rot*np.pi/180.
#            rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
#        for i in range(len(x_roll)):
#            for j in range(3):
#                xx = d_hex/3.*np.sin(j/3.*2.*np.pi)
#                yy = d_hex/3.*np.cos(j/3.*2.*np.pi)
#                xaptr = x_roll[i]+xx
#                yaptr = y_roll[i]+yy
#                
#                # Rotate pupil
#                if (pupil_rot is not None):
#                    temp = rot.dot(np.array([xaptr, yaptr]))
#                    xaptr = temp[0]
#                    yaptr = temp[1]
#                txtfile.write('%+.10f %+.10f 1.0\n' % (xaptr, yaptr))
#        txtfile.close()
        
        # Special model with three sub-apertures per individual hexagon and
        # central obscuration
        txtfile = open(txtpath, 'w')
        x_roll = [0, 0, 0, 0, 0, 0, -0.75*D_hex, -0.75*D_hex, -0.75*D_hex, -0.75*D_hex, -0.75*D_hex, -0.75*D_hex, 0.75*D_hex, 0.75*D_hex, 0.75*D_hex, 0.75*D_hex, 0.75*D_hex, 0.75*D_hex, -1.5*D_hex, -1.5*D_hex, -1.5*D_hex, -1.5*D_hex, -1.5*D_hex, 1.5*D_hex, 1.5*D_hex, 1.5*D_hex, 1.5*D_hex, 1.5*D_hex, -2.25*D_hex, -2.25*D_hex, -2.25*D_hex, -2.25*D_hex, 2.25*D_hex, 2.25*D_hex, 2.25*D_hex, 2.25*D_hex]
        y_roll = [-d_hex, d_hex, -2.*d_hex, 2.*d_hex, -3.*d_hex, 3.*d_hex, -0.5*d_hex, 0.5*d_hex, -1.5*d_hex, 1.5*d_hex, -2.5*d_hex, 2.5*d_hex, -0.5*d_hex, 0.5*d_hex, -1.5*d_hex, 1.5*d_hex, -2.5*d_hex, 2.5*d_hex, 0, -d_hex, d_hex, -2.*d_hex, 2.*d_hex, 0, -d_hex, d_hex, -2.*d_hex, 2.*d_hex, -0.5*d_hex, 0.5*d_hex, -1.5*d_hex, 1.5*d_hex, -0.5*d_hex, 0.5*d_hex, -1.5*d_hex, 1.5*d_hex]
        if (pupil_rot is not None):
            theta = pupil_rot*np.pi/180.
            rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        for i in range(len(x_roll)):
            for j in range(3):
                xx = d_hex/3.*np.sin(j/3.*2.*np.pi)
                yy = d_hex/3.*np.cos(j/3.*2.*np.pi)
                xaptr = x_roll[i]+xx
                yaptr = y_roll[i]+yy
                
                # Rotate pupil
                if (pupil_rot is not None):
                    temp = rot.dot(np.array([xaptr, yaptr]))
                    xaptr = temp[0]
                    yaptr = temp[1]
                
                # Central obscuration
                if (np.sqrt(xaptr**2+yaptr**2) > D_cen/2.):
                    txtfile.write('%+.10f %+.10f 1.0\n' % (xaptr, yaptr))
        txtfile.close()
        
#        # Plot pupil model and apertures
#        plt.figure()
#        plt.imshow(pupil+dummy)
#        plt.show()
#        plt.close()
        
        # Load pupil model into XARA and save it as fits file
        KPI = xara.KPI(fname=txtpath,
                       bmax=bmax)
        KPI.package_as_fits(fname=fitspath)
        
        # Plot pupil model and apertures
        f = KPI.plot_pupil_and_uv()
        plt.show(block=False)
        plt.close()
        
        print('--------------------')
        
        # Return paths to the text file and the fits file of the pupil model
        return txtpath, fitspath
    
    def recenter(self,
                 fitsfile_in,
                 fitsfile_out,
                 wrad=None):
        """
        Function for re-centering the data using FFT. This is required for
        estimating the kernel-phase covariance based on photon noise using a
        linear basis transform.
        
        Parameters
        ----------
        fitsfile_in : Path where the fits file to be processed is located.
        fitsfile_out : Path where the kernel-phase fits file will be saved to.
        wrad : Radius of the super-Gaussian mask by which the data is windowed.
               This can be useful for minimizing edge effects when FFT is
               applied. If 'None', no windowing will be applied.
        """
        
        print('Re-centering '+fitsfile_in)
        
        # Open input fits file
        hdul = pyfits.open(self.idir+fitsfile_in)
        data = hdul[0].data.copy()
        xsz = hdul[0].header['NAXIS1'] # Image x-size (pixels)
        ysz = hdul[0].header['NAXIS2'] # Image y-size (pixels)
        
        # Compute the super-Gaussian mask
        self.sgmask = None
        if (wrad is not None):
            self.sgmask = xara.core.super_gauss(ysz, xsz, ysz/2, xsz/2, wrad)
        
        # Re-center each frame
        for i in range(data.shape[0]):
            data[i] = xara.core.recenter(data[i],
                                         mask=self.sgmask,
                                         algo='BCEN',
                                         subpix=True,
                                         between=False,
                                         verbose=True)
            if (self.sgmask is not None):
                data[i] *= self.sgmask
        
        # Save data cube
        hdul[0].data = data
        hdul.writeto(self.odir+fitsfile_out, clobber=True) # FIXME
        hdul.close()
        
        pass
    
    def extract_kerphase(self,
                         fitsfiles,
                         recenter=True,
                         wrad=None,
                         method='LDFT1'):
        """
        Function for extracting the kernel-phase. This function calls recenter
        and safes the centered frames to a copy of the original fits files.
        
        Parameters
        ----------
        fitsfiles : Paths where the fits files to be processed are located.
        recenter : Re-center the frames in complex visibility space when
                   extracing the kernel-phase.
        wrad : Radius of the super-Gaussian mask by which the data is windowed.
               This can be useful for minimizing edge effects when FFT is
               applied. If 'None', no windowing will be applied.
        method : Method for computing the Fourier transform of the data.
                 Possible options are 'LDFT1', 'LDFT2', 'FFT'.
        """
        
        print('--> Extracting kernel-phase from '+str(len(fitsfiles))+' fits file(s)')
        
        # Generate names for the kernel-phase files
        self.fitsfiles = fitsfiles
        self.kpfiles = []
        self.wrad = wrad
        for i in range(len(self.fitsfiles)):
            ww = self.fitsfiles[i].find('_lucky') # If the fits file comes from
                                                  # the PyConica pipeline, cut
                                                  # off the end before adding
                                                  # kernel.fits
            if (ww == -1):
                temp = self.fitsfiles[i][:-5]+'_kernel.fits'
            else:
                temp = self.fitsfiles[i][:ww]+'_kernel.fits'
            self.kpfiles += [temp]
            
            # Re-center each frame and save it into the kernel-phase file for
            # estimating the kernel-phase covariance based on photon noise
            self.recenter(self.fitsfiles[i],
                          self.kpfiles[i],
                          wrad=self.wrad)
            
            # Plot Fourier plane phase and spatial frequencies of the pupil
            # model
            hdul = pyfits.open(self.odir+self.kpfiles[i], memmap=False)
            temp = np.median(hdul[0].data, axis=0)
            temp = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(temp)))
            xsz = hdul[0].header['NAXIS1'] # Image x-size (pixels)
            ysz = hdul[0].header['NAXIS2'] # Image y-size (pixels)
            if (self.tel == 'VLT'):
                self.cwave = hdul[0].header['CWAVE'] # Central filter
                                                     # wavelength (meters)
                self.pscale = hdul[0].header['PSCALE'] # Pixel scale
                                                       # (milli-arcseconds)
            elif (self.tel == 'Keck'):
                self.cwave = hdul[0].header['CENWAVE']*1E-6 # Central filter
                                                            # wavelength
                                                            # (meters)
                self.pscale = 10.0 # Pixel scale (milli-arcseconds)
            else:
                raise UserWarning(str(self.tel)+' is not a known telescope')
            hdul.close()
            m2pix = xara.core.mas2rad(self.pscale)*xsz/self.cwave # Fourier scaling
            u = self.KPI.UVC[:, 0]*m2pix+xsz/2.
            v = self.KPI.UVC[:, 1]*m2pix+ysz/2.
            plt.figure()
            plt.imshow(np.angle(temp))
            plt.scatter(u, v)
            plt.show()
            plt.close()
            
            # Extract kernel-phase from each frame from the non-centered fits
            # file in order to avoid errors from the FFT
            self.KPO.extract_KPD(self.idir+self.fitsfiles[i],
                                 recenter=recenter,
                                 wrad=self.wrad,
                                 method=method)
            
#            # Extract Fourier plane phase and kernel-phase for determining the
#            # image quality
#            imgs = pyfits.getdata(self.odir+self.kpfiles[i])
#            phase = []
#            for j in range(imgs.shape[0]):
#                phase += [np.angle(self.KPO.extract_cvis_from_img(imgs[j], m2pix, method=method))]
#            phase = np.array(phase)
#            kernel_phase = self.KPO.KPDT[i]
        
        print('')
        
        print('--------------------')
        
        pass
    
    def compute_covariance(self):
        """
        Function for computing the kernel-phase covariance based on photon
        noise using Monte Carlo simulations as well as a basis transform.
        """
        
#        print('--> Computing covariance matrices for '+str(len(self.kpfiles))+' fits file(s); MC')
#        
#        # Firstly, estimate the kernel-phase covariance based on MC simulations
#        # Note: In principle, this requires at least as many frames as there
#        # are kernel-phases
#        nc = len(self.fitsfiles) # Number of data cubes
#        niter = int(10.*self.KPI.KPM.shape[0]) # Simulate 10 times as many frames as
#                                               # there are kernel-phases in order to
#                                               # obtain a good estimate of the
#                                               # covariance matrix
#        self.kpcov_MC = []
#        for i in range(nc):
#            
#            # Open input fits file
#            hdul = pyfits.open(self.idir+self.fitsfiles[i])
#            data = hdul[0].data.copy()
#            nf = hdul[0].header['NAXIS3'] # Number of frames
#            gain = hdul[0].header['GAIN'] # Detector gain (e-/ADU)
#            if (self.tel == 'VLT'):
#                bgs = hdul['TEL'].data['BACKS'] # Backgrounds from PyConica
#                                                # pipeline
#                self.cwave = hdul[0].header['CWAVE'] # Central filter
#                                                     # wavelength (meters)
#                self.pscale = hdul[0].header['PSCALE'] # Pixel scale
#                                                       # (milli-arcseconds)
#            elif (self.tel == 'Keck'):
#                bgs = hdul[1].data['BACKGROUND']
#                self.cwave = hdul[0].header['CENWAVE']*1E-6 # Central filter
#                                                            # wavelength
#                                                            # (meters)
#                self.pscale = 10.0 # Pixel scale (milli-arcseconds)
#            else:
#                raise UserWarning(str(self.tel)+' is not a known telescope')
#            hdul.close()
#            
#            # Estimate the covariance matrix based on MC simulations
#            kpcov = []
#            for j in range(nf):
#                sys.stdout.write('\rFrame %.0f of %.0f, data cube %.0f of %.0f' % (j+1, nf, i+1, nc))
#                sys.stdout.flush()
#                
#                # Add noise to each frame based on Poisson statistics, then
#                # extract the kernel-phase and compute its covariance
#                frame = data[j]*gain
#                if (self.sgmask is not None):
#                    varframe = (frame+bgs[j]*self.sgmask*gain)*self.sgmask
#                else:
#                    varframe = (frame+bgs[j]*gain)
#                stdframe = np.sqrt(varframe)
#                kps = []
#                for k in range(niter):
#                    temp = frame+np.random.normal(loc=0.,
#                                                  scale=stdframe,
#                                                  size=stdframe.shape)
#                    temp = self.KPO.extract_KPD_single_frame(temp,
#                                                             self.pscale,
#                                                             self.cwave,
#                                                             recenter=True,
#                                                             wrad=self.wrad,
#                                                             method='LDFT1')
#                    kps += [temp]
#                kps = np.array(kps)
#                kpcov += [np.cov(kps.T)]
#            self.kpcov_MC += [np.array(kpcov)]
#        print('')
        
        print('--> Computing covariance matrices for '+str(len(self.kpfiles))+' fits file(s); BT')
        
        # Secondly, estimate the kernel-phase covariance based on photon noise
        self.kpcov_ph = []
        nc = len(self.fitsfiles) # Number of data cubes
        for i in range(nc):
            
            # Open input fits file
            hdul = pyfits.open(self.odir+self.kpfiles[i], memmap=False)
            data = hdul[0].data.copy()
            nf = hdul[0].header['NAXIS3'] # Number of frames
            gain = hdul[0].header['GAIN'] # Detector gain (e-/ADU)
            if (self.tel == 'VLT'):
                bgs = hdul['TEL'].data['BACKS'] # Backgrounds from PyConica
                                                # pipeline
            elif (self.tel == 'Keck'):
                bgs = hdul[1].data['BACKGROUND']
            else:
                raise UserWarning(str(self.tel)+' is not a known telescope')
            hdul.close()
            
            # Estimate the kernel-phase covariance based on photon noise for
            # each frame
            kpcov = []
            for j in range(nf):
                sys.stdout.write('\rFrame %.0f of %.0f, data cube %.0f of %.0f' % (j+1, nf, i+1, nc))
                sys.stdout.flush()
                
                # Compute the basis transform B which maps the photon count to
                # its kernel-phase
                # Note: B = K.Im(F)/|F.frame|
                frame = data[j]*gain # Convert ADUs to photo-electrons
                B = self.KPI.KPM.dot(np.divide(self.KPO.FF.imag.T, np.abs(self.KPO.FF.dot(frame.flatten()))).T)
                
                # Compute the photon count variance (assuming Poisson
                # statistics)
                # Note: Account for the super-Gaussian mask if it was applied
                # during the re-centering
                if (self.sgmask is not None):
                    varframe = (frame+bgs[j]*self.sgmask*gain)*self.sgmask
                else:
                    varframe = (frame+bgs[j]*gain)
                
                # Apply the basis transform B
                kpcov += [np.multiply(B, varframe.flatten()).dot(B.T)]
            self.kpcov_ph += [np.array(kpcov)]
        print('')
        
        print('--------------------')
        
        pass
    
    def save_as_fits(self):
        """
        Function for saving data cubes as kernel-phase fits files.
        """
        
        # Save each data cube as a kernel-phase fits file
        nc = len(self.fitsfiles) # Number of data cubes
        for i in range(nc):
            
            # Open data cube
            hdul = pyfits.open(self.odir+self.kpfiles[i], memmap=False)
            
            # Plot Fourier plane phase, covariance matrix and kernel-phase
            f, axarr = plt.subplots(2, 2)
            temp = np.median(hdul[0].data, axis=0)
            temp = np.angle(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(temp))))
            p00 = axarr[0, 0].imshow(temp, vmin=-np.pi, vmax=np.pi)
            plt.colorbar(p00, ax=axarr[0, 0])
            p01 = axarr[0, 1].imshow(np.mean(self.kpcov_ph[i], axis=0))
            plt.colorbar(p01, ax=axarr[0, 1])
            temp = []
            xsz = int(hdul[0].header['NAXIS1']) # Image x-size (pixels)
            nf = hdul[0].header['NAXIS3'] # Number of frames
            if (self.tel == 'VLT'):
                cwave = hdul[0].header['CWAVE'] # Central filter wavelength
                                                # (meters)
                pscale = hdul[0].header['PSCALE'] # Pixel scale
                                                  # (milli-arcseconds)
            elif (self.tel == 'Keck'):
                cwave = hdul[0].header['CENWAVE']*1E-6 # Central filter
                                                       # wavelength (meters)
                pscale = 10.0 # Pixel scale (milli-arcseconds)
            else:
                raise UserWarning(str(self.tel)+' is not a known telescope')
            for j in range(nf):
                temp += [np.angle(self.KPO.extract_cvis_from_img(hdul[0].data[j],
                                                                 m2pix=xara.core.mas2rad(pscale)*xsz/cwave,
                                                                 method='LDFT1'))]
            temp = np.array(temp)
            axarr[1, 0].plot(np.mean(temp, axis=0), label='mean')
            axarr[1, 0].plot(np.min(temp, axis=0), label='min')
            axarr[1, 0].plot(np.max(temp, axis=0), label='max')
            axarr[1, 0].set_ylim([-np.pi, np.pi])
            axarr[1, 0].grid()
#            axarr[1, 0].legend()
            temp = np.sqrt(np.diag(np.mean(self.kpcov_ph[i], axis=0)))
            axarr[1, 1].errorbar(np.arange(self.KPO.KPDT[i].shape[1]), np.mean(self.KPO.KPDT[i], axis=0), yerr=temp, label='mean')
            axarr[1, 1].grid()
#            axarr[1, 1].legend()
            plt.savefig(self.odir+self.kpfiles[i][:-5]+'.pdf')
            plt.close()
            
            # Prepare pupil plane columns
            xy1 = pyfits.Column(name='XXC', format='D', array=self.KPI.VAC[:, 0])
            xy2 = pyfits.Column(name='YYC', format='D', array=self.KPI.VAC[:, 1])
            trm = pyfits.Column(name='TRM', format='D', array=self.KPI.TRM)
            
            # Prepare Fourier plane columns
            uv1 = pyfits.Column(name='UUC', format='D', array=self.KPI.UVC[:, 0])
            uv2 = pyfits.Column(name='VVC', format='D', array=self.KPI.UVC[:, 1])
            red = pyfits.Column(name='RED', format='I', array=self.KPI.RED)
            
            # Prepare baseline mapping matrix
            BLM = np.diag(self.KPI.RED).dot(self.KPI.TFM)
            
            # Primary HDU
            pri_hdu = pyfits.PrimaryHDU(hdul[0].data.copy())
            pri_hdu.header = hdul[0].header.copy()
            pri_hdu.header['CWAVE'] = self.cwave
            pri_hdu.header['PSCALE'] = self.pscale
            
            # Secondary HDU
            if (self.tel == 'VLT'):
                sec_hdu = pyfits.ImageHDU(hdul[1].data.copy())
                sec_hdu.header = hdul[1].header.copy()
            
            # Tertiary HDU
            ter_hdu = pyfits.ImageHDU(hdul[2].data.copy())
            ter_hdu.header = hdul[2].header.copy()
            ter_hdu.header['EXTNAME'] = 'BP-MAP'
            
            # Telemetry HDU
            if (self.tel == 'VLT'):
                tb1_hdu = pyfits.BinTableHDU(hdul['TEL'].data.copy())
                tb1_hdu.header = hdul['TEL'].header.copy()
            elif (self.tel == 'Keck'):
                tb1_hdu = pyfits.BinTableHDU(hdul[1].data.copy())
                tb1_hdu.header = hdul[1].header.copy()
                tb1_hdu.header['EXTNAME'] = 'TEL'
            
            # Close data cube
            hdul.close()
            
            # Pupil plane HDU
            tb2_hdu = pyfits.BinTableHDU.from_columns([xy1, xy2, trm])
            tb2_hdu.header['EXTNAME'] = 'APERTURE'
            tb2_hdu.header['TTYPE1'] = ('XXC', 'Virtual aperture x-coord (meters)')
            tb2_hdu.header['TTYPE2'] = ('YYC', 'Virtual aperture y-coord (meters)')
            tb2_hdu.header['TTYPE3'] = ('TRM', 'Virtual aperture transmission (0 < t <= 1)')
            
            # Fourier plane HDU
            tb3_hdu = pyfits.BinTableHDU.from_columns([uv1, uv2, red])
            tb3_hdu.header['EXTNAME'] = 'UV-PLANE'
            tb3_hdu.header['TTYPE1'] = ('UUC', 'Baseline u coordinate (meters)')
            tb3_hdu.header['TTYPE2'] = ('VVC', 'Baseline v coordinate (meters)')
            tb3_hdu.header['TTYPE3'] = ('RED', 'Baseline redundancy (int)')
            
            # Kernel-phase relations HDU
            kpm_hdu = pyfits.ImageHDU(self.KPI.KPM)
            kpm_hdu.header['EXTNAME'] = 'KER-MAT'
            kpm_hdu.header.add_comment('Kernel-phase matrix')
            
            # Baseline mapping matrix HDU
            blm_hdu = pyfits.ImageHDU(BLM)
            blm_hdu.header['EXTNAME'] = 'BLM-MAT'
            blm_hdu.header.add_comment('Baseline mapping matrix')
            
            # Kernel-phase HDU
            kpd_hdu = pyfits.ImageHDU(self.KPO.KPDT[i])
            kpd_hdu.header['EXTNAME'] = 'KP-DATA'
            kpd_hdu.header.add_comment('Kernel-phase')
            
            # Kernel-phase covariance HDU
            kpc_ph_hdu = pyfits.ImageHDU(self.kpcov_ph[i])
            kpc_ph_hdu.header['EXTNAME'] = 'KP-SIGM'
            kpc_ph_hdu.header.add_comment('Covariance of kernel-phase (using a basis transform)')
#            kpc_MC_hdu = pyfits.ImageHDU(self.kpcov_MC[i])
#            kpc_MC_hdu.header['EXTNAME'] = 'KP-SIGM MC'
#            kpc_MC_hdu.header.add_comment('Covariance of kernel-phase (using Monte-Carlo simulations)')
            
            # Save data cube
            if (self.tel == 'VLT'):
#                outfile = pyfits.HDUList([pri_hdu, sec_hdu, ter_hdu, tb1_hdu, tb2_hdu, tb3_hdu, kpm_hdu, blm_hdu, kpd_hdu, kpc_ph_hdu, kpc_MC_hdu])
                outfile = pyfits.HDUList([pri_hdu, sec_hdu, ter_hdu, tb1_hdu, tb2_hdu, tb3_hdu, kpm_hdu, blm_hdu, kpd_hdu, kpc_ph_hdu])
            elif (self.tel == 'Keck'):
#                outfile = pyfits.HDUList([pri_hdu, ter_hdu, tb1_hdu, tb2_hdu, tb3_hdu, kpm_hdu, blm_hdu, kpd_hdu, kpc_ph_hdu, kpc_MC_hdu])
                outfile = pyfits.HDUList([pri_hdu, ter_hdu, tb1_hdu, tb2_hdu, tb3_hdu, kpm_hdu, blm_hdu, kpd_hdu, kpc_ph_hdu])
            outfile.writeto(self.odir+self.kpfiles[i], clobber=True, output_verify='fix') # FIXME
            outfile.close()
        
        pass
