# This Maximum-Entropy method, like any MEM method, has 2 regularizers:
# one for total flux and one for entropy. In an ideal situation, total 
# flux is fixed from the data (not here as we are in the high-contrast
# limit with an unresolved point source), and the MEM regularizer can
# be set to give a reduced chi-squared = 1. Often in a MEM technique, 
# the total flux is constrained using a prior. This is the approach
# that we will take here.

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import astropy.io.fits as pyfits
import os
from scipy.optimize import minimize
import pdb
plt.ion()

# Requires XARA (https://github.com/fmartinache/xara) and opticstools
# (https://github.com/mikeireland/opticstools.git)
import sys
#sys.path.append('F:\\Python\\Development\\NIRC2\\xara')
sys.path.append('/home/kjens/Python/Development/NIRC2/xara')
#sys.path.append('F:\\Python\\Packages\\opticstools\\opticstools\\opticstools')
sys.path.append('/home/kjens/Python/Packages/opticstools/opticstools/opticstools')
import xara
import opticstools as ot

import kl_calibration
import pybinary

#Input a fits file, containing an linear image-plane (i.e. high-contrast)
#representation of a set of kernel phases, plus data and errors in 
#the fits extension
def data_in(filename):
    hdulist = pyfits.open(filename)
    pxscale = hdulist[0].header['PXSCALE']
    Amat = hdulist[0].data # JK: One matrix to obtain kernel-phases from image
    dimx = Amat.shape[2]
    dimy = Amat.shape[1]
    Amat = Amat.reshape([Amat.shape[0],dimx*dimy])
    tbdata = hdulist[1].data
    theta = tbdata[0]
    theta_sig = tbdata[1]
    return Amat,theta,theta_sig,dimx,dimy,pxscale

def data_in_jens(pbfile):
    """
    TODO
    """
    
    hdul = pyfits.open(pbfile)
    UVC = hdul['UV-PLANE'].data
    coords = np.vstack([UVC['UUC'], UVC['VVC']]).T
    pscale = hdul[0].header['PSCALE']
    isz = hdul[0].header['NAXIS1']
    cwave = hdul[0].header['CWAVE']
    m2pix = xara.core.mas2rad(pscale)*isz/cwave
    pas = hdul['TEL'].data['pa']
    KPM = hdul['KER-MAT'].data # Fourier plane phase --> kernel-phase
    P = hdul['KL-PROJ'].data # Kernel-phase --> calibrated kernel-phase
    
    Amats = []
    thetas = []
    theta_sigs = []
    for i in range(hdul['KP-DATA'].data.shape[0]):
        
        FF = xara.core.compute_DFTM1(coords, m2pix, isz, pa=None) # Image plane --> Fourier plane
        
        Amats += [P.dot(KPM).dot(np.imag(FF))]
        thetas += [hdul['KP-DATA'].data[i]]
        theta_sigs += [np.sqrt(np.diagonal(hdul['KP-SIGM'].data[i]))]
    
    Amat = np.concatenate(Amats)
    theta = np.concatenate(thetas)
    theta_sig = np.concatenate(theta_sigs)
    
    print('--> K_klip = %.0f' % hdul[0].header['KKLIP'])
    plt.figure()
    plt.plot(np.array(thetas).T)
    plt.show(block=True)
    print ('The kernel-phases look fine')
    plt.figure()
    plt.plot(np.array(theta_sigs).T)
    plt.show(block=True)
    print('The kernel-phase uncertainties vary a lot')
    hdul.close()
    
    return Amat, theta, theta_sig, isz, isz, pscale

def data_in_multi(filenames):
    Amats = []
    thetas = []
    theta_sigs = []
    
    for filename in filenames:
        hdulist = pyfits.open(filename)
        pxscale = hdulist[0].header['PXSCALE']
        Amat = hdulist[0].data
        dimx = Amat.shape[2]
        dimy = Amat.shape[1]
        Amat = Amat.reshape([Amat.shape[0],dimx*dimy])
        tbdata = hdulist[1].data
        theta = tbdata[0]
        theta_sig = tbdata[1]
        Amats.append(Amat)
        thetas.append(theta)
        theta_sigs.append(theta_sig)
    Amat = np.concatenate(Amats)
    theta = np.concatenate(thetas)
    theta_sig = np.concatenate(theta_sigs)
    return Amat,theta,theta_sig,dimx,dimy,pxscale


#Begin with a Uniform prior, based on the size of the input data
def uniform_prior(pbfile):
    hdulist = pyfits.open(pbfile)
    sz = int(hdulist[0].header['NAXIS1'])
    if (sz != hdulist[0].header['NAXIS2']):
        print("Major Error: So far this needs a square array!")
        raise UserWarning
    return (np.ones(int(sz**2.0))/sz**2.0,sz) # JK: Prior is vector of length sz^2 with values 1/sz^2

#Return the entropy functional, e.g. Narayan and Nituananda 1986
def ent(vector,prior):
    return np.sum(vector*np.log(np.maximum(vector,1e-20)/prior)) # JK: Entropy is value*log(value/prior) = value*log(value*sz^2)

#Return the gradient of the entropy functional.
def grad_ent(vector,prior):
    return (np.log(np.maximum(vector,1e-20)/prior) + 1.0) # JK: (x*log(x/c))' = log(x/c)+c, sum(prior) = 1

#Return the function we are trying to minimise.
def func(z,Amat,theta,theta_sig,alpha,prior):
    theta_mod = np.dot(Amat,z) # JK: Modified kernel-phases
    chi2 = np.sum((theta_mod - theta)**2/theta_sig**2) # JK: Chi^2 between original and modified kernel-phases
    stat = chi2 
    if (alpha != 0):
        stat = stat + alpha*ent(z,prior)  # JK: Chi^2+alpha*entropy, minimize chi^2 & entropy
    return stat,theta_mod,chi2

def f_minimize(z,Amat,theta,theta_sig,theta_mod,alpha,prior):
    stat,theta_mod,chi2 = func(z,Amat,theta,theta_sig,alpha,prior)
    return stat
    
#Return the gradient of the function we are trying to minimise.
def grad_f(z,Amat,theta,theta_sig,theta_mod,alpha,prior):
    if len(theta_mod)==0:
        theta_mod = np.dot(Amat,z)
    the_vect = (theta_mod - theta)/theta_sig**2 # Chi^2 vector
    retvect = 2*np.dot(the_vect.reshape(1,theta.size),Amat) # JK: ???
    if (alpha != 0):
        retvect = retvect + alpha*grad_ent(z,prior)
    return retvect.flatten()

#Return the function we are trying to minimise along a line 
#defined by z = s + p*t.
#Inputs include the matrix products A * s and A * p.
def fline(t,s,p,As,Ap,theta,theta_sig,alpha,prior):
    z = s + p*t # JK: z is modified image
    chi2 = np.sum((As + Ap*t - theta)**2/theta_sig**2) # JK: As+Ap*t are modified kernel-phases
    if (alpha != 0):
        chi2 = chi2 + alpha*ent(z,prior)
    return chi2

#Perform a line search from a starting image "s" in a direction "p"
def line_search(s,p,As,Ap,theta,theta_sig,alpha,prior):
    #This is a golden section search with a fixed 20 iterations.
    #Returns the t value of the minimum.
    niter=20
    if (min(p) > 0):
        #This shouldn't really happen
        tmax = 2*np.max(s)/np.max(p)
    else:
        ts = -s/p.clip(-1e12,-1e-12)
        #A somewhat arbitrary limit below to prevent log(0)
        if (alpha != 0):
            tmax = np.min(ts) * (1-1e-2)
        #If we're not doing MEM, then let up to 10% of the pixels go below zero.
        else:
            tmax = np.percentile(ts,33)
    phi = (1 + np.sqrt(5))/2.0
    resphi = 2 - phi
    t1 = 0
    t3 = tmax
    t2 = t3/phi
    f1 = fline(t1,s,p,As,Ap,theta,theta_sig,alpha,prior)
    f2 = fline(t2,s,p,As,Ap,theta,theta_sig,alpha,prior)
    f3 = fline(t3,s,p,As,Ap,theta,theta_sig,alpha,prior)
    for i in range(niter):
        if (t3 - t2 > t2 - t1):
          t4 = t2 + resphi * (t3 - t2)
          f4 = fline(t4,s,p,As,Ap,theta,theta_sig,alpha,prior)
          if (f4 < f2):
              f1 = f2
              f2 = f4
              t1 = t2
              t2 = t4
          else:
              f3 = f4
              t3 = t4
        else:
          t4 = t2 - resphi * (t2 - t1)
          f4 = fline(t4,s,p,As,Ap,theta,theta_sig,alpha,prior)            
          if (f2 < f4):
              f1 = f4
              t1 = t4
          else:
              f3 = f2
              f2 = f4
              t3 = t2
              t2 = t4
    return [t1,t2,t3][np.argmin([f1,f2,f3])]

def unreg_image(pbfile, niter=20, guess=[],flux=0.1, median_cutoff=False, 
        no_sigma=False, full_output=False):
    """Create a simple unregularised image
    """
    if True:
#        [Amat,theta,theta_sig,dimx,dimy,pxscale] = data_in(infile)
        [Amat,theta,theta_sig,dimx,dimy,pxscale] = data_in_jens(pbfile)
        [dummy,sz]=uniform_prior(pbfile)
    else:
        [Amat,theta,theta_sig,dimx,dimy,pxscale] = data_in_multi(infile)
        [dummy,sz]=uniform_prior(infile[0])
    if median_cutoff:
        theta_sig = np.maximum(theta_sig,np.median(theta_sig)) # JK: Ignore everything below median
    if no_sigma:
        theta_sig[:] = np.median(theta_sig) # JK: Set everything to median
    
    if len(guess)==0:
        z = dummy.copy()*flux
    #plt.clf()
    for i in range(niter):
        stat,theta_mod,chi2 = func(z,Amat,theta,theta_sig,0.0,dummy)
        grad = -grad_f(z,Amat,theta,theta_sig,theta_mod,0.0,dummy).flatten()
        s = z.copy()
        As = theta_mod 
        Ap = np.dot(Amat,grad)
        tmin = line_search(s,grad,As,Ap,theta,theta_sig,0.0,dummy) # JK: Modify z in the direction of its steepest gradient
        z = s + tmin*grad # JK: New image is old image-const*gradient(old image)
#        plt.imshow(z.reshape(sz,sz), interpolation='nearest')
#        plt.draw()
#        pdb.set_trace()
    [stat,theta_mod,chi2] = func(z,Amat,theta,theta_sig,0,prior)
    print("Chi2 (unreg): {0:6.2f}".format(chi2/len(theta)))
    extent = [dimx/2.0*pxscale,-dimx/2.0*pxscale,-dimy/2.0*pxscale,dimy/2.0*pxscale]
    if full_output:
        return z, extent,theta_mod,theta,theta_sig
    else:
        return z,extent

#Create a Maximum-Entropy image based on an input fits
#file and a prior. Optional inputs are:
# alpha: A starting value for the MEM functional multiplier (default=1.0)
# gain: The servo gain for adjusting alpha to achieve chi^2=1
# niter: The number of iterations. 
def mem_image(infile,prior,alpha=1.0, gain=0.1, niter=150, target_chi2=1.0, guess=[], \
        adjust_alpha=False, start_i=9):
    if len(guess)==0:
      z = prior.copy()
    else:
      z = guess.copy()
    if type(infile)==str:
        [Amat,theta,theta_sig,dimx,dimy,pxscale] = data_in(infile)
        [dummy,sz]=uniform_prior(infile)
    else:
        [Amat,theta,theta_sig,dimx,dimy,pxscale] = data_in_multi(infile)
        [dummy,sz]=uniform_prior(infile[0])
    #start_i=19 #Should this be an option?
    if start_i < 0:
      alpha_use = alpha
    else:
      alpha_use=0
    #plt.clf()
    for i in range(niter):
        if ( (i+1) % 10 == 0):
            print('Done: {0:04d} of {1:04d} iterations. Chi2: {2:6.3f} Alpha: {3:6.3f} Stat: {4:6.3f} nzero: {5:d}'.format(i+1,
                niter,chi2/len(theta),alpha_use,stat, np.sum(z==0)))
            plt.imshow(z.reshape(sz,sz), interpolation='nearest')
            plt.draw()
        #While this loop runs, we can adjust alpha so that chi^2=1.
        #Lower values of alpha give lower chi^2. Not knowing how to do this,
        #lets use a servo-loop approach.
        if (i == start_i): # JK: Determine initial value for alpha
            alpha_use = alpha
            sumz = np.sum(z)
            z *= prior
            z /= np.sum(z)
            z *= sumz
            z = np.maximum(z,1e-2*np.max(z))
#            pdb.set_trace()
        elif (i > start_i) and adjust_alpha: # JK: Adjust alpha based on targeted chi^2
           err = target_chi2 - chi2/len(theta) #The difference in chi-squared
           alpha_use = alpha_use*np.exp( gain*err )
        [stat,theta_mod,chi2] = func(z,Amat,theta,theta_sig,alpha_use,prior)
        #print chi2/len(theta), alpha
        [grad] = grad_f(z,Amat,theta,theta_sig,theta_mod,alpha_use,prior)
#        plt.figure(1)
#        plt.imshow(grad.reshape(101,101))
#        plt.figure(2)
#        plt.ginput(1)
        #The following is the Polak-Ribiere Nonlinear conjugant gradient method. 
        #(http://en.wikipedia.org/wiki/Nonlinear_conjugate_gradient_method)
        #It seems to be only about a factor of 2 faster than steepest descent. 
        if (i==0):
            print("Initial Chi^2: " + str(chi2/len(theta)))
        if (i < np.maximum(1,2*start_i)):
            dir = -grad.copy() #The direction of the minimum
            p = dir.copy()
        else: # JK: This together with adapting alpha is the only change wrt unregularised image
            lastdir = dir.copy()
            dir = -grad.copy() #The direction of the minimum  
            beta = np.sum(dir*(dir - lastdir))/np.sum(dir*lastdir)
            beta = np.max(beta,0)
            p = dir + beta*p # JK: The special direction determined by the Polak-Ribiere method
        s = z.copy()
        As = theta_mod 
        Ap = np.dot(Amat,p)
        tmin = line_search(s,p,As,Ap,theta,theta_sig,alpha_use,prior)
        #if (i > start_i):
        #    pdb.set_trace()
        z = s + tmin*p
        if (alpha_use == 0):
            z = np.maximum(z,0)
    [stat,theta_mod,chi2] = func(z,Amat,theta,theta_sig,0,prior)
    print("Chi2: " + str(chi2/len(theta)))
    extent = [dimx/2*pxscale,-dimx/2*pxscale,-dimy/2*pxscale,dimy/2*pxscale]
    return z,extent

if True:
#    fn = 'LkCa15_implane.fits'
#    pngfn = 'kerphase_mem.png'
#
#    flux = 0.02
#
#    dir = '/Users/mireland/tel/nirc2/redux/141130/Ms/'
#    title = "LkCa 15, 2014 Ms"
#
#    flux = 0.01
#
#    dir = '/Users/mireland/tel/nirc2/redux/141130/Kp_9h/'
#    title = "LkCa 15, 2014 Kp 9h"
#
#    fn = 'TW Hya_implane.fits'
#    dir = '/Users/mireland/tel/nirc2/redux/141209/TW_Hya_Ms/'
#    title = "TW Hya, 2014 Ms"
    
    idir = '161107_pbfiles_sub3/'
    pbfile = idir+'MWC 480.fits'
    
    pngfn = 'mem.png'
    flux = 0.01
    dir = ''
    title = 'MWC 480'
    pngfile = dir + pngfn
    
    pngunregfile = dir + 'unreg' + pngfn
    pngnosigfile = dir + 'cutoff' + pngfn
    
    if True:
        (prior,sz) = uniform_prior(pbfile)
    else:
        (prior,sz) = uniform_prior(pbfile)
    xy = np.meshgrid(np.arange(sz) - (sz/2),np.arange(sz) - (sz/2))
    rr = np.sqrt(xy[0]**2 + xy[1]**2) # JK: Distance in Fourier plane
    g4 = np.exp(-(rr/(sz/2))**4).flatten()
    g6 = np.exp(-(rr/(sz/2))**6).flatten()
    g8 = np.exp(-(rr/(sz/2))**8).flatten()
    #[z,extent] = mem_image(file,0.1*prior*g4, alpha=1e4, niter=2000)
    
    #First, an unregularised image without taking into account sigma
    z,extent = unreg_image(pbfile, flux=flux, no_sigma=True)
    im_unreg = z.copy().reshape(sz,sz)[::-1,:]
    plt.figure(3)
    plt.clf()
    plt.imshow(np.maximum(im_unreg,0),extent=extent)
    plt.plot(0,0,'w*',ms=15)
    plt.axis(extent)
    plt.xlabel('$\Delta$ RA [mas]', fontsize='large')
    plt.ylabel('$\Delta$ DEC [mas]', fontsize='large')
    plt.title(title + ' [nosig]')
#    plt.savefig(pngnosigfile)
    plt.show(block=True)
    
    #Now, a better unregularised image.
    z, extent,theta_mod,theta,theta_sig = unreg_image(pbfile, flux=flux, full_output=True)
    im_unreg = z.copy().reshape(sz,sz)[::-1,:]
    plt.figure(1)
    plt.clf()
    plt.imshow(np.maximum(im_unreg,0),extent=extent)
    plt.plot(0,0,'w*',ms=15)
    plt.axis(extent)
    plt.xlabel('$\Delta$ RA [mas]', fontsize='large')
    plt.ylabel('$\Delta$ DEC [mas]', fontsize='large')
    plt.title(title + ' [unreg]')
#    plt.savefig(pngunregfile)
    plt.draw()
    plt.figure(4)
    plt.clf()
    plt.plot( (theta_mod - theta)/theta_sig, 'o')
    plt.show(block=True)
    
    bounds = [(0,1) for i in range(len(z))]
    x0 = z.copy() 
    x0 = g4*np.maximum(x0,np.max(x0)/100)
    if True:
#        [Amat,theta,theta_sig,dimx,dimy,pxscale] = data_in(file)
        [Amat,theta,theta_sig,dimx,dimy,pxscale] = data_in_jens(pbfile)
    else:
        [Amat,theta,theta_sig,dimx,dimy,pxscale] = data_in_multi(file)
    print("Minimising...")
    
    alpha = 1.0
    z_mem = minimize(f_minimize, x0, args=(Amat,theta,theta_sig,[],alpha,g4*prior*flux), method='L-BFGS-B', jac=grad_f, bounds=bounds)
    [stat,theta_mod,chi2] = func(z_mem.x,Amat,theta,theta_sig,alpha,prior)
    print("Chi2: {0:6.2f}".format(chi2/len(theta)))
    
    im = z_mem.x.reshape(sz,sz)
    #super-dodgy... lets chop of the edge 2 pixels because it tends to have edge effects.
    im = im[2:-2,2:-2]
    extent[0] -= 2*pxscale
    extent[3] -= 2*pxscale
    extent[1] += 2*pxscale
    extent[2] += 2*pxscale
    
    #plt.imshow(im[::-1,:], interpolation='nearest',cmap=plt.get_cmap('gist_heat'),extent=extent)
    trunc_im = np.maximum(np.minimum(im[::-1,:],np.max(im)),0)
    plt.figure(2)
    plt.imshow(trunc_im, interpolation='nearest',cmap=cm.gist_heat,extent=extent)
    plt.plot(0,0,'w*', ms=15)
    plt.axis(extent)
    plt.xlabel('$\Delta$ RA [mas]', fontsize='large')
    plt.ylabel('$\Delta$ DEC [mas]', fontsize='large')
    plt.title(title)
    print("Total contrast (mags): " + str(-2.5*np.log10(np.sum(im))))
#    plt.savefig(pngfile)
    plt.show(block=True)
