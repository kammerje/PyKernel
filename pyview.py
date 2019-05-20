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
import os
from scipy.interpolate import interp1d


# MAIN
#==============================================================================
def azavg(img):
    """
    TODO
    """
    
    # Compute azimuthal average using opticstools
    radii, azavg = ot.azimuthalAverage(img, returnradii=True, binsize=1.)
    
    # Return azimuthal average, the first value is always inf (this is
    # because the grid search finds an infinite contrast for the central
    # pixel where the binary model is not defined)
    return radii[1:], azavg[1:]

def plot_gridsearch(pbfile):
    """
    TODO
    """
    
    hdul = pyfits.open(pbfile, memmap=False)
    kps = hdul['KP-DATA'].data
    icvs = hdul['KP-SIGM INV'].data
    seps = hdul['GRIDSEARCH'].data[4]
    pas = hdul['GRIDSEARCH'].data[5]
    K_klip = hdul['GRIDSEARCH'].header['KKLIP']
    ramp = seps[0]*np.cos(np.radians(pas[0]))
    step = (ramp[-1]-ramp[0])/(ramp.shape[0]-1.)
    ticks = np.arange(ramp.shape[0])[::20]
    ticklabels = ['%.0f' % f for f in ramp[::20]]
    
    f, axarr = plt.subplots(1, 2, figsize=(4*2, 3*1))
    
    cs_mean = hdul['GRIDSEARCH'].data[0]
    cs_sdev = hdul['GRIDSEARCH'].data[1]
    chi2s = hdul['GRIDSEARCH'].data[2]
    chi2s_c = hdul['GRIDSEARCH'].data[3]
    p0 = hdul['P0'].data
    s0 = hdul['S0'].data
    cc = (cs_mean.shape[0]-1)/2.
    x_p0 = cc+(p0[0]/step)*np.cos(np.radians(p0[1]))
    y_p0 = cc+(p0[0]/step)*np.sin(np.radians(p0[1]))
    x_s0 = cc+(s0[0]/step)*np.cos(np.radians(s0[1]))
    y_s0 = cc+(s0[0]/step)*np.sin(np.radians(s0[1]))
    x_rr = int(round(x_s0))
    y_rr = int(round(y_s0))
    if (x_rr < 0):
        x_rr = 0.
    if (y_rr < 0):
        y_rr = 0.
    if (x_rr >= cs_mean.shape[1]):
        x_rr = cs_mean.shape[1]-1
    if (y_rr >= cs_mean.shape[0]):
        y_rr = cs_mean.shape[0]-1
    
    temp = np.true_divide(cs_mean, cs_sdev)
    p0 = axarr[0].imshow(temp, cmap='hot', vmin=0, origin='lower', zorder=0)
    plt.colorbar(p0, ax=axarr[0])
    axarr[0].contour(temp, levels=[5], colors='white', zorder=1)
    c0 = plt.Circle((x_p0, y_p0), 25./step, color='black', lw=5, fill=False, zorder=3)
    axarr[0].add_artist(c0)
    c0 = plt.Circle((x_p0, y_p0), 25./step, color='magenta', lw=2.5, fill=False, zorder=3)
    axarr[0].add_artist(c0)
    c0 = plt.Circle((x_s0, y_s0), 25./step, color='black', lw=5, fill=False, zorder=3)
    axarr[0].add_artist(c0)
    c0 = plt.Circle((x_s0, y_s0), 25./step, color='cyan', lw=2.5, fill=False, zorder=3)
    axarr[0].add_artist(c0)
    axarr[0].text(0.05, 0.05, '$SNR_{ph}$ = %.1f$\sigma$' % temp[y_rr, x_rr]+' (%.0f KL)' % K_klip+'\n%.0f mas, %.0f deg, c = %.4f' % (s0[0], s0[1], s0[2]), ha='left', va='bottom', transform=axarr[0].transAxes, bbox=dict(facecolor='white', alpha=0.75), zorder=2)
    axarr[0].set_xticks(ticks)
    axarr[0].set_xticklabels(ticklabels)
    axarr[0].set_xlabel('$\Delta$DEC [mas]')
    axarr[0].set_yticks(ticks)
    axarr[0].set_yticklabels(ticklabels)
    axarr[0].set_ylabel('$\Delta$RA [mas]')
    
    temp = chi2s_c
    temp[seps < 25] = np.nan
    p1 = axarr[1].imshow(temp, cmap='cubehelix', origin='lower', zorder=0)
    plt.colorbar(p1, ax=axarr[1])
    c1 = plt.Circle((x_p0, y_p0), 25./step, color='black', lw=5, fill=False, zorder=3)
    axarr[1].add_artist(c1)
    c1 = plt.Circle((x_p0, y_p0), 25./step, color='magenta', lw=2.5, fill=False, zorder=3)
    axarr[1].add_artist(c1)
    c1 = plt.Circle((x_s0, y_s0), 25./step, color='black', lw=5, fill=False, zorder=3)
    axarr[1].add_artist(c1)
    c1 = plt.Circle((x_s0, y_s0), 25./step, color='cyan', lw=2.5, fill=False, zorder=3)
    axarr[1].add_artist(c1)
    axarr[1].text(0.05, 0.05, '$\chi^2$ = %.1f (raw)' % s0[3]+'\n$\chi^2$ = %.1f (bin)' % s0[4], ha='left', va='bottom', transform=axarr[1].transAxes, bbox=dict(facecolor='white', alpha=0.75), zorder=2)
    axarr[1].set_xticks(ticks)
    axarr[1].set_xticklabels(ticklabels)
    axarr[1].set_xlabel('$\Delta$DEC [mas]')
    axarr[1].set_yticks(ticks)
    axarr[1].set_yticklabels(ticklabels)
    axarr[1].set_ylabel('$\Delta$RA [mas]')
    
    plt.tight_layout()
    plt.savefig(pbfile[:-5]+'_gridsearch.pdf')
    plt.show()
    plt.close()
    hdul.close()
    
    pass

def plot_multiklip(pbfile):
    """
    TODO
    """
    
    hdul = pyfits.open(pbfile, memmap=False)
    kps = hdul['KP-DATA'].data
    icvs = hdul['KP-SIGM INV'].data
    seps = hdul['MULTIKLIP'].data[4, 0]
    pas = hdul['MULTIKLIP'].data[5, 0]
    ramp = seps[0]*np.cos(np.radians(pas[0]))
    step = (ramp[-1]-ramp[0])/(ramp.shape[0]-1.)
    ticks = np.arange(ramp.shape[0])[::20]
    ticklabels = ['%.0f' % f for f in ramp[::20]]
    
    K_klip = ast.literal_eval(hdul['MULTIKLIP'].header['KKLIP'])
    nk = len(K_klip)
    f, axarr = plt.subplots(nk, 2, figsize=(4*2, 3*nk))
    for i in range(nk):
        
        cs_mean = hdul['MULTIKLIP'].data[0, i]
        cs_sdev = hdul['MULTIKLIP'].data[1, i]
        chi2s = hdul['MULTIKLIP'].data[2, i]
        chi2s_c = hdul['MULTIKLIP'].data[3, i]
        p0 = hdul['P0S'].data[i]
        s0 = hdul['S0S'].data[i]
        cc = (cs_mean.shape[0]-1)/2.
        x_p0 = cc+(p0[0]/step)*np.cos(np.radians(p0[1]))
        y_p0 = cc+(p0[0]/step)*np.sin(np.radians(p0[1]))
        x_s0 = cc+(s0[0]/step)*np.cos(np.radians(s0[1]))
        y_s0 = cc+(s0[0]/step)*np.sin(np.radians(s0[1]))
        x_rr = int(round(x_s0))
        y_rr = int(round(y_s0))
        if (x_rr < 0):
            x_rr = 0.
        if (y_rr < 0):
            y_rr = 0.
        if (x_rr >= cs_mean.shape[1]):
            x_rr = cs_mean.shape[1]-1
        if (y_rr >= cs_mean.shape[0]):
            y_rr = cs_mean.shape[0]-1
        
        temp = np.true_divide(cs_mean, cs_sdev)
        pi0 = axarr[i, 0].imshow(temp, cmap='hot', vmin=0, origin='lower', zorder=0)
        plt.colorbar(pi0, ax=axarr[i, 0])
        axarr[i, 0].contour(temp, levels=[5], colors='white', zorder=1)
        ci0 = plt.Circle((x_p0, y_p0), 25./step, color='black', lw=5, fill=False, zorder=3)
        axarr[i, 0].add_artist(ci0)
        ci0 = plt.Circle((x_p0, y_p0), 25./step, color='magenta', lw=2.5, fill=False, zorder=3)
        axarr[i, 0].add_artist(ci0)
        ci0 = plt.Circle((x_s0, y_s0), 25./step, color='black', lw=5, fill=False, zorder=3)
        axarr[i, 0].add_artist(ci0)
        ci0 = plt.Circle((x_s0, y_s0), 25./step, color='cyan', lw=2.5, fill=False, zorder=3)
        axarr[i, 0].add_artist(ci0)
        axarr[i, 0].text(0.05, 0.05, '$SNR_{ph}$ = %.1f$\sigma$' % temp[y_rr, x_rr]+' (%.0f KL)' % K_klip[i]+'\n%.0f mas, %.0f deg, c = %.4f' % (s0[0], s0[1], s0[2]), ha='left', va='bottom', transform=axarr[i, 0].transAxes, bbox=dict(facecolor='white', alpha=0.75), zorder=2)
        axarr[i, 0].set_xticks(ticks)
        axarr[i, 0].set_xticklabels(ticklabels)
        axarr[i, 0].set_xlabel('$\Delta$DEC [mas]')
        axarr[i, 0].set_yticks(ticks)
        axarr[i, 0].set_yticklabels(ticklabels)
        axarr[i, 0].set_ylabel('$\Delta$RA [mas]')
        
        temp = chi2s_c
        temp[seps < 25] = np.nan
        pi1 = axarr[i, 1].imshow(temp, cmap='cubehelix', origin='lower', zorder=0)
        plt.colorbar(pi1, ax=axarr[i, 1])
        ci1 = plt.Circle((x_p0, y_p0), 25./step, color='black', lw=5, fill=False, zorder=3)
        axarr[i, 1].add_artist(ci1)
        ci1 = plt.Circle((x_p0, y_p0), 25./step, color='magenta', lw=2.5, fill=False, zorder=3)
        axarr[i, 1].add_artist(ci1)
        ci1 = plt.Circle((x_s0, y_s0), 25./step, color='black', lw=5, fill=False, zorder=3)
        axarr[i, 1].add_artist(ci1)
        ci1 = plt.Circle((x_s0, y_s0), 25./step, color='cyan', lw=2.5, fill=False, zorder=3)
        axarr[i, 1].add_artist(ci1)
        axarr[i, 1].text(0.05, 0.05, '$\chi^2$ = %.1f (raw)' % s0[3]+'\n$\chi^2$ = %.1f (bin)' % s0[4], ha='left', va='bottom', transform=axarr[i, 1].transAxes, bbox=dict(facecolor='white', alpha=0.75), zorder=2)
        axarr[i, 1].set_xticks(ticks)
        axarr[i, 1].set_xticklabels(ticklabels)
        axarr[i, 1].set_xlabel('$\Delta$DEC [mas]')
        axarr[i, 1].set_yticks(ticks)
        axarr[i, 1].set_yticklabels(ticklabels)
        axarr[i, 1].set_ylabel('$\Delta$RA [mas]')
    
    plt.tight_layout()
    plt.savefig(pbfile[:-5]+'_multiklip.pdf')
    plt.show()
    plt.close()
    hdul.close()
    
    pass

def plot_empirical(pbfile):
    """
    TODO
    """
    
    hdul = pyfits.open(pbfile, memmap=False)
    seps = hdul['MULTIKLIP'].data[4, 0]
    pas = hdul['MULTIKLIP'].data[5, 0]
    ramp = seps[0]*np.cos(np.radians(pas[0]))
    step = (ramp[-1]-ramp[0])/(ramp.shape[0]-1.)
    ticks = np.arange(ramp.shape[0])[::20]
    ticklabels = ['%.0f' % f for f in ramp[::20]]
    
    K_klip = ast.literal_eval(hdul['MULTIKLIP'].header['KKLIP'])
    nk = len(K_klip)
    f, axarr = plt.subplots(nk, 3, figsize=(4*3, 3*nk))
    for i in range(nk):
        
        cs_mean = hdul['MULTIKLIP'].data[0, i]
        cs_sdev = hdul['MULTIKLIP'].data[1, i]
        cs_sdev_emp = hdul['EMPIRICAL'].data[i]
        chi2s = hdul['MULTIKLIP'].data[2, i]
        chi2s_c = hdul['MULTIKLIP'].data[3, i]
        p0 = hdul['P0S'].data[i]
        s0 = hdul['S0S'].data[i]
        cc = (cs_mean.shape[0]-1)/2.
        x_p0 = cc+(p0[0]/step)*np.cos(np.radians(p0[1]))
        y_p0 = cc+(p0[0]/step)*np.sin(np.radians(p0[1]))
        x_s0 = cc+(s0[0]/step)*np.cos(np.radians(s0[1]))
        y_s0 = cc+(s0[0]/step)*np.sin(np.radians(s0[1]))
        x_rr = int(round(x_s0))
        y_rr = int(round(y_s0))
        if (x_rr < 0):
            x_rr = 0.
        if (y_rr < 0):
            y_rr = 0.
        if (x_rr >= cs_mean.shape[1]):
            x_rr = cs_mean.shape[1]-1
        if (y_rr >= cs_mean.shape[0]):
            y_rr = cs_mean.shape[0]-1
        
        temp = np.true_divide(cs_mean, cs_sdev)
        pi0 = axarr[i, 0].imshow(temp, cmap='hot', vmin=0, origin='lower', zorder=0)
        plt.colorbar(pi0, ax=axarr[i, 0])
        axarr[i, 0].contour(temp, levels=[5], colors='white', zorder=1)
        ci0 = plt.Circle((x_p0, y_p0), 25./step, color='black', lw=5, fill=False, zorder=3)
        axarr[i, 0].add_artist(ci0)
        ci0 = plt.Circle((x_p0, y_p0), 25./step, color='magenta', lw=2.5, fill=False, zorder=3)
        axarr[i, 0].add_artist(ci0)
        ci0 = plt.Circle((x_s0, y_s0), 25./step, color='black', lw=5, fill=False, zorder=3)
        axarr[i, 0].add_artist(ci0)
        ci0 = plt.Circle((x_s0, y_s0), 25./step, color='cyan', lw=2.5, fill=False, zorder=3)
        axarr[i, 0].add_artist(ci0)
        axarr[i, 0].text(0.05, 0.05, '$SNR_{ph}$ = %.1f$\sigma$' % temp[y_rr, x_rr]+' (%.0f KL)' % K_klip[i]+'\n%.0f mas, %.0f deg, c = %.4f' % (s0[0], s0[1], s0[2]), ha='left', va='bottom', transform=axarr[i, 0].transAxes, bbox=dict(facecolor='white', alpha=0.75), zorder=2)
        axarr[i, 0].set_xticks(ticks)
        axarr[i, 0].set_xticklabels(ticklabels)
        axarr[i, 0].set_xlabel('$\Delta$RA [mas]')
        axarr[i, 0].set_yticks(ticks)
        axarr[i, 0].set_yticklabels(ticklabels)
        axarr[i, 0].set_ylabel('$\Delta$DEC [mas]')
        
        temp = np.true_divide(cs_mean, cs_sdev_emp)
        pi1 = axarr[i, 1].imshow(temp, cmap='hot', vmin=0, origin='lower', zorder=0)
        plt.colorbar(pi1, ax=axarr[i, 1])
        axarr[i, 1].contour(temp, levels=[5], colors='white', zorder=1)
        ci1 = plt.Circle((x_p0, y_p0), 25./step, color='black', lw=5, fill=False, zorder=3)
        axarr[i, 1].add_artist(ci1)
        ci1 = plt.Circle((x_p0, y_p0), 25./step, color='magenta', lw=2.5, fill=False, zorder=3)
        axarr[i, 1].add_artist(ci1)
        ci1 = plt.Circle((x_s0, y_s0), 25./step, color='black', lw=5, fill=False, zorder=3)
        axarr[i, 1].add_artist(ci1)
        ci1 = plt.Circle((x_s0, y_s0), 25./step, color='cyan', lw=2.5, fill=False, zorder=3)
        axarr[i, 1].add_artist(ci1)
        axarr[i, 1].text(0.05, 0.05, '$SNR_{emp}$ = %.1f$\sigma$' % temp[y_rr, x_rr]+' (%.0f KL)' % K_klip[i]+'\n%.0f mas, %.0f deg, c = %.4f' % (s0[0], s0[1], s0[2]), ha='left', va='bottom', transform=axarr[i, 1].transAxes, bbox=dict(facecolor='white', alpha=0.75), zorder=2)
        axarr[i, 1].set_xticks(ticks)
        axarr[i, 1].set_xticklabels(ticklabels)
        axarr[i, 1].set_xlabel('$\Delta$RA [mas]')
        axarr[i, 1].set_yticks(ticks)
        axarr[i, 1].set_yticklabels(ticklabels)
        axarr[i, 1].set_ylabel('$\Delta$DEC [mas]')
        
        temp = chi2s_c
        temp[seps < 25] = np.nan
        pi2 = axarr[i, 2].imshow(temp, cmap='cubehelix', origin='lower', zorder=0)
        plt.colorbar(pi2, ax=axarr[i, 2])
        ci2 = plt.Circle((x_p0, y_p0), 25./step, color='black', lw=5, fill=False, zorder=3)
        axarr[i, 2].add_artist(ci2)
        ci2 = plt.Circle((x_p0, y_p0), 25./step, color='magenta', lw=2.5, fill=False, zorder=3)
        axarr[i, 2].add_artist(ci2)
        ci2 = plt.Circle((x_s0, y_s0), 25./step, color='black', lw=5, fill=False, zorder=3)
        axarr[i, 2].add_artist(ci2)
        ci2 = plt.Circle((x_s0, y_s0), 25./step, color='cyan', lw=2.5, fill=False, zorder=3)
        axarr[i, 2].add_artist(ci2)
        axarr[i, 2].text(0.05, 0.05, '$\chi^2$ = %.1f (raw)' % s0[3]+'\n$\chi^2$ = %.1f (bin)' % s0[4], ha='left', va='bottom', transform=axarr[i, 2].transAxes, bbox=dict(facecolor='white', alpha=0.75), zorder=2)
        axarr[i, 2].set_xticks(ticks)
        axarr[i, 2].set_xticklabels(ticklabels)
        axarr[i, 2].set_xlabel('$\Delta$RA [mas]')
        axarr[i, 2].set_yticks(ticks)
        axarr[i, 2].set_yticklabels(ticklabels)
        axarr[i, 2].set_ylabel('$\Delta$DEC [mas]')
    
    plt.tight_layout()
    plt.savefig(pbfile[:-5]+'_empirical.pdf')
    plt.show()
    plt.close()
    hdul.close()
    
    cs_sdev_emp = hdul['EMPIRICAL'].data
    
    plt.figure()
    for i in range(nk):
        
        radii, detlim = azavg(cs_sdev_emp[i])
        f_log = interp1d(radii, np.log(detlim), kind='linear')
        rs = np.linspace(radii.min(), radii.max(), 1024)
        cs_curv = np.exp(f_log(rs))
        
        plt.plot(rs*step, cs_curv, label=str(K_klip[i]))
    
    plt.yscale('log')
    plt.xlabel('Separation [mas]')
    plt.ylabel('Empirical contrast limit')
    plt.legend()
    plt.tight_layout()
    plt.savefig(pbfile[:-5]+'_empirical_lim.pdf')
    plt.show()
    plt.close()
    
    pass

idir = '161108_pbfiles_test/'
pbfiles = [f for f in os.listdir(idir) if f.endswith('.fits')]
import pdb; pdb.set_trace()

for i in range(len(pbfiles)):
    plot_multiklip(idir+pbfiles[i])

#idir = '161109_emp/'
#pbfile = 'UZ Tau A.fits'
#
##plot_gridsearch(idir+pbfile)
##plot_multiklip(idir+pbfile)
#plot_empirical(idir+pbfile)
