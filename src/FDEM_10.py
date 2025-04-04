# Import libraries

import numpy as np
import empymod 
from scipy.constants import mu_0
import pygimli as pg

# Forward function for the DUALEM instrument
def FDEM_10(sigmas, thicks, coil_orient=np.array(['H', 'P', 'V']), height=0.15):

    """
    Forward function for the DUALEM instrument

    Inputs:
    sigmas : Electrical conductivities array in S/m
    thicks : Thicknesses array in m
    coil_orient :  coil orientations
    height : height of the instrument
    
    Returns array [HOP, HIP, VOP, VIP, POP, PIP] 

    """

    Freq = np.logspace(2,4.5,15)
    coil_spacing = [2, 4, 8]
    coil_spacing_p = [2.1, 4.1, 8.2]
    
    res_air = 1e6 # air resistivity
    
    sigmas = np.array(sigmas)
    res = np.hstack(([res_air], 1/sigmas))    
    depth = np.hstack(([0],-np.cumsum(thicks)))
        
    # Define source and receivers geometry
    
    source = [0, 0, -height]
    receivers = [coil_spacing, np.zeros_like(coil_spacing), -height]
    receivers_p = [coil_spacing_p, np.zeros_like(coil_spacing_p), -height]
       
    # Empty array to store store responses
    OUT = []
    
    # Calculate for horizontal coil orientation
    if any(coil_orient == 'H'):
        # Secondary magnetic field
        H_Hs = empymod.dipole(source, receivers, depth, res, Freq, ab = 66, xdirect = None, 
                              verb=0)
        # Primary magnetic field
        H_Hp = empymod.dipole(source, receivers, depth=[], res=[res_air], freqtime = Freq,
                              ab = 66, verb=0) 
        op = (H_Hs/H_Hp).imag * 1e3 # Out of Phase
        ip = (H_Hs/H_Hp).real * 1e3 # In Phase
        OUT.append([op, ip])

    # Calculate for vertical coil orientation
    if any(coil_orient == 'V'):
        # Secondary magnetic field
        V_Hs = empymod.dipole(source, receivers, depth, res, Freq, ab = 55, xdirect = None, 
                              verb=0)
        # Primary magnetic field
        V_Hp = empymod.dipole(source, receivers, depth=[], res=[res_air], freqtime = Freq, ab = 55, 
                              verb=0)
        op = (V_Hs/V_Hp).imag * 1e3 # Out of Phase
        ip = (V_Hs/V_Hp).real * 1e3 # In Phase
        OUT.append([op, ip])

    # Calculate for perpendicular coil orientation
    if any(coil_orient == 'P'):
        P_Hs = empymod.dipole(source, receivers, depth, res, Freq, ab = 46, xdirect = None, 
                              verb=0) 
        P_Hp = empymod.dipole(source, receivers, depth=[], res=[res_air], freqtime= Freq,
                              ab = 66, verb = 0)
        op = (P_Hs/P_Hp).imag * 1e3 # Out of Phase
        ip = (P_Hs/P_Hp).real * 1e3 # In Phase

        OUT.append([op, ip])

    # Returns array [HOP, HIP, VOP, VIP, POP, PIP] 
    # Shape (ncoil, nphase, nfreq, nrec) 
    return np.array(OUT).ravel()

class FDEM_1D_10(pg.frameworks.Modelling):
    """ Class to Initialize the model for Gauss-Newton inversion
    using the quadrature (Q) and in-phase (IP) components of the measurements
    for a n-layered model
    
    Input:

        nlay : number of layers
    """   
    def __init__(self, nlay=3):
        self.nlay = nlay
        mesh = pg.meshtools.createMesh1DBlock(nlay)
        super().__init__()
        self.setMesh(mesh)

    def response(self, par):
        """ Compute response vector for a certain model [mod] 
        par = [thickness_1, thickness_2, ..., thickness_n, sigma_1, sigma_2, ..., sigma_n]
        """
      #  print('model:', par)  
        resp = FDEM_10(sigmas = par[self.nlay-1:],
                           thicks = par[:self.nlay-1]
                              )
        return resp
    
    def response_mt(self, par, i=0):
        """Multi-threaded forward response."""
        return self.response(par)
        
    def drawModel(self, ax, model):
        pg.viewer.mpl.drawModel1D(ax = ax,
                                  model = model,
                                  plot = 'semilogx',
                                  xlabel = 'Electrical conductivity (S/m)',
                                  )
        ax.set_ylabel('Depth in (m)')

