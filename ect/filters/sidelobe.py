import numpy as np
import math

from .utils import *

def sidelobe(
    shape: tuple[int,int],
    slope: float = 0.5, 
    offset: float = 20.0,
    flags: int = ECT_OMIT_ORIGIN | ECT_GRAYSCALE | ECT_START_NY
) -> np.ndarray:
    """Generates sidelobe filter in logpolar domain using
    the following equation

    f(rho,phi) = 1/(1+exp((a-exp(rho)*cos(phi))/slope)) + 1/(1+exp((a+exp(rho)*cos(phi))/slope)) 

    Args:
        shape (tuple): Shape of output array
        radius (float): Max radius of logpolar transform

    Returns:
        np.ndarray: Sidelobe filter for 
    """

    if flags & ECT_START_NY:
        phi = np.linspace(-0.5*math.pi, 1.5*math.pi, shape[0])    
    else:
        phi = np.linspace(0, 2*math.pi, shape[0])
    

    if flags & ECT_INCLUDE_ORIGIN:
        rho = np.linspace(0, math.log(shape[1]), shape[1])
        m_rho, m_phi = np.meshgrid(rho, phi)
        m_x = (np.exp(m_rho)-1)*np.cos(m_phi)
    else:
        rho = np.linspace(1, math.log(shape[1]), shape[1])
        m_rho, m_phi = np.meshgrid(rho, phi)
        m_x = np.exp(m_rho)*np.cos(m_phi)

    # if flags & ECT_OFFSET_ORIGIN:
    #     P, R = shape
    #     m_x[:P//2, :] += offset
    #     m_x[P//2:, :] -= offset
    
    out = sigmoid((m_x-offset)/slope) + sigmoid((-m_x-offset)/slope)
    # out = sigmoid((m_x-2*offset)/slope) + sigmoid((-m_x-2*offset)/slope)
    

    if flags & ECT_RGB:
        return np.stack((out, out, out), axis=-1)
    else:
        return np.expand_dims(out, 2)