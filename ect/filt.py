import math
import numpy as np

# origin mode flags
ECT_INCLUDE_ORIGIN = 1 
ECT_OMIT_ORIGIN = 2
ECT_OFFSET_ORIGIN = 4

# interpolation flags
ECT_RGB = 8
ECT_GRAYSCALE = 16

# angle flags
ECT_START_PX = 32
ECT_START_NY = 64

def sigmoid(x: float) -> float:
    """Calculated sigmoid function of an x

    Args:
        x (float): input

    Returns:
        float: output
    """
    x = np.clip(x, -100, 100)
    return 1/(1 + np.exp(-x))


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
    
    out = sigmoid((m_x-offset)/slope) + sigmoid((-m_x-offset)/slope)
    
    if flags & ECT_RGB:
        return np.stack((out, out, out), axis=-1)
    else:
        return np.expand_dims(out, 2)

