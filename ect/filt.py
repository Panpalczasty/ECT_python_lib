import math
import numpy as np
from scipy.interpolate import CubicSpline

from typing import Iterable

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

# DEFAULT_ANG = [0.08819866, 0.13181471, 0.20496488, 0.29324702, 0.3755276 ,
#        0.45355567, 0.52609183, 0.59462407, 0.66007136, 0.72165186,
#        0.78169232, 0.83761392, 0.88578629, 0.92918131, 0.96587902,
#        0.98862107, 1.00247915, 0.99946212]

DEFAULT_ANG = [1, 1]

# tune this on whitenoise
DEFAULT_FNF = [2.23546183, 0.02089067, 0.59506666, 0.24344334, 0.3490346 ,
       0.24784119, 0.22250579, 0.20775864, 0.21038973, 0.22876757,
       0.33631792, 0.27018188, 0.28442893, 0.30625369, 0.32004706,
       0.30109776, 0.28106156, 0.27342773, 0.26489311, 0.32855774]

# tune this on white
DEFAULT_SNF = [-0.12253074,  0.08635138, -0.11990753,  0.15919272, -0.12419914,
        2.7911736 ,  3.51461431,  3.45581535,  3.45001963,  3.41822584,
        3.39092369,  3.35342852,  3.29070184,  3.2462659 ,  3.22734647,
        3.28645839,  3.25459136,  3.24035778,  3.25257675,  3.29658477]

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


def spacenorm(
    dsize: tuple[int, int],
    radius: int,
    knot_values: Iterable[float] = DEFAULT_SNF):

    if knot_values is None:
        knot_values = DEFAULT_SNF

    # knot_zeros = [1] * int(len(knot_values)*0.5)
    # knot_values = np.r_[knot_zeros, knot_values]

    num_knots = len(knot_values)
    return splinefilt_rho(dsize, radius, knot_values, num_knots)


def freqnorm(
    dsize: tuple[int, int],
    radius: int,
    knot_values: Iterable[float] = DEFAULT_FNF):


    if knot_values is None:
        knot_values = DEFAULT_FNF


    # knot_zeros = [0] * int(len(knot_values))
    # knot_values = np.r_[knot_zeros, knot_values]

    num_knots = len(knot_values)
    return np.exp(splinefilt_rho(dsize, radius, knot_values, num_knots))
    

def splinefilt_rho(dsize, radius, knot_values, num_knots):
    
    if len(knot_values) != num_knots:
        raise ValueError(f"Knot values array of invalid shape: needed {num_knots}, got {len(knot_values)}.")

    knots = np.linspace(1, radius, num_knots)
    
    polyfilt = CubicSpline(x=knots, y=knot_values, bc_type='natural')

    rhos, _ = vector_gen(shape=dsize)

    return polyfilt(rhos, extrapolate=True)


def angular_filter(dsize, knot_values: list = DEFAULT_ANG):

    # print(f"{knot_values=}")
    # knot_values[-1] = knot_values[0]
    knot_values = list(knot_values)
    knot_values += [1] + knot_values[::-1]
    knot_values += knot_values
    num_knots = len(knot_values)
    # print(f"{knot_values=}")
    knots = np.linspace(0, 1, num_knots)
    # print(knots)
    _, phis = vector_gen(shape=dsize)

    polyfilt = CubicSpline(x=knots, y=knot_values, bc_type='periodic')
    return polyfilt(phis)

def vector_gen(shape: tuple[int, int]):
    '''
    Generates rho vector.

    Parameters
    ----------
    shape : tuple[int, int]
        shape of kernel image

    Returns
    -------
    np.ndarray
        Rho vector
    '''

    P, R = shape

    rho = np.linspace(1, R, R)
    phi = np.linspace(0, 1, P)

    rhos, phis, _ = np.meshgrid(rho, phi, 0)

    return rhos, phis