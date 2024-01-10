from typing import Iterable

from .spline_filter import splinefilt_rho, splinefilt_phi
from .utils import *
from .weights import *

def freqnorm(
    dsize: tuple[int, int],
    radius: int,
    knot_values: Iterable[float] = DEFAULT_FNF):

    # knot_zeros = [0] * int(len(knot_values))
    # knot_values = np.r_[knot_zeros, knot_values]

    num_knots = len(knot_values)
    return np.exp(splinefilt_rho(dsize, radius, knot_values, num_knots))


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


def spacenorm_phi(dsize, knot_values: list = DEFAULT_SPHI):
    return splinefilt_phi(dsize, knot_values, len(knot_values))


def freqnorm_phi(dsize, knot_values: list = DEFAULT_FPHI):
    return np.exp(splinefilt_phi(dsize, knot_values, len(knot_values)))
