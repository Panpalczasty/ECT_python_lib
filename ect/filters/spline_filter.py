from scipy.interpolation import CubicSpline

from .utils import vector_gen

def splinefilt_rho(dsize, radius, knot_values, num_knots):
    
    if len(knot_values) != num_knots:
        raise ValueError(f"Knot values array of invalid shape: needed {num_knots}, got {len(knot_values)}.")

    knots = np.linspace(1, radius, num_knots)
    
    polyfilt = CubicSpline(x=knots, y=knot_values, bc_type='natural')

    rhos, _ = vector_gen(shape=dsize)

    return polyfilt(rhos, extrapolate=True)


def splinefilt_phi(dsize, knot_values, num_knots):

    if len(knot_values) != num_knots:
            raise ValueError(f"Knot values array of invalid shape: needed {num_knots}, got {len(knot_values)}.")

    # knot_values = list(knot_values)
    # knot_values += [1] + knot_values[::-1]
    # knot_values += knot_values
    # num_knots = len(knot_values)

    knots = np.linspace(0, 1, num_knots)
    
    polyfilt = CubicSpline(x=knots, y=knot_values, bc_type='natural')

    _, phis = vector_gen(shape=dsize)

    return polyfilt(phis)
