import torch
import numpy as np

from sgkigp.config import InterpType


# Prototype linear basis function: value 1 at zero, and ramps linearly
# down to value 0 at +-1
def linear_kernel(dist):
    vals = np.zeros_like(dist)
    inds = dist < 1
    vals[inds] = 1-dist[inds]
    return vals


# Prototype cubic basis function centered at zero:
# positive between [-1, +1], negative in [-2, -1] and [1, 2], and zero outside of [-2, 2]
def cubic_kernel(dist, a=-0.5):
    # Formula from:
    #   https://en.wikipedia.org/wiki/Bicubic_interpolation#Bicubic_convolution_algorithm
    #
    # Original source:
    #   R. Keys (1981). "Cubic convolution interpolation for digital image processing".
    #   IEEE Transactions on Acoustics, Speech, and Signal Processing. 29 (6): 1153–1160.
    #   doi:10.1109/TASSP.1981.1163711.

    vals = np.zeros_like(dist)

    inds = dist <= 1
    vals[inds] = (a + 2) * dist[inds] ** 3 - (a + 3) * dist[inds] ** 2 + 1

    inds = (1 < dist) & (dist < 2)
    vals[inds] = a * dist[inds] ** 3 - 5 * a * dist[inds] ** 2 + 8 * a * dist[inds] - 4 * a
    return vals


def eval_interp_kernel(kind, dist):

    # Now evaluate the kernel
    if kind == InterpType.CUBIC:
        vals = cubic_kernel(dist)
    elif kind == InterpType.LINEAR:
        vals = linear_kernel(dist)
    else:
        raise NotImplementedError
    return vals


class InterpKernels(object):

    @staticmethod
    def _linear_interpolation_kernel(scaled_grid_dist):
        return (1 - scaled_grid_dist) * (scaled_grid_dist < 1)

    @staticmethod
    def _cubic_interpolation_kernel(dist, a=-0.5):
        vals = torch.zeros_like(dist)
        inds = dist <= 1
        vals[inds] = (a + 2) * dist[inds] ** 3 - (a + 3) * dist[inds] ** 2 + 1
        inds = (1 < dist) & (dist < 2)
        vals[inds] = a * dist[inds] ** 3 - 5 * a * dist[inds] ** 2 + 8 * a * dist[inds] - 4 * a
        return vals

    def apply_kernel(self, dist, interp_type=InterpType.LINEAR):

        if interp_type == InterpType.LINEAR:
            return self._linear_interpolation_kernel(dist)
        elif interp_type == InterpType.CUBIC:
            return self._cubic_interpolation_kernel(dist)
        else:
            raise NotImplementedError

    def get_offsets(self, interp_type, device, dtype):

        if interp_type == InterpType.CUBIC:
            offsets = torch.Tensor([-1.0, 0.0, 1.0, 2.0])

        elif interp_type == InterpType.LINEAR:
            offsets = torch.Tensor([0.0, 1.0])

        else:
            raise NotImplementedError
        return offsets.to(dtype=dtype, device=device)
