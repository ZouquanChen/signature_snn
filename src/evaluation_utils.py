"""Shared numerical helpers for signature-verification evaluation."""

import numpy


def compute_eer(far, frr):
    """Return a stable EER estimate and fractional threshold index.

    Linear interpolation is used at a FAR/FRR crossing.  If a finite sampled
    curve does not contain a crossing, the closest operating point is returned
    instead of reporting a false zero or NaN EER.
    """
    far = numpy.asarray(far, dtype=numpy.float64).reshape(-1)
    frr = numpy.asarray(frr, dtype=numpy.float64).reshape(-1)
    if far.size == 0 or far.size != frr.size:
        raise ValueError("FAR and FRR must be non-empty arrays of equal length")
    finite = numpy.isfinite(far) & numpy.isfinite(frr)
    if not numpy.any(finite):
        raise ValueError("FAR/FRR curves contain no finite operating point")
    far = far[finite]
    frr = frr[finite]
    diff = far - frr

    exact = numpy.flatnonzero(diff == 0)
    if exact.size:
        index = int(exact[0])
        return numpy.array([far[index], float(index)], dtype=numpy.float64)

    crossings = numpy.flatnonzero(diff[:-1] * diff[1:] < 0)
    if crossings.size:
        left = int(crossings[0])
        denominator = diff[left] - diff[left + 1]
        fraction = diff[left] / denominator
        far_interp = far[left] + fraction * (far[left + 1] - far[left])
        frr_interp = frr[left] + fraction * (frr[left + 1] - frr[left])
        eer = 0.5 * (far_interp + frr_interp)
        return numpy.array([eer, left + fraction], dtype=numpy.float64)

    index = int(numpy.argmin(numpy.abs(diff)))
    eer = 0.5 * (far[index] + frr[index])
    return numpy.array([eer, float(index)], dtype=numpy.float64)
