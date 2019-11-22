import numpy as np
import awkward as ak
from coffea.util import load

compiled = load(__file__.replace('.py', '.coffea'))


def _msoftdrop_weight(pt, eta):
    gpar = np.array([1.00626, -1.06161, 0.0799900, 1.20454])
    cpar = np.array([1.09302, -0.000150068, 3.44866e-07, -2.68100e-10, 8.67440e-14, -1.00114e-17])
    fpar = np.array([1.27212, -0.000571640, 8.37289e-07, -5.20433e-10, 1.45375e-13, -1.50389e-17])
    genw = gpar[0] + gpar[1]*np.power(pt*gpar[2], -gpar[3])
    ptpow = np.power.outer(pt, np.arange(cpar.size))
    cenweight = np.dot(ptpow, cpar)
    forweight = np.dot(ptpow, fpar)
    weight = np.where(np.abs(eta) < 1.3, cenweight, forweight)
    return genw*weight


def corrected_msoftdrop(fatjets):
    if not isinstance(fatjets, ak.JaggedArray):
        raise ValueError
    sf_flat = _msoftdrop_weight(fatjets.p4.pt.flatten(), fatjets.p4.eta.flatten())
    sf_flat = np.maximum(1e-5, sf_flat)
    return fatjets.msoftdrop * fatjets.copy(content=sf_flat)


def n2ddt_shift(fatjets, year='2017'):
    return compiled[f'{year}_n2ddt_rho_pt'](fatjets.rho, fatjets.p4.pt)
