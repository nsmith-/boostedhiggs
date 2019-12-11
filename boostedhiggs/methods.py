import numpy as np
import awkward as ak
from uproot_methods.classes.TLorentzVector import PtEtaPhiMassArrayMethods
from uproot_methods.classes.TVector2 import ArrayMethods as XYArrayMethods


def _memoize(obj, name, constructor):
    memoname = '_memo_' + name
    if memoname not in obj.columns:
        obj[memoname] = constructor(obj)
    return obj[memoname]


class METVector(XYArrayMethods):
    def __getitem__(self, key):
        if ak.AwkwardArray._util_isstringslice(key) and key == 'fX':
            return _memoize(self, 'fX', lambda self: self['pt'] * np.cos(self['phi']))
        elif ak.AwkwardArray._util_isstringslice(key) and key == 'fY':
            return _memoize(self, 'fY', lambda self: self['pt'] * np.sin(self['phi']))
        return super(METVector, self).__getitem__(key)


class LorentzVector(PtEtaPhiMassArrayMethods):
    _keymap = {'fPt': 'pt', 'fEta': 'eta', 'fPhi': 'phi', 'fMass': 'mass'}

    def __getitem__(self, key):
        if ak.AwkwardArray._util_isstringslice(key) and key in self._keymap:
            if key == 'fMass' and 'mass' not in self.columns:
                return _memoize(self, 'fMass', lambda self: self['pt'].zeros_like())
            return self[self._keymap[key]]
        return super(LorentzVector, self).__getitem__(key)


class Candidate(LorentzVector):
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        out = super(Candidate, self).__array_ufunc__(ufunc, method, *inputs, **kwargs)
        if ufunc is np.add and all(isinstance(i, Candidate) for i in inputs):
            out['charge'] = getattr(ufunc, method)(*(i['charge'] for i in inputs), **kwargs)
        # TODO else: type demotion?
        return out


class Electron(Candidate):
    FAIL, VETO, LOOSE, MEDIUM, TIGHT = range(5)

    @property
    def isLoose(self):
        return (self.cutBased >= self.LOOSE).astype(bool)


class Muon(Candidate):
    pass


class Photon(Candidate):
    LOOSE, MEDIUM, TIGHT = range(3)

    @property
    def isLoose(self):
        return (self.cutBasedBitmap & (1 << self.LOOSE)).astype(bool)


class Tau(Candidate):
    pass


class GenParticle(LorentzVector):
    pass
