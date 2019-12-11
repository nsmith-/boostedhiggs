import numpy as np
import awkward as ak
import numba

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


@numba.njit
def _find_distinctParent(pdg_self, pdg_all, parent_self, parent_all):
    out = parent_self.copy()
    for i in range(len(out)):
        if out[i] < 0:
            continue
        thispdg = pdg_self[i]
        parent = parent_self[i]
        parentpdg = pdg_all[parent]
        while parent >= 0 and parentpdg == thispdg:
            parent = parent_all[parent]
            parentpdg = pdg_all[parent]
        out[i] = parent

    return out


class GenParticle(LorentzVector):
    @property
    def distinctParent(self):
        array = self
        jagged = None
        if isinstance(array, ak.VirtualArray):
            array = array.array
        if isinstance(array, ak.JaggedArray):
            jagged = array
            array = array.content
        pdg_self = np.array(array.pdgId)
        if isinstance(array.parent, ak.VirtualArray):
            parent = array.parent.array
        else:
            parent = array.parent
        parent_self = parent.mask
        pdg_all = np.array(parent.content.pdgId)
        parent_all = parent.content['_%s_globalindex' % self.rowname]
        globalindex = _find_distinctParent(pdg_self, pdg_all, parent_self, parent_all)
        out = ak.IndexedMaskedArray(
            globalindex,
            parent.content,
        )
        if jagged is not None:
            out = jagged.copy(content=out)
        return out
