import numpy as np
import awkward as ak
from .methods import (
    METVector,
    LorentzVector,
    Candidate,
    Electron,
    Muon,
    Photon,
    Tau,
)


def _mixin(methods, awkwardtype):
    '''Like ak.Methods.mixin but also captures methods in dir() and propagate docstr'''
    newtype = type(methods.__name__ + 'Array', (methods, awkwardtype), {})
    newtype.__dir__ = lambda self: dir(methods) + awkwardtype.__dir__(self)
    newtype.__doc__ = methods.__doc__
    return newtype


class NanoCollection(ak.VirtualArray):
    @classmethod
    def _lazyflatten(cls, array):
        return array.array.content

    @classmethod
    def from_arrays(cls, arrays, name, methods=None):
        '''
        arrays : object
            An object with attributes: columns, __len__, and __getitem__
            where the latter returns virtual arrays or virtual jagged arrays
        '''
        jagged = 'n' + name in arrays.columns
        columns = {k[len(name) + 1:]: arrays[k] for k in arrays.columns if k.startswith(name + '_')}
        if len(columns) == 0:
            # single-item collection, just forward lazy array (possibly jagged)
            if name not in arrays.columns:
                raise RuntimeError('Could not find collection %s in dataframe' % name)
            if methods:
                ArrayType = _mixin(methods, type(arrays[name]))
                return ArrayType(arrays[name])
            return arrays[name]
        elif not jagged:
            if methods is None:
                Table = ak.Table
            else:
                Table = _mixin(methods, ak.Table)
            table = Table.named(name)
            for k, v in columns.items():
                table[k] = v
            return table
        else:  # jagged
            if methods:
                cls = _mixin(methods, cls)
            tabletype = ak.type.TableType()
            for k, array in columns.items():
                tabletype[k] = array.type.to.to
            counts = arrays['n' + name]
            out = cls(
                cls._lazyjagged,
                (name, counts, columns, methods),
                type=ak.type.ArrayType(len(arrays), float('inf'), tabletype),
            )
            out.__doc__ = counts.__doc__
            return out

    @classmethod
    def _lazyjagged(cls, name, counts, columns, methods=None):
        offsets = ak.JaggedArray.counts2offsets(counts.array)
        if methods is None:
            JaggedArray = ak.JaggedArray
            Table = ak.Table
        else:
            JaggedArray = _mixin(methods, ak.JaggedArray)
            Table = _mixin(methods, ak.Table)
        table = Table.named(name)
        for k, v in columns.items():
            if not isinstance(v, ak.VirtualArray):
                raise RuntimeError
            col = type(v)(NanoCollection._lazyflatten, (v,), type=ak.type.ArrayType(offsets[-1], v.type.to.to))
            col.__doc__ = v.__doc__
            table[k] = col
        out = JaggedArray.fromoffsets(offsets, table)
        out.__doc__ = counts.__doc__
        return out

    def _lazyindexed(self, indices, destination):
        if not isinstance(destination.array, ak.JaggedArray):
            raise RuntimeError
        if not isinstance(self.array, ak.JaggedArray):
            raise NotImplementedError
        content = np.zeros(len(self.array.content) * len(indices), dtype=ak.JaggedArray.INDEXTYPE)
        for i, k in enumerate(indices):
            content[i::len(indices)] = np.array(self.array.content[k])
        globalindices = ak.JaggedArray.fromoffsets(
            self.array.offsets,
            content=ak.JaggedArray.fromoffsets(
                np.arange((len(self.array.content) + 1) * len(indices), step=len(indices)),
                content,
            )
        )
        globalindices = globalindices[globalindices >= 0] + destination.array.starts
        out = globalindices.copy(
            content=type(destination.array).fromoffsets(
                globalindices.content.offsets,
                content=destination.array.content[globalindices.flatten().flatten()]
            )
        )
        return out

    def __setitem__(self, key, value):
        if self.ismaterialized:
            super(NanoCollection, self).__setitem__(key, value)
        _, _, columns, _ = self._args
        columns[key] = value
        self._type.to.to[key] = value.type.to.to

    def __delitem__(self, key):
        if self.ismaterialized:
            super(NanoCollection, self).__delitem__(key)
        _, _, columns, _ = self._args
        del columns[key]
        del self._type.to.to[key]


class NanoEvents(ak.Table):
    collection_methods = {
        'CaloMET': METVector,
        'ChsMET': METVector,
        'GenMET': METVector,
        'MET': METVector,
        'METFixEE2017': METVector,
        'PuppiMET': METVector,
        'RawMET': METVector,
        'TkMET': METVector,
        # pseudo-lorentz: pt, eta, phi, mass=0
        'IsoTrack': LorentzVector,
        'SoftActivityJet': LorentzVector,
        'TrigObj': LorentzVector,
        # True lorentz: pt, eta, phi, mass
        'FatJet': LorentzVector,
        'GenDressedLepton': LorentzVector,
        'GenJet': LorentzVector,
        'GenJetAK8': LorentzVector,
        'GenPart': LorentzVector,
        'Jet': LorentzVector,
        'LHEPart': LorentzVector,
        'SV': LorentzVector,
        'SubGenJetAK8': LorentzVector,
        'SubJet': LorentzVector,
        # Candidate: LorentzVector + charge
        'Electron': Electron,
        'Muon': Muon,
        'Photon': Photon,
        'Tau': Tau,
        'GenVisTau': Candidate,
    }

    @classmethod
    def from_arrays(cls, arrays, collection_methods_overrides={}):
        events = cls.named('event')
        collections = {k.split('_')[0] for k in arrays.columns}
        collections -= {k for k in collections if k.startswith('n') and k[1:] in collections}
        allmethods = {**cls.collection_methods, **collection_methods_overrides}
        for name in collections:
            methods = allmethods.get(name, None)
            events[name] = NanoCollection.from_arrays(arrays, name, methods)

        # finalize
        del events.Photon['mass']

        embedded_subjets = type(events.SubJet)(
            events.FatJet._lazyindexed,
            args=(['subJetIdx1', 'subJetIdx2'], events.SubJet),
            type=ak.type.ArrayType(len(events), float('inf'), float('inf'), events.SubJet.type.to.to),
        )
        embedded_subjets.__doc__ = events.SubJet.__doc__
        events.FatJet['subjets'] = embedded_subjets

        return events
