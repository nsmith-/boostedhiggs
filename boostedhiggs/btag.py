import os
import pandas
import numpy
import numexpr
import numba
from coffea import processor, hist, util
from coffea.lookup_tools.dense_lookup import dense_lookup


class BTagEfficiency(processor.ProcessorABC):
    btagWPs = {
        '2016': {
            'medium': 0.6321,
        },
        '2017': {
            'medium': 0.4941,
        },
        '2018': {
            'medium': 0.4184,
        },
    }

    def __init__(self, year='2017'):
        self._year = year
        self._accumulator = hist.Hist(
            'Events',
            hist.Cat('btag', 'BTag WP pass/fail'),
            hist.Bin('flavor', 'Jet hadronFlavour', [0, 4, 5, 6]),
            hist.Bin('pt', 'Jet pT', [30, 50, 70, 100, 140, 200, 500]),
            hist.Bin('eta', 'Jet abseta', 4, 0, 2.4),
        )

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        jets = events.Jet[
            (events.Jet.pt > 30.)
            & (events.Jet.jetId & 2)  # tight id
        ]

        passbtag = jets.btagDeepB > BTagEfficiency.btagWPs[self._year]['medium']

        out = self.accumulator.identity()
        out.fill(
            btag='pass',
            flavor=jets[passbtag].hadronFlavour.flatten(),
            pt=jets[passbtag].pt.flatten(),
            eta=abs(jets[passbtag].eta).flatten(),
        )
        out.fill(
            btag='fail',
            flavor=jets[~passbtag].hadronFlavour.flatten(),
            pt=jets[~passbtag].pt.flatten(),
            eta=abs(jets[~passbtag].eta).flatten(),
        )
        return out

    def postprocess(self, a):
        return a


class BTagScaleFactor:
    LOOSE, MEDIUM, TIGHT, RESHAPE = range(4)
    wpString = {'loose': LOOSE, 'medium': MEDIUM, 'tight': TIGHT}

    def __init__(self, year, workingpoint, lightmethod='comb'):
        try:
            workingpoint = BTagScaleFactor.wpString[workingpoint]
        except KeyError:
            pass
        if workingpoint == 3:
            raise NotImplementedError('Reshape corrections are not yet supported')
        elif workingpoint not in [0, 1, 2, 3]:
            raise ValueError('Unrecognized working point')
        if lightmethod not in ['comb', 'mujets']:
            raise ValueError('Unrecognized light jet correction method')
        self.year = year
        self.workingpoint = workingpoint
        files = {
            '2017': 'DeepCSV_94XSF_V5_B_F.csv.gz',
        }
        csvfile = os.path.join(os.path.dirname(__file__), 'data', files[year])
        df = pandas.read_csv(csvfile, skipinitialspace=True)
        for var in ['eta', 'pt', 'discr']:
            df[var + 'Bin'] = list(zip(df[var + 'Min'], df[var + 'Max']))
            del df[var + 'Min']
            del df[var + 'Max']
        df = df[df['DeepCSV;OperatingPoint'] == workingpoint]
        df = df[(df['measurementType'] == lightmethod) | (df['jetFlavor'] == 2)]
        df = df.set_index(['sysType', 'jetFlavor', 'etaBin', 'ptBin', 'discrBin']).sort_index()
        self._corrections = {}
        self._compiled = {}
        for syst in list(df.index.levels[0]):
            corr = df.loc[syst]
            edges_flavor = numpy.array([0, 1, 2, 3])  # udsg, c, b
            edges_eta = numpy.array(sorted(set(x for tup in corr.index.levels[1] for x in tup)))
            edges_pt = numpy.array(sorted(set(x for tup in corr.index.levels[2] for x in tup)))
            edges_discr = numpy.array(sorted(set(x for tup in corr.index.levels[3] for x in tup)))
            alledges = numpy.meshgrid(edges_flavor[:-1], edges_eta[:-1], edges_pt[:-1], edges_discr[:-1], indexing='ij')
            mapping = numpy.full(alledges[0].shape, -1)

            def findbin(flavor, eta, pt, discr):
                for i, (fbin, ebin, pbin, dbin) in enumerate(corr.index):
                    if flavor == fbin and ebin[0] <= eta < ebin[1] and pbin[0] <= pt < pbin[1] and dbin[0] <= discr < dbin[1]:
                        return i
                return -1

            for idx, _ in numpy.ndenumerate(mapping):
                flavor, eta, pt, discr = (x[idx] for x in alledges)
                mapidx = findbin(flavor, eta, pt, discr)
                if mapidx < 0 and flavor == 2:
                    # sometimes btag flavor is abs(eta)
                    eta = eta + 1e-5  # add eps to avoid edge effects
                    mapidx = findbin(flavor, abs(eta), pt, discr)
                mapping[idx] = mapidx

            self._corrections[syst] = (
                edges_flavor,
                edges_eta,
                edges_pt,
                edges_discr,
                mapping,
                numpy.array(corr['formula']),
            )
            self._compiled[syst] = [BTagScaleFactor.compile(f) for f in self._corrections[syst][5]]

    def __getstate__(self):
        state = dict(self.__dict__)
        state.pop('_compiled')
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self._compiled = {}
        for syst in self._corrections:
            self._compiled[syst] = [BTagScaleFactor.compile(f) for f in self._corrections[syst][5]]

    @classmethod
    def compile(cls, formula):
        if 'x' in formula:
            feval = eval('lambda x: ' + formula, {'log': numpy.log, 'sqrt': numpy.sqrt})
            return numba.vectorize([
                numba.float32(numba.float32),
                numba.float64(numba.float64),
            ])(feval)
        val = numexpr.evaluate(formula)

        def duck(_, out, where):
            out[where] = val
        return duck

    def lookup(self, axis, values):
        return numpy.clip(numpy.searchsorted(axis, values, side='right') - 1, 0, len(axis) - 2)

    def eval(self, systematic, flavor, eta, pt, discr=None, ignore_missing=False):
        try:
            flavor.counts
            jin, flavor = flavor, flavor.flatten()
            eta = eta.flatten()
            pt = pt.flatten()
            discr = discr.flatten() if discr is not None else None
        except AttributeError:
            jin = None
        corr = self._corrections[systematic]
        idx = (
            self.lookup(corr[0], flavor),
            self.lookup(corr[1], eta),
            self.lookup(corr[2], pt),
            self.lookup(corr[3], discr) if discr is not None else 0,
        )
        mapidx = corr[4][idx]
        out = numpy.zeros_like(pt)
        for ifunc in numpy.unique(mapidx):
            if ifunc < 0 and not ignore_missing:
                raise ValueError('No correction was available for some items')
            func = self._compiled[systematic][ifunc]
            var = discr if self.workingpoint == BTagScaleFactor.RESHAPE else pt
            func(var, out=out, where=(mapidx == ifunc))

        if jin is not None:
            out = jin.copy(content=out)
        return out


class BTagCorrector:
    def __init__(self, year, workingpoint):
        self._year = year
        self._wp = BTagEfficiency.btagWPs[year][workingpoint]
        self.sf = BTagScaleFactor(year, workingpoint)
        files = {
            '2017': 'btagQCD2017.coffea',
        }
        file = os.path.join(os.path.dirname(__file__), 'data', files[year])
        btag = util.load(file)
        bpass = btag.integrate('btag', 'pass').values()[()]
        ball = btag.integrate('btag').values()[()]
        nom = bpass / ball
        dn, up = hist.clopper_pearson_interval(bpass, ball)
        self.eff = dense_lookup(nom, [ax.edges() for ax in btag.axes()[1:]])
        self.eff_statUp = dense_lookup(up, [ax.edges() for ax in btag.axes()[1:]])
        self.eff_statDn = dense_lookup(dn, [ax.edges() for ax in btag.axes()[1:]])

    def addBtagWeight(self, weights, jets):
        # 0 = udsg, 1 or 4 = c, 2 or 5 = b
        flavor = jets.hadronFlavour % 3
        passbtag = jets.btagDeepB > self._wp

        # https://twiki.cern.ch/twiki/bin/viewauth/CMS/BTagSFMethods#1a_Event_reweighting_using_scale
        def combine(eff, sf):
            # tagged SF = SF*eff / eff = SF
            tagged_sf = sf[passbtag].prod()
            # untagged SF = (1 - SF*eff) / (1 - eff)
            untagged_sf = ((1 - sf*eff) / (1 - eff))[~passbtag].prod()
            return tagged_sf * untagged_sf

        eff_nom = self.eff(jets.hadronFlavour, jets.pt, abs(jets.eta))
        eff_statUp = self.eff_statUp(jets.hadronFlavour, jets.pt, abs(jets.eta))
        eff_statDn = self.eff_statDn(jets.hadronFlavour, jets.pt, abs(jets.eta))
        sf_nom = self.sf.eval('central', flavor, jets.eta, jets.pt)
        sf_systUp = self.sf.eval('up', flavor, jets.eta, jets.pt)
        sf_systDn = self.sf.eval('down', flavor, jets.eta, jets.pt)

        nom = combine(eff_nom, sf_nom)
        weights.add('btagWeight', nom, weightUp=combine(eff_nom, sf_systUp), weightDown=combine(eff_nom, sf_systDn))
        weights.add('btagEffStat', numpy.ones_like(nom), weightUp=combine(eff_statUp, sf_nom) / nom, weightDown=combine(eff_statDn, sf_nom) / nom)
        for i in numpy.where(nom < 0.05)[0][:4]:
            jet = jets[i]
            print("Small weight for event:", nom[i], jet.pt, jet.eta, jet.hadronFlavour, jet.btagDeepB, eff_nom[i], sf_nom[i])
        return nom


if __name__ == '__main__':
    b = BTagScaleFactor('2017', BTagScaleFactor.MEDIUM)
    b.eval('central', numpy.array([0, 1, 2]), numpy.array([-2.3, 2., 0.]), numpy.array([20.1, 300., 10.]))
    b.eval('down_uncorrelated', numpy.array([2, 2, 2]), numpy.array([-2.6, 2.9, 0.]), numpy.array([20.1, 300., 1000.]))
    import awkward as ak
    b.eval('central', ak.fromiter([[0], [1, 2]]), ak.fromiter([[-2.3], [2., 0.]]), ak.fromiter([[20.1], [300., 10.]]))
    b = BTagCorrector('2017', 'medium')
    import pickle
    bb = pickle.loads(pickle.dumps(b))
    bb.sf.eval('central', ak.fromiter([[0], [1, 2]]), ak.fromiter([[-2.3], [2., 0.]]), ak.fromiter([[20.1], [300., 10.]]))
