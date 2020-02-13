import os
import numpy
from coffea import processor, hist, util
from coffea.lookup_tools.dense_lookup import dense_lookup
from coffea.btag_tools import BTagScaleFactor


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


class BTagCorrector:
    def __init__(self, year, workingpoint):
        self._year = year
        self._wp = BTagEfficiency.btagWPs[year][workingpoint]
        files = {
            '2017': 'DeepCSV_94XSF_V5_B_F.csv.gz',
        }
        filename = os.path.join(os.path.dirname(__file__), 'data', files[year])
        self.sf = BTagScaleFactor(filename, workingpoint)
        files = {
            '2017': 'btagQCD2017.coffea',
        }
        filename = os.path.join(os.path.dirname(__file__), 'data', files[year])
        btag = util.load(filename)
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
    b = BTagCorrector('2017', 'medium')
    b.sf.eval('central', numpy.array([0, 1, 2]), numpy.array([-2.3, 2., 0.]), numpy.array([20.1, 300., 10.]))
    b.sf.eval('down_uncorrelated', numpy.array([2, 2, 2]), numpy.array([-2.6, 2.9, 0.]), numpy.array([20.1, 300., 1000.]))
    import awkward as ak
    b.sf.eval('central', ak.fromiter([[0], [1, 2]]), ak.fromiter([[-2.3], [2., 0.]]), ak.fromiter([[20.1], [300., 10.]]))
    import pickle
    bb = pickle.loads(pickle.dumps(b))
    bb.sf.eval('central', ak.fromiter([[0], [1, 2]]), ak.fromiter([[-2.3], [2., 0.]]), ak.fromiter([[20.1], [300., 10.]]))
