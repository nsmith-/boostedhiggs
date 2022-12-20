import os
import numpy
import logging
import awkward
import hist
from hist.intervals import clopper_pearson_interval
from coffea import processor, util
from coffea.lookup_tools.dense_lookup import dense_lookup
from coffea.btag_tools import BTagScaleFactor


logger = logging.getLogger(__name__)


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

    def process(self, events):
        jets = events.Jet[
            (events.Jet.pt > 30.)
            & (abs(events.Jet.eta) < 2.5)
            & (events.Jet.jetId & 2)  # tight id
        ]

        passbtag = jets.btagDeepB > BTagEfficiency.btagWPs[self._year]['medium']

        out = hist.Hist(
            hist.axis.StrCat(name='btag', label='BTag WP pass/fail'),
            hist.axis.IntCat([0, 4, 5, 6], name='flavor', label='Jet hadronFlavour'),
            hist.axis.Variable([20, 30, 50, 70, 100, 140, 200, 300, 600, 1000], name='pt', label='Jet pT'),
            hist.axis.Variable([0, 1.4, 2.0, 2.5], name='abseta', label='Jet abseta'),
            label='Events',
        )
        out.fill(
            btag='pass',
            flavor=jets[passbtag].hadronFlavour.flatten(),
            pt=jets[passbtag].pt.flatten(),
            abseta=abs(jets[passbtag].eta.flatten()),
        )
        out.fill(
            btag='fail',
            flavor=jets[~passbtag].hadronFlavour.flatten(),
            pt=jets[~passbtag].pt.flatten(),
            abseta=abs(jets[~passbtag].eta.flatten()),
        )
        return out

    def postprocess(self, a):
        return a


class BTagCorrector:
    def __init__(self, year, workingpoint):
        self._year = year
        self._wp = BTagEfficiency.btagWPs[year][workingpoint]
        files = {
            '2016': 'DeepCSV_Moriond17_B_H.csv.gz',
            '2017': 'DeepCSV_94XSF_V5_B_F.csv.gz',
            '2018': 'DeepCSV_102XSF_V1.csv.gz',
        }
        filename = os.path.join(os.path.dirname(__file__), 'data', files[year])
        self.sf = BTagScaleFactor(filename, workingpoint)
        files = {
            '2016': 'btagQCD2016.coffea',
            '2017': 'btagQCD2017.coffea',
            '2018': 'btagQCD2018.coffea',
        }
        filename = os.path.join(os.path.dirname(__file__), 'data', files[year])
        btag = util.load(filename)
        bpass = btag["pass", :, :, :].values()
        ball = btag[::sum, :, :, :].values()
        nom = bpass / numpy.maximum(ball, 1.)
        dn, up = clopper_pearson_interval(bpass, ball)
        self.eff = dense_lookup(nom, [ax.edges for ax in btag.axes[1:]])
        self.eff_statUp = dense_lookup(up, [ax.edges for ax in btag.axes[1:]])
        self.eff_statDn = dense_lookup(dn, [ax.edges for ax in btag.axes[1:]])

    def addBtagWeight(self, weights, jets):
        abseta = abs(jets.eta)
        passbtag = jets.btagDeepB > self._wp

        # https://twiki.cern.ch/twiki/bin/viewauth/CMS/BTagSFMethods#1a_Event_reweighting_using_scale
        def combine(eff, sf):
            # tagged SF = SF*eff / eff = SF
            tagged_sf = awkward.prod(sf[passbtag], axis=-1)
            # untagged SF = (1 - SF*eff) / (1 - eff)
            untagged_sf = awkward.prod(((1 - sf*eff) / (1 - eff))[~passbtag], axis=-1)
            return awkward.fill_none(tagged_sf * untagged_sf, 1.)  # TODO: move None guard to coffea

        eff_nom = self.eff(jets.hadronFlavour, jets.pt, abseta)
        eff_statUp = self.eff_statUp(jets.hadronFlavour, jets.pt, abseta)
        eff_statDn = self.eff_statDn(jets.hadronFlavour, jets.pt, abseta)
        sf_nom = self.sf.eval('central', jets.hadronFlavour, abseta, jets.pt)
        sf_systUp = self.sf.eval('up', jets.hadronFlavour, abseta, jets.pt)
        sf_systDn = self.sf.eval('down', jets.hadronFlavour, abseta, jets.pt)

        nom = combine(eff_nom, sf_nom)
        weights.add('btagWeight', nom, weightUp=combine(eff_nom, sf_systUp), weightDown=combine(eff_nom, sf_systDn))
        weights.add('btagEffStat', numpy.ones(len(nom)), weightUp=combine(eff_statUp, sf_nom) / nom, weightDown=combine(eff_statDn, sf_nom) / nom)
        for i in numpy.where((nom < 0.01) | (nom > 10) | numpy.isnan(nom))[0][:4]:
            jet = jets[i]
            logger.info("Strange weight for event: %r", nom[i])
            logger.info("    jet pts: %r", jet.pt)
            logger.info("    jet etas: %r", jet.eta)
            logger.info("    jet flavors: %r", jet.hadronFlavour)
            logger.info("    jet btags: %r", jet.btagDeepB)
            logger.info("    result eff: %r up %r down %r", eff_nom[i], eff_statUp[i], eff_statDn[i])
            logger.info("    result sf: %r", sf_nom[i])
        return nom


if __name__ == '__main__':
    b = BTagCorrector('2017', 'medium')
    b.sf.eval('central', numpy.array([0, 1, 2]), numpy.array([-2.3, 2., 0.]), numpy.array([20.1, 300., 10.]))
    b.sf.eval('down_uncorrelated', numpy.array([2, 2, 2]), numpy.array([-2.6, 2.9, 0.]), numpy.array([20.1, 300., 1000.]))
    import awkward as ak
    b.sf.eval('central', ak.Array([[0], [1, 2]]), ak.Array([[-2.3], [2., 0.]]), ak.Array([[20.1], [300., 10.]]))
    import pickle
    bb = pickle.loads(pickle.dumps(b))
    bb.sf.eval('central', ak.Array([[0], [1, 2]]), ak.Array([[-2.3], [2., 0.]]), ak.Array([[20.1], [300., 10.]]))
    b2 = BTagCorrector('2016', 'medium')
    b3 = BTagCorrector('2018', 'medium')
