import logging
import numpy as np
import awkward1 as ak
from coffea import processor, hist
from coffea.analysis_tools import Weights, PackedSelection
from boostedhiggs.btag import BTagEfficiency, BTagCorrector
from boostedhiggs.common import (
    getBosons,
    bosonFlavor,
)
from boostedhiggs.corrections import (
    corrected_msoftdrop,
    n2ddt_shift,
    add_pileup_weight,
    add_VJets_NLOkFactor,
    add_jetTriggerWeight,
)


logger = logging.getLogger(__name__)


class HbbProcessor(processor.ProcessorABC):
    def __init__(self, year='2017', jet_arbitration='pt'):
        self._year = year
        self._jet_arbitration = jet_arbitration

        self._btagSF = BTagCorrector(year, 'medium')

        self._msdSF = {
            '2016': 1.,
            '2017': 0.987,
            '2018': 0.970,
        }

        self._muontriggers = {
            '2016': [
                'Mu50',  # TODO: check
            ],
            '2017': [
                'Mu50',
                'TkMu50',
            ],
            '2018': [
                'Mu50',  # TODO: check
            ],
        }

        self._triggers = {
            '2016': [
                'PFHT800',
                'PFHT900',
                'AK8PFJet360_TrimMass30',
                'AK8PFHT700_TrimR0p1PT0p03Mass50',
                'PFHT650_WideJetMJJ950DEtaJJ1p5',
                'PFHT650_WideJetMJJ900DEtaJJ1p5',
                'AK8DiPFJet280_200_TrimMass30_BTagCSV_p20',
                'PFJet450',
            ],
            '2017': [
                'AK8PFJet330_PFAK8BTagCSV_p17',
                'PFHT1050',
                'AK8PFJet400_TrimMass30',
                'AK8PFJet420_TrimMass30',
                'AK8PFHT800_TrimMass50',
                'PFJet500',
                'AK8PFJet500',
            ],
            '2018': [
                'AK8PFJet400_TrimMass30',
                'AK8PFJet420_TrimMass30',
                'AK8PFHT800_TrimMass50',
                'PFHT1050',
                'PFJet500',
                'AK8PFJet500',
                # 'AK8PFJet330_PFAK8BTagCSV_p17', not present in 2018D?
                'AK8PFJet330_TrimMass30_PFAK8BoostedDoubleB_np4',
            ],
        }

        self._accumulator = processor.dict_accumulator({
            # dataset -> sumw
            'sumw': processor.defaultdict_accumulator(float),
            'cutflow': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('genflavor', 'Gen. jet flavor', [0, 1, 2, 3, 4]),
                hist.Bin('cut', 'Cut index', 11, 0, 11),
            ),
            'nminus1_n2ddt': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('n2ddt', 'N2ddt value', 40, -0.25, 0.25),
            ),
            'btagWeight': hist.Hist('Events', hist.Cat('dataset', 'Dataset'), hist.Bin('val', 'BTag correction', 50, 0, 3)),
            'templates': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Cat('systematic', 'Systematic'),
                hist.Bin('genflavor', 'Gen. jet flavor', [0, 1, 2, 3, 4]),
                hist.Bin('pt', r'Jet $p_{T}$ [GeV]', [450, 500, 550, 600, 675, 800, 1200]),
                hist.Bin('msd', r'Jet $m_{sd}$', 23, 40, 201),
                hist.Bin('ddb', r'Jet ddb score', [0, 0.89, 1]),
            ),
            'genresponse_noweight': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Cat('systematic', 'Systematic'),
                hist.Bin('pt', r'Jet $p_{T}$ [GeV]', [450, 500, 550, 600, 675, 800, 1200]),
                hist.Bin('genpt', r'Generated Higgs $p_{T}$ [GeV]', [200, 300, 450, 650, 7500]),
            ),
            'genresponse': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Cat('systematic', 'Systematic'),
                hist.Bin('pt', r'Jet $p_{T}$ [GeV]', [450, 500, 550, 600, 675, 800, 1200]),
                hist.Bin('genpt', r'Generated Higgs $p_{T}$ [GeV]', [200, 300, 450, 650, 7500]),
            ),
        })

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        dataset = events.metadata['dataset']
        isRealData = not hasattr(events, "genWeight")
        selection = PackedSelection()
        weights = Weights(len(events))
        output = self.accumulator.identity()
        if not isRealData:
            output['sumw'][dataset] += ak.sum(events.genWeight)

        if isRealData:
            trigger = np.zeros(len(events), dtype='bool')
            for t in self._triggers[self._year]:
                trigger = trigger | events.HLT[t]
        else:
            trigger = np.ones(len(events), dtype='bool')
        selection.add('trigger', trigger)

        if isRealData:
            trigger = np.zeros(len(events), dtype='bool')
            for t in self._muontriggers[self._year]:
                trigger = trigger | events.HLT[t]
        else:
            trigger = np.ones(len(events), dtype='bool')
        selection.add('muontrigger', trigger)

        fatjets = events.FatJet
        fatjets['msdcorr'] = corrected_msoftdrop(fatjets)
        fatjets['qcdrho'] = 2 * np.log(fatjets.msdcorr / fatjets.pt)
        fatjets['n2ddt'] = fatjets.n2b1 - n2ddt_shift(fatjets, year=self._year)
        fatjets['msdcorr_full'] = fatjets['msdcorr'] * self._msdSF[self._year]

        candidatejet = fatjets[
            # https://github.com/DAZSLE/BaconAnalyzer/blob/master/Analyzer/src/VJetLoader.cc#L269
            (fatjets.pt > 200)
            & (abs(fatjets.eta) < 2.5)
            & fatjets.isTight  # this is loose in sampleContainer
        ]
        if self._jet_arbitration == 'pt':
            candidatejet = ak.firsts(candidatejet)
        elif self._jet_arbitration == 'mass':
            candidatejet = candidatejet[
                ak.argmax(candidatejet.msdcorr)
            ]
        elif self._jet_arbitration == 'n2':
            candidatejet = candidatejet[
                ak.argmin(candidatejet.n2ddt)
            ]
        elif self._jet_arbitration == 'ddb':
            candidatejet = candidatejet[
                ak.argmax(candidatejet.btagDDBvL)
            ]
        else:
            raise RuntimeError("Unknown candidate jet arbitration")

        selection.add('minjetkin',
            (candidatejet.pt >= 450)
            & (candidatejet.msdcorr >= 40.)
            & (abs(candidatejet.eta) < 2.5)
        )
        selection.add('jetacceptance',
            (candidatejet.msdcorr >= 47.)
            & (candidatejet.pt < 1200)
            & (candidatejet.msdcorr < 201.)
        )
        selection.add('jetid', candidatejet.isTight)
        selection.add('n2ddt', (candidatejet.n2ddt < 0.))
        selection.add('ddbpass', (candidatejet.btagDDBvL >= 0.89))

        jets = events.Jet[
            (events.Jet.pt > 30.)
            & (abs(events.Jet.eta) < 2.5)
            & events.Jet.isTight
        ]
        # only consider first 4 jets to be consistent with old framework
        jets = jets[:, :4]
        dphi = abs(jets.delta_phi(candidatejet))
        selection.add('antiak4btagMediumOppHem', ak.max(jets[dphi > np.pi / 2].btagDeepB, axis=1, mask_identity=False) < BTagEfficiency.btagWPs[self._year]['medium'])
        ak4_away = jets[dphi > 0.8]
        selection.add('ak4btagMedium08', ak.max(ak4_away.btagDeepB, axis=1, mask_identity=False) > BTagEfficiency.btagWPs[self._year]['medium'])

        selection.add('met', events.MET.pt < 140.)

        goodmuon = (
            (events.Muon.pt > 10)
            & (abs(events.Muon.eta) < 2.4)
            & (events.Muon.pfRelIso04_all < 0.25)
            & events.Muon.looseId
        )
        nmuons = ak.sum(goodmuon, axis=1)
        leadingmuon = ak.firsts(events.Muon[goodmuon])

        nelectrons = ak.sum(
            (events.Electron.pt > 10)
            & (abs(events.Electron.eta) < 2.5)
            & (events.Electron.cutBased >= events.Electron.LOOSE),
            axis=1,
        )

        ntaus = ak.sum(
            (events.Tau.pt > 20)
            & events.Tau.idDecayMode,  # bacon iso looser than Nano selection
            axis=1,
        )

        selection.add('noleptons', (nmuons == 0) & (nelectrons == 0) & (ntaus == 0))
        selection.add('onemuon', (nmuons == 1) & (nelectrons == 0) & (ntaus == 0))
        selection.add('muonkin', (leadingmuon.pt > 55.) & (abs(leadingmuon.eta) < 2.1))
        selection.add('muonDphiAK8', abs(leadingmuon.delta_phi(candidatejet)) > 2*np.pi/3)

        if isRealData:
            genflavor = 0
        else:
            weights.add('genweight', events.genWeight)
            add_pileup_weight(weights, events.Pileup.nPU, self._year, dataset)
            bosons = getBosons(events.GenPart)
            matchedBoson = candidatejet.nearest(bosons, axis=None, threshold=0.8)
            genflavor = bosonFlavor(matchedBoson)
            genBosonPt = ak.fill_none(ak.firsts(bosons.pt), 0)
            add_VJets_NLOkFactor(weights, genBosonPt, self._year, dataset)
            add_jetTriggerWeight(weights, candidatejet.msdcorr, candidatejet.pt, self._year)
            output['btagWeight'].fill(dataset=dataset, val=self._btagSF.addBtagWeight(weights, ak4_away))
            logger.debug("Weight statistics: %r" % weights.weightStatistics)

        msd_matched = candidatejet.msdcorr * self._msdSF[self._year] * (genflavor > 0) + candidatejet.msdcorr * (genflavor == 0)

        regions = {
            'signal': ['trigger', 'minjetkin', 'jetacceptance', 'jetid', 'n2ddt', 'antiak4btagMediumOppHem', 'met', 'noleptons'],
            'muoncontrol': ['muontrigger', 'minjetkin', 'jetacceptance', 'jetid', 'n2ddt', 'ak4btagMedium08', 'onemuon', 'muonkin', 'muonDphiAK8'],
            'noselection': [],
        }

        for region, cuts in regions.items():
            allcuts = set()
            output['cutflow'].fill(dataset=dataset, region=region, genflavor=genflavor, cut=0, weight=weights.weight())
            for i, cut in enumerate(cuts + ['ddbpass']):
                allcuts.add(cut)
                cut = selection.all(*allcuts)
                output['cutflow'].fill(dataset=dataset, region=region, genflavor=genflavor[cut], cut=i + 1, weight=weights.weight()[cut])

        systematics = [
            None,
            'jet_triggerUp',
            'jet_triggerDown',
            'btagWeightUp',
            'btagWeightDown',
            'btagEffStatUp',
            'btagEffStatDown',
        ]

        def normalize(val, cut):
            return ak.to_numpy(ak.fill_none(val[cut], np.nan))

        def fill(region, systematic, wmod=None):
            selections = regions[region]
            cut = selection.all(*selections)
            sname = 'nominal' if systematic is None else systematic
            if wmod is None:
                weight = weights.weight(modifier=systematic)[cut]
            else:
                weight = weights.weight()[cut] * wmod[cut]

            output['templates'].fill(
                dataset=dataset,
                region=region,
                systematic=sname,
                genflavor=genflavor[cut],
                pt=normalize(candidatejet.pt, cut),
                msd=normalize(msd_matched, cut),
                ddb=normalize(candidatejet.btagDDBvL, cut),
                weight=weight,
            )
            if wmod is not None:
                output['genresponse_noweight'].fill(
                    dataset=dataset,
                    region=region,
                    systematic=sname,
                    pt=normalize(candidatejet.pt, cut),
                    genpt=normalize(genBosonPt, cut),
                    weight=events.genWeight[cut] * wmod[cut],
                )
                output['genresponse'].fill(
                    dataset=dataset,
                    region=region,
                    systematic=sname,
                    pt=normalize(candidatejet.pt, cut),
                    genpt=normalize(genBosonPt, cut),
                    weight=weight,
                )

        for region in regions:
            cut = selection.all(*(set(regions[region]) - {'n2ddt'}))
            output['nminus1_n2ddt'].fill(
                dataset=dataset,
                region=region,
                n2ddt=normalize(candidatejet.n2ddt, cut),
                weight=weights.weight()[cut],
            )
            for systematic in systematics:
                fill(region, systematic)
            if 'GluGluHToBB' in dataset:
                for i in range(9):
                    fill(region, 'LHEScale_%d' % i, events.LHEScaleWeight[:, i])
                for c in events.LHEWeight.columns[1:]:
                    fill(region, 'LHEWeight_%s' % c, events.LHEWeight[c])

        output["weightStats"] = weights.weightStatistics
        return output

    def postprocess(self, accumulator):
        return accumulator
