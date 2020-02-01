from functools import partial
import numpy as np
from coffea import processor, hist
from .common import (
    getBosons,
    matchedBosonFlavor,
)
from .corrections import (
    corrected_msoftdrop,
    n2ddt_shift,
    add_pileup_weight,
    add_VJets_NLOkFactor,
    add_jetTriggerWeight,
)


class HbbProcessor(processor.ProcessorABC):
    def __init__(self, year='2017'):
        self._year = year

        self._btagWPs = {
            '2016': {'med': 0.6321},
            '2017': {'med': 0.4941},
            '2018': {'med': 0.4184},
        }

        self._muontriggers = {
            '2016': [
                'Mu50',  # TODO: check
            ],
            '2017': [
                'Mu50',
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
            # dataset -> cut -> count
            'cutflow': processor.defaultdict_accumulator(partial(processor.defaultdict_accumulator, float)),
            # dataset -> sumw
            'sumw': processor.defaultdict_accumulator(float),
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
        })

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        dataset = events.metadata['dataset']
        isRealData = 'genWeight' not in events.columns
        selection = processor.PackedSelection()

        trigger = np.ones(events.size, dtype='bool')
        for t in self._triggers[self._year]:
            trigger = trigger | events.HLT[t]
        selection.add('trigger', trigger)

        trigger = np.ones(events.size, dtype='bool')
        for t in self._muontriggers[self._year]:
            trigger = trigger | events.HLT[t]
        selection.add('muontrigger', trigger)

        fatjets = events.FatJet
        fatjets['msdcorr'] = corrected_msoftdrop(fatjets)
        fatjets['rho'] = 2 * np.log(fatjets.msdcorr / fatjets.pt)
        fatjets['n2ddt'] = fatjets.n2b1 - n2ddt_shift(fatjets, year=self._year)

        candidatejet = fatjets[
            # https://github.com/DAZSLE/BaconAnalyzer/blob/master/Analyzer/src/VJetLoader.cc#L269
            (fatjets.pt > 200)
            & (abs(fatjets.eta) < 2.5)
            & (fatjets.jetId & 2).astype(bool)  # this is tight rather than loose
        ][:, 0:1]
        selection.add('jetkin', (
            (candidatejet.pt > 450)
            & (abs(candidatejet.eta) < 2.4)
            & (candidatejet.msdcorr > 40.)
        ).any())
        selection.add('jetid', (candidatejet.jetId & 2).any())  # tight id
        selection.add('n2ddt', (candidatejet.n2ddt < 0.).any())

        jets = events.Jet[
            (events.Jet.pt > 30.)
            & (events.Jet.jetId & 2)  # tight id
        ]
        # only consider first 4 jets to be consistent with old framework
        jets = jets[:, :4]
        ak4_ak8_pair = jets.cross(candidatejet, nested=True)
        dphi = ak4_ak8_pair.i0.delta_phi(ak4_ak8_pair.i1)
        ak4_opposite = jets[(abs(dphi) > np.pi / 2).all()]
        selection.add('antiak4btagMediumOppHem', ak4_opposite.btagDeepB.max() < self._btagWPs[self._year]['med'])
        ak4_away = jets[(abs(dphi) > 0.8).all()]
        selection.add('ak4btagMedium08', ak4_away.btagDeepB.max() > self._btagWPs[self._year]['med'])

        selection.add('met', events.MET.pt < 140.)

        goodmuon = (
            (events.Muon.pt > 10)
            & (abs(events.Muon.eta) < 2.4)
            & (events.Muon.pfRelIso04_all < 0.25)
            & (events.Muon.looseId).astype(bool)
        )
        nmuons = goodmuon.sum()
        leadingmuon = events.Muon[goodmuon][:, 0:1]
        muon_ak8_pair = leadingmuon.cross(candidatejet, nested=True)

        nelectrons = (
            (events.Electron.pt > 10)
            & (abs(events.Electron.eta) < 2.5)
            & (events.Electron.cutBased >= events.Electron.LOOSE)
        ).sum()

        ntaus = (
            (events.Tau.pt > 20)
            & (events.Tau.idDecayMode).astype(bool)
            # bacon iso looser than Nano selection
        ).sum()

        selection.add('noleptons', (nmuons == 0) & (nelectrons == 0) & (ntaus == 0))
        selection.add('onemuon', (nmuons == 1) & (nelectrons == 0) & (ntaus == 0))
        selection.add('muonkin', (
            (leadingmuon.pt > 55.)
            & (abs(leadingmuon.eta) < 2.1)
        ).all())
        selection.add('muonDphiAK8', (
            muon_ak8_pair.i0.delta_phi(muon_ak8_pair.i1) > 2*np.pi/3
        ).all().all())

        weights = processor.Weights(len(events))
        if isRealData:
            genflavor = candidatejet.pt.zeros_like()
        else:
            weights.add('genweight', events.genWeight)
            add_pileup_weight(weights, events.Pileup.nPU, self._year, dataset)
            bosons = getBosons(events)
            genBosonPt = bosons.pt.pad(1, clip=True).fillna(0)
            add_VJets_NLOkFactor(weights, genBosonPt, self._year, dataset)
            genflavor = matchedBosonFlavor(candidatejet, bosons)
            add_jetTriggerWeight(weights, candidatejet.msdcorr, candidatejet.pt, self._year)

        output = self.accumulator.identity()
        if not isRealData:
            output['sumw'][dataset] += events.genWeight.sum()

        regions = {
            'signal': ['jetkin', 'trigger', 'jetid', 'n2ddt', 'antiak4btagMediumOppHem', 'met', 'noleptons'],
            'muoncontrol': ['jetkin', 'muontrigger', 'jetid', 'n2ddt', 'ak4btagMedium08', 'onemuon', 'muonkin', 'muonDphiAK8'],
        }

        allcuts = set()
        output['cutflow'][dataset]['none'] += float(weights.weight().sum())
        for cut in regions['muoncontrol']:
            allcuts.add(cut)
            output['cutflow'][dataset][cut] += float(weights.weight()[selection.all(*allcuts)].sum())

        systematics = [None, 'jet_triggerUp', 'jet_triggerDown']

        for region in regions:
            for systematic in systematics:
                selections = regions[region]
                cut = selection.all(*selections)
                output['templates'].fill(
                    dataset=dataset,
                    region=region,
                    systematic='nominal' if systematic is None else systematic,
                    genflavor=genflavor[cut].flatten(),
                    pt=candidatejet[cut].pt.flatten(),
                    msd=candidatejet[cut].msdcorr.flatten(),
                    ddb=candidatejet[cut].btagDDBvL.flatten(),
                    weight=weights.weight(modifier=systematic)[cut],
                )

        return output

    def postprocess(self, accumulator):
        return accumulator
