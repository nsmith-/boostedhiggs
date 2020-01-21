import numpy as np
from coffea import processor
from .structure import buildevents
from .corrections import (
    corrected_msoftdrop,
    n2ddt_shift,
    add_pileup_weight,
    add_VJets_NLOkFactor,
)


class HbbProcessor(processor.ProcessorABC):
    def __init__(self, year='2017'):
        self._year = year

        self._btagWPs = {
            'med': {
                '2016': 0.6321,
                '2017': 0.4941,
                '2018': 0.4184,
            },
        }

        self._triggers = {
            '2016': [
                "PFHT800",
                "PFHT900",
                "AK8PFJet360_TrimMass30",
                'AK8PFHT700_TrimR0p1PT0p03Mass50',
                "PFHT650_WideJetMJJ950DEtaJJ1p5",
                "PFHT650_WideJetMJJ900DEtaJJ1p5",
                "AK8DiPFJet280_200_TrimMass30_BTagCSV_p20",
                "PFJet450",
            ],
            '2017': [
                "AK8PFJet330_PFAK8BTagCSV_p17",
                "PFHT1050",
                "AK8PFJet400_TrimMass30",
                "AK8PFJet420_TrimMass30",
                "AK8PFHT800_TrimMass50",
                "PFJet500",
                "AK8PFJet500",
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
            'cutflow': processor.defaultdict_accumulator(float),

        })

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        dataset = events.metadata['dataset']
        isRealData = 'genWeight' not in events.columns
        output = self.accumulator.identity()
        selection = processor.PackedSelection()

        trigger = np.ones(events.size, dtype='bool')
        for t in self._triggers[self._year]:
            trigger = trigger & events.HLT[t]
        selection.add('trigger', trigger)

        fatjets = events.FatJet
        fatjets['msdcorr'] = corrected_msoftdrop(fatjets)
        fatjets['rho'] = 2*np.log(fatjets.msdcorr / fatjets.pt)
        fatjets['n2ddt'] = fatjets.n2b1 - n2ddt_shift(fatjets, year=self._year)

        candidatejet = fatjets[:, 0:1]
        selection.add('jetkin', (
            (candidatejet.pt > 450)
            & (candidatejet.eta < 2.4)
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
        ak4_opposite = jets[(np.abs(dphi) > np.pi / 2).all()]
        selection.add('antiak4btagMediumOppHem', ak4_opposite.btagDeepB.max() < self._btagWPs['med'][self._year])
        ak4_away = jets[(np.abs(dphi) > 0.8).all()]
        selection.add('ak4btagMedium08', ak4_away.btagDeepB.max() > self._btagWPs['med'][self._year])

        selection.add('met', events.MET.pt < 140.)
        goodmuon = (
            (events.Muon.pt > 10)
            & (np.abs(events.Muon.eta) < 2.4)
            & (events.Muon.pfRelIso04_all < 0.25)
            & (events.Muon.looseId).astype(bool)
        )
        nmuons = goodmuon.sum()
        leadingmuon = events.Muon[goodmuon][:, 0:1]
        muon_ak8_pair = leadingmuon.cross(candidatejet, nested=True)

        nelectrons = (
            (events.Electron.pt > 10)
            & (np.abs(events.Electron.eta) < 2.5)
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
            & (np.abs(leadingmuon.eta) < 2.1)
        ).all())
        selection.add('muonDphiAK8', (
            muon_ak8_pair.i0.delta_phi(muon_ak8_pair.i1) > 2*np.pi/3
        ).all().all())

        cutflow = ['jetkin', 'trigger', 'jetid', 'n2ddt', 'antiak4btagMediumOppHem', 'met', 'noleptons']
        allcuts = set()
        output['cutflow']['none'] += len(events)
        for cut in cutflow:
            allcuts.add(cut)
            output['cutflow'][cut] += selection.all(*allcuts).sum()

        weights = processor.Weights(len(events))
        if not isRealData:
            weights.add('genweight', events.genWeight)
            add_pileup_weight(weights, events.Pileup.nPU, self._year, dataset)
            bosons = events.GenPart[
                (np.abs(events.GenPart.pdgId) >= 21)
                & (np.abs(events.GenPart.pdgId) <= 37)
                & events.GenPart.hasFlags(['isHardProcess', 'isLastCopy'])
            ]
            genBosonPt = bosons.pt.pad(1, clip=True).fillna(0)
            add_VJets_NLOkFactor(weights, genBosonPt, self._year, dataset)

            ak8_boson_pair = candidatejet.cross(bosons, nested=True)
            dR2 = ak8_boson_pair.i0.delta_r2(ak8_boson_pair.i1)
            dPt2 = ((ak8_boson_pair.i0.pt - ak8_boson_pair.i1.pt)/(ak8_boson_pair.i0.pt + ak8_boson_pair.i1.pt))**2
            matchedBoson = ak8_boson_pair.i1[(dR2 + dPt2).argmin()].flatten(axis=1)

        return output

    def postprocess(self, accumulator):
        return accumulator
