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
                "HLT_PFHT800",
                "HLT_PFHT900",
                "HLT_AK8PFJet360_TrimMass30",
                'HLT_AK8PFHT700_TrimR0p1PT0p03Mass50',
                "HLT_PFHT650_WideJetMJJ950DEtaJJ1p5",
                "HLT_PFHT650_WideJetMJJ900DEtaJJ1p5",
                "HLT_AK8DiPFJet280_200_TrimMass30_BTagCSV_p20",
                "HLT_PFJet450",
            ],
            '2017': [
                "HLT_AK8PFJet330_PFAK8BTagCSV_p17",
                "HLT_PFHT1050",
                "HLT_AK8PFJet400_TrimMass30",
                "HLT_AK8PFJet420_TrimMass30",
                "HLT_AK8PFHT800_TrimMass50",
                "HLT_PFJet500",
                "HLT_AK8PFJet500",
            ],
            '2018': [
                'HLT_AK8PFJet400_TrimMass30',
                'HLT_AK8PFJet420_TrimMass30',
                'HLT_AK8PFHT800_TrimMass50',
                'HLT_PFHT1050',
                'HLT_PFJet500',
                'HLT_AK8PFJet500',
                'HLT_AK8PFJet330_PFAK8BTagCSV_p17',
                "HLT_AK8PFJet330_TrimMass30_PFAK8BoostedDoubleB_np4",
            ],
        }

        self._accumulator = processor.dict_accumulator({
            'cutflow': processor.defaultdict_accumulator(float),

        })

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, df):
        dataset = df['dataset']
        isRealData = 'genWeight' not in df
        events = buildevents(df)
        output = self.accumulator.identity()
        selection = processor.PackedSelection()

        trigger = np.ones(df.size, dtype='bool')
        for t in self._triggers[self._year]:
            trigger &= df[t]
        selection.add('trigger', trigger)

        fatjets = events.fatjets
        fatjets['msdcorr'] = corrected_msoftdrop(fatjets)
        fatjets['rho'] = 2*np.log(fatjets.msdcorr / fatjets.p4.pt)
        fatjets['n2ddt'] = fatjets.n2 - n2ddt_shift(fatjets, year=self._year)

        candidatejet = fatjets[:, 0:1]
        selection.add('jetkin', (
            (candidatejet.p4.pt > 450)
            & (candidatejet.p4.eta < 2.4)
            & (candidatejet.msdcorr > 40.)
        ).any())
        selection.add('jetid', (candidatejet.jetId & 2).any())  # tight id
        selection.add('n2ddt', (candidatejet.n2ddt < 0.).any())

        # only consider first 4 jets to be consistent with old framework
        jets = events.jets[
            (events.jets.p4.pt > 30.)
            & (events.jets.localindex < 4)
            & (events.jets.jetId & 2)  # tight id
        ]
        ak4_ak8_pair = jets.cross(candidatejet, nested=True)
        dphi = ak4_ak8_pair.i0.p4.delta_phi(ak4_ak8_pair.i1.p4)
        ak4_opposite = jets[(np.abs(dphi) > np.pi / 2).all()]
        selection.add('antiak4btagMediumOppHem', ak4_opposite.deepcsvb.max() < self._btagWPs['med'][self._year])
        ak4_away = jets[(np.abs(dphi) > 0.8).all()]
        selection.add('ak4btagMedium08', ak4_away.deepcsvb.max() > self._btagWPs['med'][self._year])

        selection.add('met', events.met.rho < 140.)
        goodmuon = (
            (events.muons.p4.pt > 10)
            & (np.abs(events.muons.p4.eta) < 2.4)
            & (events.muons.pfRelIso04_all < 0.25)
            & (events.muons.looseId).astype(bool)
        )
        nmuons = goodmuon.sum()
        leadingmuon = events.muons[goodmuon][:, 0:1]
        muon_ak8_pair = leadingmuon.cross(candidatejet, nested=True)

        nelectrons = (
            (events.electrons.p4.pt > 10)
            & (np.abs(events.electrons.p4.eta) < 2.5)
            & (events.electrons.cutBased & (1 << 2)).astype(bool)  # 2017V2 loose
        ).sum()

        ntaus = (
            (events.taus.p4.pt > 20)
            & (events.taus.idDecayMode).astype(bool)
            # bacon iso looser than Nano selection
        ).sum()

        selection.add('noleptons', (nmuons == 0) & (nelectrons == 0) & (ntaus == 0))
        selection.add('onemuon', (nmuons == 1) & (nelectrons == 0) & (ntaus == 0))
        selection.add('muonkin', (
            (leadingmuon.p4.pt > 55.)
            & (np.abs(leadingmuon.p4.eta) < 2.1)
        ).all())
        selection.add('muonDphiAK8', (
            muon_ak8_pair.i0.p4.delta_phi(muon_ak8_pair.i1.p4) > 2*np.pi/3
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
            add_pileup_weight(weights, events.Pileup_nPU, self._year, dataset)
            bosons = events.genpart[
                (np.abs(events.genpart.pdgId) >= 21)
                & (np.abs(events.genpart.pdgId) <= 37)
                & (events.genpart.statusFlags & ((1 << 7) | (1 << 13))).astype(bool)  # isHardProcess, isLastCopy
                & (events.genpart.genPartIdxMother >= 0)
            ]
            genBosonPt = bosons.p4.pt.pad(1, clip=True).fillna(0)
            add_VJets_NLOkFactor(weights, genBosonPt, self._year, dataset)

            ak8_boson_pair = candidatejet.cross(bosons, nested=True)
            dR2 = ak8_boson_pair.i0.p4.delta_r2(ak8_boson_pair.i1.p4)
            dPt2 = ((ak8_boson_pair.i0.p4.pt - ak8_boson_pair.i1.p4.pt)/(ak8_boson_pair.i0.p4.pt + ak8_boson_pair.i1.p4.pt))**2
            matchedBoson = ak8_boson_pair.i1[(dR2 + dPt2).argmin()].flatten(axis=1)

        return output

    def postprocess(self, accumulator):
        return accumulator
