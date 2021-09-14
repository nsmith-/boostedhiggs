import logging
import numpy as np
import awkward as ak
import json
import copy
from collections import defaultdict
from coffea import processor, hist
import hist as hist2
from coffea.analysis_tools import Weights, PackedSelection
from coffea.lumi_tools import LumiMask
from boostedhiggs.btag import BTagEfficiency, BTagCorrector
from boostedhiggs.common import (
    getBosons,
    bosonFlavor,
    pass_json_array,
)
from boostedhiggs.corrections import (
    corrected_msoftdrop,
    n2ddt_shift,
    powheg_to_nnlops,
    add_ps_weight,
    add_pdf_weight,
    add_pileup_weight,
    add_VJets_NLOkFactor,
    add_VJets_kFactors,
    add_jetTriggerWeight,
    add_jetTriggerSF,
    add_mutriggerSF,
    add_mucorrectionsSF,
    jet_factory,
    fatjet_factory,
    add_jec_variables,
    met_factory,
    lumiMasks,
)


logger = logging.getLogger(__name__)


def update(events, collections):
    """Return a shallow copy of events array with some collections swapped out"""
    out = events
    for name, value in collections.items():
        out = ak.with_field(out, value, name)
    return out


class HbbProcessor(processor.ProcessorABC):
    def __init__(self, year='2017', jet_arbitration='pt', tagger='v2',
                 nnlops_rew=False, skipJER=False, tightMatch=False, newTrigger=False, looseTau=False,
                 newVjetsKfactor=False,
                 ):
        self._year = year
        self._tagger  = tagger
        self._nnlops_rew = nnlops_rew  # for 2018, reweight POWHEG to NNLOPS
        self._jet_arbitration = jet_arbitration
        self._skipJER = skipJER
        self._tightMatch = tightMatch
        self._newVjetsKfactor= newVjetsKfactor
        self._newTrigger = newTrigger  # Fewer triggers, new maps (2017 only, ~no effect)
        self._looseTau = looseTau  # Looser tau veto

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
                'AK8PFJet420_TrimMass30',  # redundant
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
                'AK8PFJet330_TrimMass30_PFAK8BoostedDoubleB_np4',
            ],
        }

        # https://twiki.cern.ch/twiki/bin/view/CMS/MissingETOptionalFiltersRun2
        self._met_filters = {
            '2016': {
                'data': [
                    'goodVertices',
                    'globalSuperTightHalo2016Filter',
                    'HBHENoiseFilter',
                    'HBHENoiseIsoFilter',
                    'EcalDeadCellTriggerPrimitiveFilter',
                    'BadPFMuonFilter',
                    'eeBadScFilter',
                ],
                'mc': [
                    'goodVertices',
                    'globalSuperTightHalo2016Filter',
                    'HBHENoiseFilter',
                    'HBHENoiseIsoFilter',
                    'EcalDeadCellTriggerPrimitiveFilter',
                    'BadPFMuonFilter',
                    # 'eeBadScFilter',
                ],
            },
            '2017': {
                'data': [
                    'goodVertices',
                    'globalSuperTightHalo2016Filter',
                    'HBHENoiseFilter',
                    'HBHENoiseIsoFilter',
                    'EcalDeadCellTriggerPrimitiveFilter',
                    'BadPFMuonFilter',
                    'eeBadScFilter',
                    'ecalBadCalibFilterV2',
                ],
                'mc': [
                    'goodVertices',
                    'globalSuperTightHalo2016Filter',
                    'HBHENoiseFilter',
                    'HBHENoiseIsoFilter',
                    'EcalDeadCellTriggerPrimitiveFilter',
                    'BadPFMuonFilter',
                    'eeBadScFilter',
                    'ecalBadCalibFilterV2',
                ],
            },
            '2018': {
                'data': [
                    'goodVertices',
                    'globalSuperTightHalo2016Filter',
                    'HBHENoiseFilter',
                    'HBHENoiseIsoFilter',
                    'EcalDeadCellTriggerPrimitiveFilter',
                    'BadPFMuonFilter',
                    'eeBadScFilter',
                    'ecalBadCalibFilterV2',
                ],
                'mc': [
                    'goodVertices',
                    'globalSuperTightHalo2016Filter',
                    'HBHENoiseFilter',
                    'HBHENoiseIsoFilter',
                    'EcalDeadCellTriggerPrimitiveFilter',
                    'BadPFMuonFilter',
                    'eeBadScFilter',
                    'ecalBadCalibFilterV2',
                ],
            },
        }

        self._json_paths = {
            '2016': 'jsons/Cert_271036-284044_13TeV_23Sep2016ReReco_Collisions16_JSON.txt',
            '2017': 'jsons/Cert_294927-306462_13TeV_EOY2017ReReco_Collisions17_JSON_v1.txt',
            '2018': 'jsons/Cert_314472-325175_13TeV_17SeptEarlyReReco2018ABC_PromptEraD_Collisions18_JSON.txt',
        }

        if self._tagger == 'v3':
            taggerbins = (
                hist2.axis.Variable([0, 0.7, 0.89, 1], name='ddb', label=r'Jet ddb score', flow=False),
                hist2.axis.Variable([0, 0.44, .84, 1], name='ddc', label=r'Jet ddc score', flow=False),
                hist2.axis.Variable([0, 0.017, 0.11, 1], name='ddcvb', label=r'Jet ddcvb score', flow=False),
            )
        else:
            taggerbins = (
                hist2.axis.Variable([0, 0.7, 0.89, 1], name='ddb', label=r'Jet ddb score', flow=False),
                hist2.axis.Variable([0, 0.34, .45, 0.49, 1], name='ddc', label=r'Jet ddc score', flow=False),
                hist2.axis.Variable([0, 0.03, 0.035, 1], name='ddcvb', label=r'Jet ddcvb score', flow=False),
            )

        optbins = np.r_[np.linspace(0, 0.15, 30, endpoint=False), np.linspace(0.15, 1, 86)]
        self.make_output = lambda: {
            'sumw': 0.,
            'cutflow_msd': hist2.Hist(
                hist2.axis.StrCategory([], name='region', growth=True),
                hist2.axis.IntCategory([0, 1, 2, 3], name='genflavor'),
                hist2.axis.IntCategory([0, 1, 2, 3], name='cut', label='Cut index', growth=True),
                hist2.axis.Regular(23, 40, 201, name='msd', label=r'Jet $m_{sd}$'),
                hist2.storage.Weight(),
            ),
            'cutflow_eta': hist2.Hist(
                hist2.axis.StrCategory([], name='region', growth=True),
                hist2.axis.IntCategory([0, 1, 2, 3], name='genflavor'),
                hist2.axis.IntCategory([0, 1, 2, 3], name='cut', label='Cut index', growth=True),
                hist2.axis.Regular(40, -2.5, 2.5, name='eta', label=r'Jet $\eta$'),
                hist2.storage.Weight(),
            ),
            'cutflow_pt': hist2.Hist(
                hist2.axis.StrCategory([], name='region', growth=True),
                hist2.axis.IntCategory([0, 1, 2, 3], name='genflavor'),
                hist2.axis.IntCategory([0, 1, 2, 3], name='cut', label='Cut index', growth=True),
                hist2.axis.Regular(100, 400, 1200, name='pt', label=r'Jet $p_{T}$ [GeV]'),
                hist2.storage.Weight(),
            ),
            'nminus1_n2ddt': hist2.Hist(
                hist2.axis.StrCategory([], name='region', growth=True),
                hist2.axis.Regular(40, -0.25, 0.25, name='n2ddt', label='N2ddt value'),
                hist2.storage.Weight(),
            ),
            'btagWeight': hist2.Hist(
                hist2.axis.Regular(50, 0, 3, name='val', label='BTag correction'),
                hist2.storage.Weight(),
            ),
            'templates': hist2.Hist(
                hist2.axis.StrCategory([], name='region', growth=True),
                hist2.axis.StrCategory([], name='systematic', growth=True),
                hist2.axis.IntCategory([0, 1, 2, 3], name='genflavor'),
                hist2.axis.Variable([450, 500, 550, 600, 675, 800, 1200], name='pt', label=r'Jet $p_{T}$ [GeV]'),
                hist2.axis.Regular(23, 40, 201, name='msd', label=r'Jet $m_{sd}$'),
                *taggerbins,
                hist2.storage.Weight(),
            ),
            'wtag': hist2.Hist(
                hist2.axis.StrCategory([], name='region', growth=True),
                hist2.axis.StrCategory([], name='systematic', growth=True),
                hist2.axis.IntCategory([0, 1, 2, 3], name='genflavor'),
                hist2.axis.Variable([-1, 0, 1], name='n2ddt', label=r'N2ddt value'),
                # hist2.axis.Variable([200, 450, 1200], name='pt', label=r'Jet $p_{T}$ [GeV]'),
                hist2.axis.Regular(46, 40, 201, name='msd', label=r'Jet $m_{sd}$'),
                *taggerbins[1:],
                hist2.storage.Weight(),
            ),
            'signal_opt': hist2.Hist(
                hist2.axis.StrCategory([], name='region', growth=True),
                hist2.axis.IntCategory([0, 1, 2, 3], name='genflavor'),
                hist2.axis.Variable(optbins, name='ddc', label=r'Jet CvL score'),
                hist2.axis.Variable(optbins, name='ddcvb', label=r'Jet CvB score'),
                hist2.storage.Weight(),
            ),
            'signal_optb': hist2.Hist(
                hist2.axis.StrCategory([], name='region', growth=True),
                hist2.axis.IntCategory([0, 1, 2, 3], name='genflavor'),
                hist2.axis.Variable(optbins, name='ddb', label=r'Jet BvL score'),
                hist2.storage.Weight(),
            ),
            'genresponse_noweight': hist2.Hist(
                hist2.axis.StrCategory([], name='region', growth=True),
                hist2.axis.StrCategory([], name='systematic', growth=True),
                hist2.axis.Variable([450, 500, 550, 600, 675, 800, 1200], name='pt', label=r'Jet $p_{T}$ [GeV]'),
                hist2.axis.Variable(np.geomspace(400, 1200, 60), name='genpt', label=r'Generated Higgs $p_{T}$ [GeV]'),
                hist2.storage.Double(),
            ),
            'genresponse': hist2.Hist(
                hist2.axis.StrCategory([], name='region', growth=True),
                hist2.axis.StrCategory([], name='systematic', growth=True),
                hist2.axis.Variable([450, 500, 550, 600, 675, 800, 1200], name='pt', label=r'Jet $p_{T}$ [GeV]'),
                hist2.axis.Variable([200, 300, 450, 650, 7500], name='genpt', label=r'Generated Higgs $p_{T}$ [GeV]'),
                hist2.storage.Weight(),
            ),
        }

    def process(self, events):
        print("HMM")
        isRealData = not hasattr(events, "genWeight")

        if isRealData:
            # Nominal JEC are already applied in data
            return self.process_shift(events, None)

        import cachetools
        jec_cache = cachetools.Cache(np.inf)
        nojer = "NOJER" if self._skipJER else ""
        fatjets = fatjet_factory[f"{self._year}mc{nojer}"].build(add_jec_variables(events.FatJet, events.fixedGridRhoFastjetAll), jec_cache)
        jets = jet_factory[f"{self._year}mc{nojer}"].build(add_jec_variables(events.Jet, events.fixedGridRhoFastjetAll), jec_cache)
        met = met_factory.build(events.MET, jets, {})

        shifts = [
            ({"Jet": jets, "FatJet": fatjets, "MET": met}, None),
            ({"Jet": jets.JES_jes.up, "FatJet": fatjets.JES_jes.up, "MET": met.JES_jes.up}, "JESUp"),
            ({"Jet": jets.JES_jes.down, "FatJet": fatjets.JES_jes.down, "MET": met.JES_jes.down}, "JESDown"),
            ({"Jet": jets, "FatJet": fatjets, "MET": met.MET_UnclusteredEnergy.up}, "UESUp"),
            ({"Jet": jets, "FatJet": fatjets, "MET": met.MET_UnclusteredEnergy.down}, "UESDown"),
        ]
        if not self._skipJER:
            shifts.extend([
                ({"Jet": jets.JER.up, "FatJet": fatjets.JER.up, "MET": met.JER.up}, "JERUp"),
                ({"Jet": jets.JER.down, "FatJet": fatjets.JER.down, "MET": met.JER.down}, "JERDown"),
            ])
        # HEM15/16 issue
        if self._year == "2018":
            _runid = (events.run >= 319077)
            j_mask = ak.where((jets.phi > -1.57) & (jets.phi < -0.87) &
                              (jets.eta > -2.5) & (jets.eta < 1.3) & _runid, 0.8, 1)
            fj_mask = ak.where((fatjets.phi > -1.57) & (fatjets.phi < -0.87) &
                               (fatjets.eta > -2.5) & (fatjets.eta < 1.3) & _runid, 
                               0.8, 1)
            shift_jets = copy.deepcopy(jets)
            shift_fatjets = copy.deepcopy(fatjets)
            for collection, mask in zip([shift_jets, shift_fatjets], [j_mask, fj_mask]):
                collection["pt"] = mask * collection.pt
                collection["mass"] = mask * collection.mass
            shifts.extend([
                ({"Jet": shift_jets, "FatJet": shift_fatjets, "MET": met}, "HEM18"),
            ])

        return processor.accumulate(self.process_shift(update(events, collections), name) for collections, name in shifts)

    def process_shift(self, events, shift_name):
        dataset = events.metadata['dataset']
        isRealData = not hasattr(events, "genWeight")
        selection = PackedSelection()
        weights = Weights(len(events), storeIndividual=True)
        output = self.make_output()
        if shift_name is None and not isRealData:
            output['sumw'] = ak.sum(events.genWeight)

        if isRealData or self._newTrigger:
            trigger = np.zeros(len(events), dtype='bool')
            for t in self._triggers[self._year]:
                if t in events.HLT.fields:
                    trigger = trigger | events.HLT[t]
            selection.add('trigger', trigger)
            del trigger
        else:
            selection.add('trigger', np.ones(len(events), dtype='bool'))

        if isRealData:
            selection.add('lumimask', lumiMasks[self._year](events.run, events.luminosityBlock))
        else:
            selection.add('lumimask', np.ones(len(events), dtype='bool'))

        if isRealData:
            trigger = np.zeros(len(events), dtype='bool')
            for t in self._muontriggers[self._year]:
                if t in events.HLT.fields:
                    trigger |= np.array(events.HLT[t])
            selection.add('muontrigger', trigger)
            del trigger
        else:
            selection.add('muontrigger', np.ones(len(events), dtype='bool'))

        metfilter = np.ones(len(events), dtype='bool')
        for flag in self._met_filters[self._year]['data' if isRealData else 'mc']:
            metfilter &= np.array(events.Flag[flag])
        selection.add('metfilter', metfilter)
        del metfilter

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

        candidatejet = candidatejet[:, :2]  # Only consider first two to match generators
        if self._jet_arbitration == 'pt':
            candidatejet = ak.firsts(candidatejet)
        elif self._jet_arbitration == 'mass':
            candidatejet = ak.firsts(candidatejet[ak.argmax(candidatejet.msdcorr, axis=1, keepdims=True)])
        elif self._jet_arbitration == 'n2':
            candidatejet = ak.firsts(candidatejet[ak.argmin(candidatejet.n2ddt, axis=1, keepdims=True)])
        elif self._jet_arbitration == 'ddb':
            candidatejet = ak.firsts(candidatejet[ak.argmax(candidatejet.btagDDBvLV2, axis=1, keepdims=True)])
        elif self._jet_arbitration == 'ddc':
            candidatejet = ak.firsts(candidatejet[ak.argmax(candidatejet.btagDDCvLV2, axis=1, keepdims=True)])
        else:
            raise RuntimeError("Unknown candidate jet arbitration")

        if self._tagger == 'v1':
            bvl = candidatejet.btagDDBvL
            cvl = candidatejet.btagDDCvL
            cvb = candidatejet.btagDDCvB
        elif self._tagger == 'v2':
            bvl = candidatejet.btagDDBvLV2
            cvl = candidatejet.btagDDCvLV2
            cvb = candidatejet.btagDDCvBV2
        elif self._tagger == 'v3':
            bvl = candidatejet.particleNetMD_Xbb
            cvl = candidatejet.particleNetMD_Xcc / (1 - candidatejet.particleNetMD_Xbb)
            cvb = candidatejet.particleNetMD_Xcc / (candidatejet.particleNetMD_Xcc + candidatejet.particleNetMD_Xbb)

        elif self._tagger == 'v4':
            bvl = candidatejet.particleNetMD_Xbb
            cvl = candidatejet.btagDDCvLV2
            cvb = candidatejet.particleNetMD_Xcc / (candidatejet.particleNetMD_Xcc + candidatejet.particleNetMD_Xbb)
        else:
            raise ValueError("Not an option")

        selection.add('minjetkin',
            (candidatejet.pt >= 450)
            & (candidatejet.pt < 1200)
            & (candidatejet.msdcorr >= 40.)
            & (candidatejet.msdcorr < 201.)
            & (abs(candidatejet.eta) < 2.5)
        )
        selection.add('minjetkinmu',
            (candidatejet.pt >= 400)
            & (candidatejet.pt < 1200)
            & (candidatejet.msdcorr >= 40.)
            & (candidatejet.msdcorr < 201.)
            & (abs(candidatejet.eta) < 2.5)
        )
        selection.add('jetid', candidatejet.isTight)
        selection.add('n2ddt', (candidatejet.n2ddt < 0.))
        if not self._tagger == 'v2':
            selection.add('ddbpass', (bvl >= 0.89))
            selection.add('ddcpass', (cvl >= 0.83))
            selection.add('ddcvbpass', (cvb >= 0.2))
        else:
            selection.add('ddbpass', (bvl >= 0.7))
            selection.add('ddcpass', (cvl >= 0.45))
            selection.add('ddcvbpass', (cvb >= 0.03))

        jets = events.Jet
        jets = jets[
            (jets.pt > 30.)
            & (abs(jets.eta) < 2.5)
            & jets.isTight
        ]
        # only consider first 4 jets to be consistent with old framework
        jets = jets[:, :4]
        dphi = abs(jets.delta_phi(candidatejet))
        selection.add('antiak4btagMediumOppHem', ak.max(jets[dphi > np.pi / 2].btagDeepB, axis=1, mask_identity=False) < BTagEfficiency.btagWPs[self._year]['medium'])
        ak4_away = jets[dphi > 0.8]
        selection.add('ak4btagMedium08', ak.max(ak4_away.btagDeepB, axis=1, mask_identity=False) > BTagEfficiency.btagWPs[self._year]['medium'])

        met = events.MET
        selection.add('met', met.pt < 140.)

        goodmuon = (
            (events.Muon.pt > 10)
            & (abs(events.Muon.eta) < 2.4)
            & (events.Muon.pfRelIso04_all < 0.25)
            & events.Muon.looseId
        )
        nmuons = ak.sum(goodmuon, axis=1)
        leadingmuon = ak.firsts(events.Muon[goodmuon])

        if self._looseTau:
            goodelectron = (
                (events.Electron.pt > 10)
                & (abs(events.Electron.eta) < 2.5)
                & (events.Electron.cutBased >= events.Electron.VETO)
            )
            nelectrons = ak.sum(goodelectron, axis=1)

            ntaus = ak.sum(
                (
                    (events.Tau.pt > 20)
                    & (abs(events.Tau.eta) < 2.3)
                    & events.Tau.idDecayMode
                    & ((events.Tau.idMVAoldDM2017v2 & 2) != 0)
                    & ak.all(events.Tau.metric_table(events.Muon[goodmuon]) > 0.4, axis=2)
                    & ak.all(events.Tau.metric_table(events.Electron[goodelectron]) > 0.4, axis=2)
                ),
                axis=1,
            )
        else:
            goodelectron = (
                (events.Electron.pt > 10)
                & (abs(events.Electron.eta) < 2.5)
                & (events.Electron.cutBased >= events.Electron.LOOSE)
            )
            nelectrons = ak.sum(goodelectron, axis=1)

            ntaus = ak.sum(
                (events.Tau.pt > 20)
                & events.Tau.idDecayMode  # bacon iso looser than Nano selection
                & ak.all(events.Tau.metric_table(events.Muon[goodmuon]) > 0.4, axis=2)
                & ak.all(events.Tau.metric_table(events.Electron[goodelectron]) > 0.4, axis=2),
                axis=1,
            )

        selection.add('noleptons', (nmuons == 0) & (nelectrons == 0) & (ntaus == 0))
        selection.add('onemuon', (nmuons == 1) & (nelectrons == 0) & (ntaus == 0))
        selection.add('muonkin', (leadingmuon.pt > 55.) & (abs(leadingmuon.eta) < 2.1))
        selection.add('muonDphiAK8', abs(leadingmuon.delta_phi(candidatejet)) > 2*np.pi/3)

        # W-Tag (Tag and Probe)
        # tag side
        selection.add('ak4btagMediumOppHem', ak.max(jets[dphi > np.pi / 2].btagDeepB, axis=1, mask_identity=False) > BTagEfficiency.btagWPs[self._year]['medium'])
        selection.add('met40p', met.pt > 40.)
        selection.add('tightMuon', (leadingmuon.tightId) & (leadingmuon.pt > 53.))
        selection.add('ptrecoW', (leadingmuon + met).pt > 250.)
        selection.add('ptrecoW200', (leadingmuon + met).pt > 200.)
        selection.add('ak4btagNearMu', leadingmuon.delta_r(leadingmuon.nearest(ak4_away, axis=None)) < 2.0 )
        _bjets = jets.btagDeepB > BTagEfficiency.btagWPs[self._year]['medium']
        _nearAK8 = jets.delta_r(candidatejet)  < 0.8
        _nearMu = jets.delta_r(ak.firsts(events.Muon))  < 0.3
        selection.add('ak4btagOld', ak.sum(_bjets & ~_nearAK8 & ~_nearMu, axis=1) >= 1)
        # probe side
        selection.add('minWjetpteta', (candidatejet.pt >= 200) & (abs(candidatejet.eta) < 2.4))
        selection.add('noNearMuon', candidatejet.delta_r(candidatejet.nearest(events.Muon[goodmuon], axis=None)) > 1.0)
        #####

        if isRealData :
            genflavor = ak.zeros_like(candidatejet.pt)
        else:
            weights.add('genweight', events.genWeight)
            if "PSWeight" in events.fields:
                add_ps_weight(weights, events.PSWeight)
            else:
                add_ps_weight(weights, None)
            if "LHEPdfWeight" in events.fields:
                add_pdf_weight(weights, events.LHEPdfWeight)
            else:
                add_pdf_weight(weights, None)
            add_pileup_weight(weights, events.Pileup.nPU, self._year, dataset)
            bosons = getBosons(events.GenPart)
            matchedBoson = candidatejet.nearest(bosons, axis=None, threshold=0.8)
            if self._tightMatch:
                match_mask = ((candidatejet.pt - matchedBoson.pt)/matchedBoson.pt < 0.5) & ((candidatejet.msdcorr - matchedBoson.mass)/matchedBoson.mass < 0.3)
                selmatchedBoson = ak.mask(matchedBoson, match_mask)
                genflavor = bosonFlavor(selmatchedBoson)
            else:
                genflavor = bosonFlavor(matchedBoson)
            genBosonPt = ak.fill_none(ak.firsts(bosons.pt), 0)
            if self._newVjetsKfactor:
                add_VJets_kFactors(weights, events.GenPart, dataset)
            else:
                add_VJets_NLOkFactor(weights, genBosonPt, self._year, dataset)
            if shift_name is None:
                output['btagWeight'].fill(val=self._btagSF.addBtagWeight(weights, ak4_away))
            if self._nnlops_rew and dataset in ['GluGluHToCC_M125_13TeV_powheg_pythia8']:
                weights.add('minlo_rew', powheg_to_nnlops(ak.to_numpy(genBosonPt)))

            if self._newTrigger:
                add_jetTriggerSF(weights, ak.firsts(fatjets), self._year, selection)
            else:
                add_jetTriggerWeight(weights, candidatejet.msdcorr, candidatejet.pt, self._year)

            add_mutriggerSF(weights, leadingmuon, self._year, selection)
            add_mucorrectionsSF(weights, leadingmuon, self._year, selection)

            if self._year in ("2016", "2017"):
                weights.add("L1Prefiring", events.L1PreFiringWeight.Nom, events.L1PreFiringWeight.Up, events.L1PreFiringWeight.Dn)

            logger.debug("Weight statistics: %r" % weights.weightStatistics)

        msd_matched = candidatejet.msdcorr * self._msdSF[self._year] * (genflavor > 0) + candidatejet.msdcorr * (genflavor == 0)

        regions = {
            'signal': ['noleptons', 'minjetkin', 'met', 'jetid', 'antiak4btagMediumOppHem', 'n2ddt', 'trigger', 'lumimask', 'metfilter'],
            'signal_noddt': ['noleptons', 'minjetkin', 'met', 'jetid', 'antiak4btagMediumOppHem', 'trigger', 'lumimask', 'metfilter'],
            'muoncontrol': ['minjetkinmu', 'jetid', 'n2ddt', 'ak4btagMedium08', 'onemuon', 'muonkin', 'muonDphiAK8', 'muontrigger', 'lumimask', 'metfilter'],
            'muoncontrol_noddt': ['minjetkinmu', 'jetid', 'ak4btagMedium08', 'onemuon', 'muonkin', 'muonDphiAK8', 'muontrigger', 'lumimask', 'metfilter'],
            'wtag': ['tightMuon', 'onemuon', 'noNearMuon', 'ak4btagNearMu', 'met40p', 'ak4btagMediumOppHem',
                     'minWjetpteta', 'ptrecoW', 'muontrigger', 'lumimask', 'metfilter'],
            'wtag2': ['tightMuon', 'onemuon', 'met40p', 'ptrecoW200' , 'ak4btagOld', 'muontrigger', 'lumimask', 'metfilter'],
            'noselection': [],
        }

        def normalize(val, cut):
            if cut is None:
                ar = ak.to_numpy(ak.fill_none(val, np.nan))
                return ar
            else:
                ar = ak.to_numpy(ak.fill_none(val[cut], np.nan))
                return ar

        import time
        tic = time.time()
        if shift_name is None:
            for region, cuts in regions.items():
                allcuts = set([])
                cut = selection.all(*allcuts)
                output['cutflow_msd'].fill(region=region,
                                           genflavor=normalize(genflavor, None),
                                           cut=0,
                                           weight=weights.weight(),
                                           msd=normalize(msd_matched, None))
                output['cutflow_eta'].fill(region=region,
                                           genflavor=normalize(genflavor, cut),
                                           cut=0,
                                           weight=weights.weight()[cut],
                                           eta=normalize(candidatejet.eta, cut))
                output['cutflow_pt'].fill(region=region,
                                          genflavor=normalize(genflavor, cut),
                                          cut=0,
                                          weight=weights.weight()[cut],
                                          pt=normalize(candidatejet.pt, cut))
                for i, cut in enumerate(cuts + ['ddcvbpass', 'ddcpass']):
                    allcuts.add(cut)
                    cut = selection.all(*allcuts)
                    output['cutflow_msd'].fill(region=region,
                                               genflavor=normalize(genflavor, cut),
                                               cut=i + 1,
                                               weight=weights.weight()[cut],
                                               msd=normalize(msd_matched, cut))
                    output['cutflow_eta'].fill(region=region,
                                               genflavor=normalize(genflavor, cut),
                                               cut=i + 1,
                                               weight=weights.weight()[cut],
                                               eta=normalize(candidatejet.eta, cut))
                    output['cutflow_pt'].fill(region=region,
                                              genflavor=normalize(genflavor, cut),
                                              cut=i + 1,
                                              weight=weights.weight()[cut],
                                              pt=normalize(candidatejet.pt, cut))


        if shift_name is None:
            systematics = [None] + list(weights.variations)
        else:
            systematics = [shift_name]

        def fill(region, systematic, wmod=None):
            selections = regions[region]
            cut = selection.all(*selections)
            sname = 'nominal' if systematic is None else systematic
            if wmod is None:
                if systematic in weights.variations:
                    weight = weights.weight(modifier=systematic)[cut]
                else:
                    weight = weights.weight()[cut]
            else:
                weight = weights.weight()[cut] * wmod[cut]

            output['templates'].fill(
                region=region,
                systematic=sname,
                genflavor=normalize(genflavor, cut),
                pt=normalize(candidatejet.pt, cut),
                msd=normalize(msd_matched, cut),
                ddb=normalize(bvl, cut),
                ddc=normalize(cvl, cut),
                ddcvb=normalize(cvb, cut),
                weight=weight,
            )
            if region in ['wtag', 'wtag2', 'noselection']:# and sname in ['nominal', 'pileup_weightDown', 'pileup_weightUp', 'jet_triggerDown', 'jet_triggerUp']:
                output['wtag'].fill(
                    region=region,
                    systematic=sname,
                    genflavor=normalize(genflavor, cut),
                    # pt=normalize(candidatejet.pt, cut),
                    msd=normalize(msd_matched, cut),
                    n2ddt=normalize(candidatejet.n2ddt, cut),
                    ddc=normalize(cvl, cut),
                    ddcvb=normalize(cvb, cut),
                    weight=weight,
                )
            if not isRealData:
                if wmod is not None:
                    _custom_weight = events.genWeight[cut] * wmod[cut]
                else:
                    _custom_weight = np.ones_like(weight)
                output['genresponse_noweight'].fill(
                    region=region,
                    systematic=sname,
                    pt=normalize(candidatejet.pt, cut),
                    genpt=normalize(genBosonPt, cut),
                    weight=_custom_weight,
                )

                output['genresponse'].fill(
                    region=region,
                    systematic=sname,
                    pt=normalize(candidatejet.pt, cut),
                    genpt=normalize(genBosonPt, cut),
                    weight=weight,
                )
            if systematic is None:
                output['signal_opt'].fill(
                    region=region,
                    genflavor=normalize(genflavor, cut),
                    ddc=normalize(cvl, cut),
                    ddcvb=normalize(cvb, cut),
                    weight=weight,
                )
                output['signal_optb'].fill(
                    region=region,
                    genflavor=normalize(genflavor, cut),
                    ddb=normalize(bvl, cut),
                    weight=weight,
                )

        for region in regions:
            cut = selection.all(*(set(regions[region]) - {'n2ddt'}))
            if shift_name is None:
                output['nminus1_n2ddt'].fill(
                    region=region,
                    n2ddt=normalize(candidatejet.n2ddt, cut),
                    weight=weights.weight()[cut],
                )
            for systematic in systematics:
                if isRealData and systematic is not None:
                    continue
                fill(region, systematic)
            if shift_name is None and 'GluGluH' in dataset and 'LHEWeight' in events.fields:
                for i in range(9):
                    fill(region, 'LHEScale_%d' % i, events.LHEScaleWeight[:, i])
                for c in events.LHEWeight.fields[1:]:
                    fill(region, 'LHEWeight_%s' % c, events.LHEWeight[c])

        toc = time.time()
        output["filltime"] = toc - tic
        if shift_name is None:
            output["weightStats"] = weights.weightStatistics
        return {dataset: output}

    def postprocess(self, accumulator):
        return accumulator
