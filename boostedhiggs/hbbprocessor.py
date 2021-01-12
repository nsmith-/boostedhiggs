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
    def __init__(self, year='2017', jet_arbitration='pt', v2=False, v3=False, v4=False):
        # v2 DDXv2
        # v3 ParticleNet
        # v4 mix
        self._year = year
        self._v2 = v2
        self._v3 = v3
        self._v4 = v4
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
            'cutflow_msd': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('genflavor', 'Gen. jet flavor', [0, 1, 2, 3, 4]),
                hist.Bin('cut', 'Cut index', 11, 0, 11),
                hist.Bin('msd', r'Jet $m_{sd}$', 23, 40, 201),
            ),
            'cutflow_eta': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('genflavor', 'Gen. jet flavor', [0, 1, 2, 3, 4]),
                hist.Bin('cut', 'Cut index', 11, 0, 11),
                hist.Bin('eta', r'Jet $\eta$', 40, -2.5, 2.5),
            ),
            'cutflow_pt': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('genflavor', 'Gen. jet flavor', [0, 1, 2, 3, 4]),
                hist.Bin('cut', 'Cut index', 11, 0, 11),
                hist.Bin('pt', r'Jet $p_{T}$ [GeV]', 100, 400, 1200),
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
                hist.Bin('ddc', r'Jet ddc score', [0, 0.1, 0.44, .83, 1]),
                hist.Bin('ddcvb', r'Jet ddcvb score', [0, 0.017, 0.2, 1]),
            ),
            'signal_opt': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('genflavor', 'Gen. jet flavor', [0, 1, 2, 3, 4]),
                hist.Bin('ddc', r'Jet ddc score', np.r_[0, np.geomspace(0.0001, 1, 101)]) if self._v2 else hist.Bin('ddc', r'Jet ddc score', 100, 0, 1), 
                hist.Bin('ddcvb', r'Jet ddcvb score',np.r_[0, np.geomspace(0.0001, 1, 101)]) if self._v2 else hist.Bin('ddcvb', r'Jet ddc score', 100, 0, 1), 
            ),
            'signal_optb': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('genflavor', 'Gen. jet flavor', [0, 1, 2, 3, 4]),
                hist.Bin('ddb', r'Jet ddb score', np.r_[0, np.geomspace(0.0001, 1, 101)]) if self._v2 else hist.Bin('ddb', r'Jet ddc score', 100, 0, 1), 
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
                # FIXME
                if t in events.HLT.fields:
                    trigger = trigger | events.HLT[t]
        else:
            trigger = np.ones(len(events), dtype='bool')
        selection.add('trigger', trigger)

        if isRealData:
            trigger = np.zeros(len(events), dtype='bool')
            for t in self._muontriggers[self._year]:
                # FIXME
                if t in events.HLT.fields:
                    trigger = trigger | events.HLT[t]
        else:
            trigger = np.ones(len(events), dtype='bool')
        selection.add('muontrigger', trigger)

        fatjets = events.FatJet
        
        # FIXME
        # fatjets['msdcorr'] = corrected_msoftdrop(fatjets)
        fatjets['msdcorr'] = corrected_msoftdrop(fatjets, events.SubJet)
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
            (candidatejet.msdcorr >= 40.)
            & (candidatejet.pt < 1200)
            & (candidatejet.msdcorr < 201.)
        )
        selection.add('jetkinematics',
            (candidatejet.pt > 450)
        )
        selection.add('jetid', candidatejet.isTight)
        selection.add('n2ddt', (candidatejet.n2ddt < 0.))
        if not self._v2:
            selection.add('ddbpass', (candidatejet.btagDDBvL >= 0.89))
            selection.add('ddcpass', (candidatejet.btagDDCvL >= 0.83))
            selection.add('ddcvbpass', (candidatejet.btagDDCvB >= 0.2))
        else:
            selection.add('ddbpass', (candidatejet.btagDDBvLV2 >= 0.89))
            selection.add('ddcpass', (candidatejet.btagDDCvLV2 >= 0.83))
            selection.add('ddcvbpass', (candidatejet.btagDDCvBV2 >= 0.2))

        jets = events.Jet[
            (events.Jet.pt > 30.)
            & (abs(events.Jet.eta) < 2.5)
            & events.Jet.isTight
        ]
        # Protect again "empty" arrays [None, None, None...]
        # if ak.sum(candidatejet.phi) == 0.:
        #     return self.accumulator.identity()
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

        if isRealData :
            genflavor = candidatejet.pt - candidatejet.pt  # zeros_like
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
        print(dataset)
        # print(msd_matched)
        # print(msd_matched[~ak.is_none(msd_matched)])

        regions = {
            # 'signal': ['trigger', 'minjetkin', 'jetacceptance', 'jetid', 'n2ddt', 'antiak4btagMediumOppHem', 'met', 'noleptons'],
            #'signal': ['trigger', 'noleptons', 'jetkinematics',  'met', 'jetid', 'antiak4btagMediumOppHem', 'n2ddt'],
            'signal': ['noleptons', 'minjetkin',  'met', 'jetid', 'antiak4btagMediumOppHem', 'n2ddt', 'trigger'],
            #'signal': ['noleptons', 'jetkinematics',  'met',  'n2ddt', 'jetid', 'antiak4btagMediumOppHem'],
            #oldcuts = ["", 'trigger', 'noLeptons', 'jetKinematics', 'pfmet', 'n2ddtPass', 'tightVjet', 'antiak4btagMediumOppHem']
            'signal_noddt': ['noleptons', 'minjetkin',  'met', 'jetid', 'antiak4btagMediumOppHem', 'trigger'],
            'muoncontrol': ['muontrigger', 'minjetkin', 'jetacceptance', 'jetid', 'n2ddt', 'ak4btagMedium08', 'onemuon', 'muonkin', 'muonDphiAK8'],
            'muoncontrolCC': ['muontrigger', 'minjetkin', 'jetacceptance', 'jetid', 'n2ddt', 'ak4btagMedium08', 'onemuon', 'muonkin', 'muonDphiAK8', 'ddcvbpass'],
            'noselection': [],
        }
        idar = np.array([446736, 446740, 446743, 446748, 446759, 446762, 446770, 446780,
            446786, 446787, 446819, 446835, 446837, 446844, 446868, 446869,
            446885, 446896, 446898, 446920, 446924, 446956, 446962, 446963,
            446966, 446977, 446993, 447018, 447025, 447036, 447065, 447066,
            447069, 447079, 447091, 447094, 447098, 447111, 447113, 447117,
            447121, 447131, 447132, 447150, 447152, 447153, 447166, 447178,
            447190, 447279, 447280, 447319, 447339, 447347, 447361, 447367,
            447375, 447389, 447395, 447410, 447525, 447541, 447551, 447552,
            447555, 447595, 447603, 447613, 447616, 447620, 447677, 447680,
            447732, 447734, 447737, 447739, 447747, 447765, 447772, 447778,
            447788, 447793, 447844, 447855, 447863, 447866, 447872, 447897,
            447918, 447929, 447935, 447965, 447966, 447972, 447993, 448007,
            448014, 448015, 448029, 448030, 448058, 448065, 448069, 448070,
            448076, 448088, 448098, 448119, 448126, 448132, 448143, 448145,
            448149, 448177, 448178, 448179, 448194, 448235, 448244, 448252,
            448268, 448269, 448275, 448283, 448296, 448297, 448326, 448330,
            448338, 448352, 448356, 448374, 448385, 448395, 448409, 448427,
            448439, 448467, 448468, 448505, 448512, 448514, 448529, 448536,
            448546, 448549, 448568, 448573, 448604, 448609, 448616, 448620,
            448651, 448663, 448668, 448676, 448677, 448678, 448691, 448692,
            448703, 448737, 448746, 448753, 448754, 448763, 448780, 448782,
            448814, 448825, 448846, 448869, 448906, 448924, 448928, 448931,
            448946, 448947, 448951, 448953, 448969, 448973, 448999, 449032,
            449049, 449058, 449081, 449082, 449115, 449124, 449127, 449137,
            449152, 449174, 449181, 449191, 449205, 449228, 449249, 449264,
            449274, 449278, 449284, 449293, 449300, 449319, 449334, 449350,
            449363, 449370, 449408, 449450, 449451, 449461, 449467, 449479,
            449492, 449505, 449520, 449525, 449634, 449645, 449670, 449684,
            449685, 449689, 449701, 449704, 449731, 449741, 449754, 449757,
            449762, 449764, 449774, 449780, 449786, 449788, 449790, 449796,
            449798, 449828, 449834, 449879, 449883, 449884, 449906, 449907,
            449908, 449928, 449945, 449953, 449956, 449965, 449977, 449988,
            449991, 450001, 450030, 450046, 450047, 450061, 450070, 450082,
            450085, 450091, 450093, 450121, 450128, 450132, 450145, 450160,
            450163, 450165, 450168, 450198, 450215, 450225, 450253, 450256,
            450275, 450294, 450296, 450301, 450311, 450324, 450328, 450334,
            450347, 450348, 450354, 450359, 450361, 450365, 450370, 450374,
            450385, 450386, 450387, 450392, 450397, 450409, 450411, 450435,
            450444, 450450, 450465, 450478, 450481, 450486, 450494, 450520,
            450525, 450552, 450557, 450570, 450598, 450605, 450611, 450633,
            450642, 450646, 450689, 450693, 450702, 450705, 450724, 450730,
            450752, 450779, 450784, 450786, 450788, 450802, 450806, 450815,
            450816, 450879, 450901, 450912, 450956, 450979, 450980, 450989,
            450997, 450999, 451001, 451017, 451024, 451029, 451034, 451036,
            451055, 451081, 451146, 451188, 451198, 451203, 451225, 451245,
            451250, 451253, 451256, 451280, 451284, 451290, 451305, 451316,
            451317, 451319, 451323, 451333, 451363, 451364, 451367, 451371,
            451374, 451383, 451391, 451400, 451404, 451410, 451413, 451429,
            451437, 451441, 451443, 451481, 451483, 451488, 451501, 451505,
            451509, 451548, 451598, 451602, 451606, 451617, 451647, 451649,
            451651, 451654, 451672, 451708, 451726, 451730, 451742, 451748,
            451754, 451759, 451762, 451768, 451770, 451774, 451782, 451811,
            451816, 451817, 451854, 451869, 451872, 451873, 451876, 451918,
            451919, 451922, 451923, 451933, 451966, 451969, 451978, 451986,
            451997, 452013, 452042, 452047, 452085, 452090, 452096, 452127,
            452131, 452133, 452138, 452140, 452141, 452142, 452148, 452150,
            452176, 452190, 452191, 452201, 452265, 452280, 452304, 452305,
            452344, 452363, 452367, 452371, 452403, 452404, 452407, 452420,
            452435, 452436, 452450, 452462, 452469, 452476, 452483, 452508,
            452544, 452567, 452588, 452590, 452599, 452601, 452602, 452614,
            452619, 452628, 452659, 452685, 452696, 452701, 452727, 452746,
            452771, 452774, 452791, 452797, 452809, 452816, 452828, 452834,
            452843, 452850, 452860, 452889, 452911, 452915, 452920, 452936,
            452944, 452957, 452967, 452974, 452993, 453002, 453017, 453027,
            453028, 453033, 453040, 453044, 453057, 453058, 453059, 453061,
            453073, 453088, 453093, 453111, 453142, 453182, 453214, 453223,
            453236, 453237, 453269, 453277, 453281, 453287, 453294, 453301,
            453306, 453321, 453325, 453343, 453347, 453359, 453371, 453382,
            453385, 453395, 453402, 453403, 453414, 453419, 453448, 453452,
            453467, 453470, 453482, 453510, 453525, 453534, 453563, 453630,
            453646, 453678, 453701, 453702, 453727, 453732, 453735, 453738,
            453793, 453801, 453802, 453809, 453811, 453821, 453822, 453834,
            453842, 453844, 453859, 453863, 453871, 453872, 453881, 453882,
            453885, 453892, 453905, 453909, 453926, 453933, 453941, 453953,
            453963, 453982, 453988, 453992, 454004, 454009, 454038, 454042,
            454053, 454058, 454059, 454075, 454103, 454107, 454113, 454127,
            454146, 454149, 454161, 454162, 454166, 454196, 454213, 454226,
            454232, 454249, 454263, 454268, 454272, 454284, 454313, 454330,
            454346, 454378, 454402, 454412, 454422, 454451, 454498, 454518,
            454531, 454543, 454551, 454555, 454560, 454561, 454574, 454592,
            454597, 454609, 454632, 454639, 454648, 454656, 454658, 454666,
            454677, 454697, 454711, 454727, 454750, 454760, 454768, 454772,
            454775, 454782, 454816, 454818, 454820, 454851, 454870, 454879,
            454904, 454944, 454957, 454960, 454996, 455052, 455064, 455092,
            455123, 455146, 455147, 455169, 455183, 455184, 455187, 455197,
            455206, 455240, 455244, 455261, 455266, 455273, 455281, 455283,
            455285, 455300, 455308, 455319, 455326, 455344, 455346, 455354,
            455357, 455382, 455393, 455430, 455431, 455434, 455449, 455452,
            455466, 455500, 455505, 455526, 455527, 455532, 455536, 455553,
            455569, 455577, 455580, 455605, 455611, 455622, 455625, 455674,
            455675, 455677, 455687, 455710, 455711, 455718, 455727, 455746,
            455749, 455753, 455758, 455765, 455771, 455796, 455855, 455862,
            455926, 455939, 455958, 455959, 455968, 455970, 455987, 455992,
            455998, 456028, 456029, 456056, 456059, 456061, 456078, 456089,
            456105, 456106, 456112, 456118, 456144, 456159, 456172, 456173,
            456181, 456185, 456187, 456221, 456226, 456244, 456246, 456251,
            456252, 456267, 456273, 456290, 456293, 456316, 456332, 456367,
            456375, 456394, 456398, 456399, 456408, 456412, 456413, 456426,
            456438, 456451, 456457, 456463, 456465, 456497, 456517, 456518,
            456520, 456542, 456596, 456601, 456604, 456616, 456634, 456651,
            456655, 456673, 456688, 456693, 456705, 456712, 456715, 456740,
            456748, 456763, 456764, 456774, 456776, 456791, 456794, 456804,
            456815, 456819, 456822, 456858, 456859, 456879, 456928, 456935,
            456946, 456952, 456979, 456980, 456985, 457005, 457010, 457014,
            457019, 457022, 457028, 457049, 457052, 457055, 457070, 457090,
            457130, 457162, 457164, 457166, 457167, 457186, 457193, 457248,
            457250, 457251, 457255, 457280, 457288, 457309, 457332, 457355,
            457360, 457365, 457375, 457376, 457377, 457390, 457397, 457402,
            457419, 457457, 457460, 457466, 457467, 457485, 457513, 457535,
            457551, 457552, 457564, 457587, 457628, 457638, 457662, 457682,
            457688, 457704, 457720, 457725, 457734, 457739, 457741, 457749,
            457757, 457825, 457832, 457856, 457861, 457872, 457878, 457879,
            457888, 457912, 457918, 457953, 457986, 458013, 458022, 458040,
            458054, 458057, 458078, 458079, 458086, 458089, 458104, 458130,
            458131, 458148, 458161, 458165, 458183, 458184, 458186, 458201,
            458202, 458223, 458224, 458237, 458247, 458253, 458297, 458299,
            458311, 458323, 458334, 458336, 458346, 458349, 458390, 458396,
            458398, 458403, 458406, 458417, 458438, 458446, 458463, 458487,
            458501, 458531, 458540, 458570, 458579, 458592, 458611, 458620,
            458642, 458649, 458650, 458653, 458660, 458663, 458679, 458687,
            458697, 458742, 458744, 458756, 458757, 458769, 458787, 458809,
            458811, 458816, 458848, 458860, 458864, 458866, 458910, 458913,
            458931, 458956, 458969, 458991, 458993, 459007, 459009, 459029,
            459033, 459050, 459059, 459076, 459083, 459092, 459094, 459115,
            459116, 459125, 459143, 459162, 459169, 459190, 459195, 459203])
        selection.add('id', np.isin(ak.to_numpy(events.event), idar))
        print(np.isin(ak.to_numpy(events.event), idar))
        print(np.sum(np.isin(ak.to_numpy(events.event), idar)))

        def normalize(val, cut):
            if cut is None:
                ar = ak.to_numpy(ak.fill_none(val, np.nan))
                return ar
            else:
                ar = ak.to_numpy(ak.fill_none(val[cut], np.nan))
                return ar

        for region, cuts in regions.items():
            # allcuts = set(["id"])
            allcuts = set([])
            cut = selection.all(*allcuts)
            output['cutflow_msd'].fill(dataset=dataset, region=region, genflavor=normalize(genflavor, None),
                                   cut=0, weight=weights.weight(), msd=normalize(msd_matched, None))
            output['cutflow_eta'].fill(dataset=dataset, region=region, genflavor=normalize(genflavor, cut),
                                   cut=0, weight=weights.weight()[cut], eta=normalize(candidatejet.eta, cut))
            output['cutflow_pt'].fill(dataset=dataset, region=region, genflavor=normalize(genflavor, cut),
                                   cut=0, weight=weights.weight()[cut], pt=normalize(candidatejet.pt, cut))
            for i, cut in enumerate(cuts + ['ddcvbpass', 'ddcpass']):
                allcuts.add(cut)
                cut = selection.all(*allcuts)
                output['cutflow_msd'].fill(dataset=dataset, region=region, genflavor=normalize(genflavor, cut),
                                       cut=i + 1, weight=weights.weight()[cut], msd=normalize(msd_matched, cut))
                output['cutflow_eta'].fill(dataset=dataset, region=region, genflavor=normalize(genflavor, cut),
                        cut=i + 1, weight=weights.weight()[cut], eta=normalize(candidatejet.eta, cut))
                output['cutflow_pt'].fill(dataset=dataset, region=region, genflavor=normalize(genflavor, cut),
                        cut=i + 1, weight=weights.weight()[cut], pt=normalize(candidatejet.pt, cut))


        systematics = [
            None,
            'jet_triggerUp',
            'jet_triggerDown',
            'btagWeightUp',
            'btagWeightDown',
            'btagEffStatUp',
            'btagEffStatDown',
        ]

        def fill(region, systematic, wmod=None):
            selections = regions[region]
            cut = selection.all(*selections)
            sname = 'nominal' if systematic is None else systematic
            if wmod is None:
                weight = weights.weight(modifier=systematic)[cut]
            else:
                weight = weights.weight()[cut] * wmod[cut]
            
            if self._v2:
                bvl = candidatejet.btagDDBvLV2
                cvl = candidatejet.btagDDCvLV2
                cvb = candidatejet.btagDDCvBV2
            elif self._v3:
                bvl = candidatejet.particleNet_HbbvsQCD
                cvl = candidatejet.particleNet_HccvsQCD
                cvb = candidatejet.particleNetMD_Xcc/(candidatejet.particleNetMD_Xcc + candidatejet.particleNetMD_Xbb)
            elif self._v4:
                bvl = candidatejet.particleNet_HbbvsQCD
                cvl = candidatejet.btagDDCvLV2
                cvb = candidatejet.particleNetMD_Xcc/(candidatejet.particleNetMD_Xcc + candidatejet.particleNetMD_Xbb)
            else:
                bvl = candidatejet.btagDDBvL
                cvl = candidatejet.btagDDCvL
                cvb = candidatejet.btagDDCvB

            output['templates'].fill(
                dataset=dataset,
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
            if systematic is None:
                output['signal_opt'].fill(
                    dataset=dataset,
                    region=region,
                    genflavor=normalize(genflavor, cut),
                    ddc=normalize(cvl, cut),
                    ddcvb=normalize(cvb, cut),
                    weight=weight,
                )
                output['signal_optb'].fill(
                    dataset=dataset,
                    region=region,
                    genflavor=normalize(genflavor, cut),
                    ddb=normalize(bvl, cut),
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
                if isRealData and systematic is not None:
                    continue
                fill(region, systematic)
            if 'GluGluHToBB' in dataset and 'LHEWeight' in events.fields:
                for i in range(9):
                    fill(region, 'LHEScale_%d' % i, events.LHEScaleWeight[:, i])
                for c in events.LHEWeight.fields[1:]:
                    fill(region, 'LHEWeight_%s' % c, events.LHEWeight[c])

        output["weightStats"] = weights.weightStatistics
        return output

    def postprocess(self, accumulator):
        return accumulator
