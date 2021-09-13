import numpy as np
import awkward as ak
import gzip
import pickle
import cloudpickle
import warnings
import importlib.resources
import correctionlib
from coffea.lookup_tools.lookup_base import lookup_base
from coffea import lookup_tools
from coffea import util

with importlib.resources.path("boostedhiggs.data", "corrections.pkl.gz") as path:
    with gzip.open(path) as fin:
        compiled = pickle.load(fin)

# hotfix some crazy large weights
compiled['2017_pileupweight']._values = np.minimum(5, compiled['2017_pileupweight']._values)
compiled['2018_pileupweight']._values = np.minimum(5, compiled['2018_pileupweight']._values)

with importlib.resources.path("boostedhiggs.data", 'powhegToMinloPtCC.coffea') as filename:
    compiled['powheg_to_nnlops'] = util.load(filename)

class SoftDropWeight(lookup_base):
    def _evaluate(self, pt, eta):
        gpar = np.array([1.00626, -1.06161, 0.0799900, 1.20454])
        cpar = np.array([1.09302, -0.000150068, 3.44866e-07, -2.68100e-10, 8.67440e-14, -1.00114e-17])
        fpar = np.array([1.27212, -0.000571640, 8.37289e-07, -5.20433e-10, 1.45375e-13, -1.50389e-17])
        genw = gpar[0] + gpar[1]*np.power(pt*gpar[2], -gpar[3])
        cenweight = np.polyval(cpar[::-1], pt)
        forweight = np.polyval(fpar[::-1], pt)
        weight = np.where(np.abs(eta) < 1.3, cenweight, forweight)
        return genw*weight


_softdrop_weight = SoftDropWeight()


def corrected_msoftdrop(fatjets):
    sf = _softdrop_weight(fatjets.pt, fatjets.eta)
    sf = np.maximum(1e-5, sf)
    dazsle_msd = (fatjets.subjets * (1 - fatjets.subjets.rawFactor)).sum()
    return dazsle_msd.mass * sf


def n2ddt_shift(fatjets, year='2017'):
    return compiled[f'{year}_n2ddt_rho_pt'](fatjets.qcdrho, fatjets.pt)


def powheg_to_nnlops(genpt):
    return compiled['powheg_to_nnlops'](genpt)

def add_pdf_weight(weights, pdf_weights):
    nom = np.ones(len(weights.weight()))
    up = np.ones(len(weights.weight()))
    down = np.ones(len(weights.weight()))

    # NNPDF31_nnlo_hessian_pdfas
    # https://lhapdfsets.web.cern.ch/current/NNPDF31_nnlo_hessian_pdfas/NNPDF31_nnlo_hessian_pdfas.info
    if pdf_weights is not None and "306000 - 306102" in pdf_weights.__doc__:
        # Hessian PDF weights
        # Eq. 21 of https://arxiv.org/pdf/1510.03865v1.pdf
        arg = pdf_weights[:, 1:-2] - np.ones((len(weights.weight()), 100))
        summed = ak.sum(np.square(arg), axis=1)
        pdf_unc = np.sqrt((1. / 99.) * summed)
        weights.add('PDF_weight', nom, pdf_unc + nom)

        # alpha_S weights
        # Eq. 27 of same ref
        as_unc = 0.5 * (pdf_weights[:, 102] - pdf_weights[:, 101])
        weights.add('aS_weight', nom, as_unc + nom)

        # PDF + alpha_S weights
        # Eq. 28 of same ref
        pdfas_unc = np.sqrt(np.square(pdf_unc) + np.square(as_unc))
        weights.add('PDFaS_weight', nom, pdfas_unc + nom)

    else:
        weights.add('aS_weight', nom, up, down)
        weights.add('PDF_weight', nom, up, down)
        weights.add('PDFaS_weight', nom, up, down)

# Jennet adds PS weights
def add_ps_weight(weights, ps_weights):
    nom = np.ones(len(weights.weight()))
    up_isr = np.ones(len(weights.weight()))
    down_isr = np.ones(len(weights.weight()))
    up_fsr = np.ones(len(weights.weight()))
    down_fsr = np.ones(len(weights.weight()))

    if ps_weights is not None:
        if len(ps_weights[0]) == 4:
            up_isr = ps_weights[:, 0]
            down_isr = ps_weights[:, 2]
            up_fsr = ps_weights[:, 1]
            down_fsr = ps_weights[:, 3]
    #        up = np.maximum.reduce([up_isr, up_fsr, down_isr, down_fsr])
    #        down = np.minimum.reduce([up_isr, up_fsr, down_isr, down_fsr])
        else:
            warnings.warn(f"PS weight vector has length {len(ps_weights[0])}")

    weights.add('UEPS_ISR', nom, up_isr, down_isr)
    weights.add('UEPS_FSR', nom, up_fsr, down_fsr)


def add_pileup_weight(weights, nPU, year='2017', dataset=None):
    if year == '2017' and dataset in compiled['2017_pileupweight_dataset']:
        weights.add(
            'pileup_weight',
            compiled['2017_pileupweight_dataset'][dataset](nPU),
            compiled['2017_pileupweight_dataset_puUp'][dataset](nPU),
            compiled['2017_pileupweight_dataset_puDown'][dataset](nPU),
        )
    else:
        weights.add(
            'pileup_weight',
            compiled[f'{year}_pileupweight'](nPU),
            compiled[f'{year}_pileupweight_puUp'](nPU),
            compiled[f'{year}_pileupweight_puDown'](nPU),
        )


def add_VJets_NLOkFactor(weights, genBosonPt, year, dataset):
    if year == '2017' and 'ZJetsToQQ_HT' in dataset:
        nlo_over_lo_qcd = compiled['2017_Z_nlo_qcd'](genBosonPt)
        nlo_over_lo_ewk = compiled['Z_nlo_over_lo_ewk'](genBosonPt)
    elif year == '2017' and 'WJetsToQQ_HT' in dataset:
        nlo_over_lo_qcd = compiled['2017_W_nlo_qcd'](genBosonPt)
        nlo_over_lo_ewk = compiled['W_nlo_over_lo_ewk'](genBosonPt)
    elif year == '2016' and 'DYJetsToQQ' in dataset:
        nlo_over_lo_qcd = compiled['2016_Z_nlo_qcd'](genBosonPt)
        nlo_over_lo_ewk = compiled['Z_nlo_over_lo_ewk'](genBosonPt)
    elif year == '2016' and 'WJetsToQQ' in dataset:
        nlo_over_lo_qcd = compiled['2016_W_nlo_qcd'](genBosonPt)
        nlo_over_lo_ewk = compiled['W_nlo_over_lo_ewk'](genBosonPt)
    else:
        return
    weights.add('VJets_NLOkFactor', nlo_over_lo_qcd * nlo_over_lo_ewk)


with importlib.resources.path("boostedhiggs.data", "vjets_corrections.json") as filename:
    vjets_kfactors = correctionlib.CorrectionSet.from_file(str(filename))


def add_VJets_kFactors(weights, genpart, dataset):
    """Revised version of add_VJets_NLOkFactor, for both NLO EW and ~NNLO QCD"""
    def get_vpt(check_offshell=False):
        """Only the leptonic samples have no resonance in the decay tree, and only
        when M is beyond the configured Breit-Wigner cutoff (usually 15*width)
        """
        boson = ak.firsts(genpart[
            ((genpart.pdgId == 23)|(abs(genpart.pdgId) == 24))
            & genpart.hasFlags(["fromHardProcess", "isLastCopy"])
        ])
        if check_offshell:
            offshell = genpart[
                genpart.hasFlags(["fromHardProcess", "isLastCopy"])
                & ak.is_none(boson)
                & (abs(genpart.pdgId) >= 11) & (abs(genpart.pdgId) <= 16)
            ].sum()
            return ak.where(ak.is_none(boson.pt), offshell.pt, boson.pt)
        return np.array(ak.fill_none(boson.pt, 0.))

    common_systs = [
        "d1K_NLO",
        "d2K_NLO",
        "d3K_NLO",
        "d1kappa_EW",
    ]
    zsysts = common_systs + [
        "Z_d2kappa_EW",
        "Z_d3kappa_EW",
    ]
    wsysts = common_systs + [
        "W_d2kappa_EW",
        "W_d3kappa_EW",
    ]

    def add_systs(systlist, qcdcorr, ewkcorr, vpt):
        ewknom = ewkcorr.evaluate("nominal", vpt)
        weights.add("vjets_nominal", qcdcorr * ewknom if qcdcorr is not None else ewknom)
        ones = np.ones_like(vpt)
        for syst in systlist:
            weights.add(syst, ones, ewkcorr.evaluate(syst + "_up", vpt) / ewknom, ewkcorr.evaluate(syst + "_down", vpt) / ewknom)

    if "ZJetsToQQ_HT" in dataset and "TuneCUETP8M1" in dataset:
        vpt = get_vpt()
        qcdcorr = vjets_kfactors["Z_MLM2016toFXFX"].evaluate(vpt)
        ewkcorr = vjets_kfactors["Z_FixedOrderComponent"]
        add_systs(zsysts, qcdcorr, ewkcorr, vpt)
    elif "WJetsToQQ_HT" in dataset and "TuneCUETP8M1" in dataset:
        vpt = get_vpt()
        qcdcorr = vjets_kfactors["W_MLM2016toFXFX"].evaluate(vpt)
        ewkcorr = vjets_kfactors["W_FixedOrderComponent"]
        add_systs(wsysts, qcdcorr, ewkcorr, vpt)
    elif "ZJetsToQQ_HT" in dataset and "TuneCP5" in dataset:
        vpt = get_vpt()
        qcdcorr = vjets_kfactors["Z_MLM2017toFXFX"].evaluate(vpt)
        ewkcorr = vjets_kfactors["Z_FixedOrderComponent"]
        add_systs(zsysts, qcdcorr, ewkcorr, vpt)
    elif "WJetsToQQ_HT" in dataset and "TuneCP5" in dataset:
        vpt = get_vpt()
        qcdcorr = vjets_kfactors["W_MLM2017toFXFX"].evaluate(vpt)
        ewkcorr = vjets_kfactors["W_FixedOrderComponent"]
        add_systs(wsysts, qcdcorr, ewkcorr, vpt)
    elif ("DY1JetsToLL_M-50" in dataset or "DY2JetsToLL_M-50" in dataset) and "amcnloFXFX" in dataset:
        vpt = get_vpt(check_offshell=True)
        ewkcorr = vjets_kfactors["Z_FixedOrderComponent"]
        add_systs(zsysts, None, ewkcorr, vpt)
    elif ("W1JetsToLNu" in dataset or "W2JetsToLNu" in dataset) and "amcnloFXFX" in dataset:
        vpt = get_vpt(check_offshell=True)
        ewkcorr = vjets_kfactors["W_FixedOrderComponent"]
        add_systs(wsysts, None, ewkcorr, vpt)


def add_jetTriggerWeight(weights, jet_msd, jet_pt, year):
    nom = compiled[f'{year}_trigweight_msd_pt'](jet_msd, jet_pt)
    up = compiled[f'{year}_trigweight_msd_pt_trigweightUp'](jet_msd, jet_pt)
    down = compiled[f'{year}_trigweight_msd_pt_trigweightDown'](jet_msd, jet_pt)
    weights.add('jet_trigger', nom, up, down)


with importlib.resources.path("boostedhiggs.data", "fatjet_triggerSF.json") as filename:
    jet_triggerSF = correctionlib.CorrectionSet.from_file(str(filename))


def add_jetTriggerSF(weights, leadingjet, year, selection):
    def mask(w):
        return np.where(selection.all('noleptons'), w, 1.)

    jet_pt = np.array(ak.fill_none(leadingjet.pt, 0.))
    jet_msd = np.array(ak.fill_none(leadingjet.msoftdrop, 0.))  # note: uncorrected
    nom = mask(jet_triggerSF[f'fatjet_triggerSF{year}'].evaluate("nominal", jet_pt, jet_msd))
    up = mask(jet_triggerSF[f'fatjet_triggerSF{year}'].evaluate("stat_up", jet_pt, jet_msd))
    down = mask(jet_triggerSF[f'fatjet_triggerSF{year}'].evaluate("stat_dn", jet_pt, jet_msd))
    weights.add('jet_trigger', nom, up, down)

def add_mutriggerSF(weights, leadingmuon, year, selection):
    def mask(w):
        return np.where(selection.all('onemuon'), w, 1.)

    mu_pt = np.array(ak.fill_none(leadingmuon.pt, 0.))
    mu_eta = np.array(ak.fill_none(abs(leadingmuon.eta), 0.))
    nom = mask(compiled[f'{year}_mutrigweight_pt_abseta'](mu_pt, mu_eta))
    shift = mask(compiled[f'{year}_mutrigweight_pt_abseta_mutrigweightShift'](mu_pt, mu_eta))
    weights.add('mu_trigger', nom, shift, shift=True)

def add_mucorrectionsSF(weights, leadingmuon, year, selection):
    def mask(w):
        return np.where(selection.all('onemuon'), w, 1.)

    mu_pt = np.array(ak.fill_none(leadingmuon.pt, 0.))
    mu_eta = np.array(ak.fill_none(abs(leadingmuon.eta), 0.))
    nom = mask(compiled[f'{year}_muidweight_abseta_pt'](mu_eta, mu_pt))
    shift = mask(compiled[f'{year}_muidweight_abseta_pt_muidweightShift'](mu_eta, mu_pt))
    weights.add('mu_idweight', nom, shift, shift=True)

    nom = mask(compiled[f'{year}_muisoweight_abseta_pt'](mu_eta, mu_pt))
    shift = mask(compiled[f'{year}_muisoweight_abseta_pt_muisoweightShift'](mu_eta, mu_pt))
    weights.add('mu_isoweight', nom, shift, shift=True)


with importlib.resources.path("boostedhiggs.data", "jec_compiled.pkl.gz") as path:
    with gzip.open(path) as fin:
        jmestuff = cloudpickle.load(fin)

jet_factory = jmestuff["jet_factory"]
fatjet_factory = jmestuff["fatjet_factory"]
met_factory = jmestuff["met_factory"]


def add_jec_variables(jets, event_rho):
    jets["pt_raw"] = (1 - jets.rawFactor)*jets.pt
    jets["mass_raw"] = (1 - jets.rawFactor)*jets.mass
    jets["pt_gen"] = ak.values_astype(ak.fill_none(jets.matched_gen.pt, 0), np.float32)
    jets["event_rho"] = ak.broadcast_arrays(event_rho, jets.pt)[0]
    return jets


def build_lumimask(filename):
    from coffea.lumi_tools import LumiMask
    with importlib.resources.path("boostedhiggs.data", filename) as path:
        return LumiMask(path)


lumiMasks = {
    '2016': build_lumimask('Cert_271036-284044_13TeV_23Sep2016ReReco_Collisions16_JSON.txt'),
    '2017': build_lumimask('Cert_294927-306462_13TeV_EOY2017ReReco_Collisions17_JSON_v1.txt'),
    '2018': build_lumimask('Cert_314472-325175_13TeV_17SeptEarlyReReco2018ABC_PromptEraD_Collisions18_JSON.txt'),
}
