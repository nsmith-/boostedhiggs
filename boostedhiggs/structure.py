import numpy as np
import awkward as ak
from uproot_methods import TVector2Array, TLorentzVectorArray
from uproot_methods.classes.TLorentzVector import PtEtaPhiMassArrayMethods
from uproot_methods.classes.TVector2 import ArrayMethods as RhoPhiArrayMethods

VirtualTVector2Array = ak.Methods.mixin(RhoPhiArrayMethods, ak.VirtualArray)
VirtualTLorentzVectorArray = ak.Methods.mixin(PtEtaPhiMassArrayMethods, ak.VirtualArray)


# this is a prototype that will move into coffea dataframe itself, do not copy!
def _getvirtual(df):
    def get(item):
        branch = df._tree[item]
        # pretty safe for NanoAOD
        counts_branch = 'n' + item.split('_')[0]
        isjagged = counts_branch in df and item != counts_branch
        if isjagged:
            memo_sum = '_sum_' + counts_branch
            if memo_sum not in df:
                df[memo_sum] = df[counts_branch].sum()
            size = df[memo_sum]
            interp = branch.interpretation.content.type
        else:
            size = df.size
            interp = branch.interpretation.type
        return ak.VirtualArray(
            df.__getitem__,
            item,
            type=ak.type.ArrayType(size, interp),
        )

    return get


def _embed_subjets(events):
    sj1 = events.fatjets.subJetIdx1.content
    sj2 = events.fatjets.subJetIdx2.content
    sjcontent = ak.concatenate(
        [
            ak.JaggedArray.fromoffsets(np.arange(len(sj1) + 1), sj1),
            ak.JaggedArray.fromoffsets(np.arange(len(sj2) + 1), sj2),
        ],
        axis=1
    )
    subjetidx = events.fatjets.copy(content=sjcontent)
    subjetidx = subjetidx[subjetidx >= 0] + events.subjets.starts
    events.fatjets['subjets'] = subjetidx.copy(
        content=subjetidx.content.copy(
            content=events.subjets.content[subjetidx.flatten().flatten()]
        )
    )


def buildevents(df, fatjet='FatJet', subjet='SubJet', usemask=False, virtual=False):
    events = ak.Table.named('event')

    if virtual:
        get = _getvirtual(df)
    else:
        get = df.__getitem__

    if 'genWeight' in df:
        events['genWeight'] = get('genWeight')
        events['Pileup_nPU'] = get('Pileup_nPU')

        events['genpart'] = ak.JaggedArray.fromcounts(
            get('nGenPart'),
            ak.Table.named(
                'particle',
                p4=TLorentzVectorArray.from_ptetaphim(
                    get('GenPart_pt'),
                    get('GenPart_eta'),
                    get('GenPart_phi'),
                    get('GenPart_mass'),
                ),
                pdgId=get('GenPart_pdgId'),
                genPartIdxMother=get('GenPart_genPartIdxMother'),
                statusFlags=get('GenPart_statusFlags'),
            ),
        )

    events['fatjets'] = ak.JaggedArray.fromcounts(
        get(f'n{fatjet}'),
        ak.Table.named(
            'fatjet',
            p4=TLorentzVectorArray.from_ptetaphim(
                get(f'{fatjet}_pt'),
                get(f'{fatjet}_eta'),
                get(f'{fatjet}_phi'),
                get(f'{fatjet}_mass'),
            ),
            msoftdrop=get(f'{fatjet}_msoftdrop'),  # ak.MaskedArray(get(f'{fatjet}_msoftdrop') <= 0, get(f'{fatjet}_msoftdrop')) if usemask else np.maximum(1e-5, get(f'{fatjet}_msoftdrop')),
            area=get(f'{fatjet}_area'),
            n2=get(f'{fatjet}_n2b1'),
            btagDDBvL=get(f'{fatjet}_btagDDBvL'),
            btagDDCvL=get(f'{fatjet}_btagDDCvL'),
            btagDDCvB=get(f'{fatjet}_btagDDCvB'),
            jetId=get(f'{fatjet}_jetId'),
            subJetIdx1=get(f'{fatjet}_subJetIdx1'),
            subJetIdx2=get(f'{fatjet}_subJetIdx2'),
        ),
    )

    events['subjets'] = ak.JaggedArray.fromcounts(
        get(f'n{subjet}'),
        ak.Table.named(
            'subjet',
            p4=TLorentzVectorArray.from_ptetaphim(
                get(f'{subjet}_pt'),
                get(f'{subjet}_eta'),
                get(f'{subjet}_phi'),
                get(f'{subjet}_mass'),
            ),
            n2=get(f'{subjet}_n2b1'),
            btagDeepB=get(f'{subjet}_btagDeepB'),
        ),
    )
    _embed_subjets(events)

    events['jets'] = ak.JaggedArray.fromcounts(
        get('nJet'),
        ak.Table.named(
            'jet',
            p4=TLorentzVectorArray.from_ptetaphim(
                get('Jet_pt'),
                get('Jet_eta'),
                get('Jet_phi'),
                get('Jet_mass'),
            ),
            deepcsvb=get('Jet_btagDeepB'),
            hadronFlavor=get('Jet_hadronFlavour'),
            jetId=get('Jet_jetId'),
        ),
    )

    events['met'] = VirtualTVector2Array(
        TVector2Array.from_polar,
        (get('MET_pt'), get('MET_phi')),
        type=ak.type.ArrayType(df.size, RhoPhiArrayMethods),
    )

    events['electrons'] = ak.JaggedArray.fromcounts(
        get('nElectron'),
        ak.Table.named(
            'electron',
            p4=TLorentzVectorArray.from_ptetaphim(
                get('Electron_pt'),
                get('Electron_eta'),
                get('Electron_phi'),
                get('Electron_mass'),
            ),
            cutBased=get('Electron_cutBased'),
        ),
    )

    events['muons'] = ak.JaggedArray.fromcounts(
        get('nMuon'),
        ak.Table.named(
            'muon',
            p4=TLorentzVectorArray.from_ptetaphim(
                get('Muon_pt'),
                get('Muon_eta'),
                get('Muon_phi'),
                get('Muon_mass'),
            ),
            looseId=get('Muon_looseId'),
            pfRelIso04_all=get('Muon_pfRelIso04_all'),
        ),
    )

    events['taus'] = ak.JaggedArray.fromcounts(
        get('nTau'),
        ak.Table.named(
            'tau',
            p4=TLorentzVectorArray.from_ptetaphim(
                get('Tau_pt'),
                get('Tau_eta'),
                get('Tau_phi'),
                get('Tau_mass'),
            ),
            idDecayMode=get('Tau_idDecayMode'),
        ),
    )

    return events
