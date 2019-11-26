import awkward as ak
from uproot_methods import TVector2Array, TLorentzVectorArray


def buildevents(df):
    events = ak.Table()

    if 'genWeight' in df:
        events['genWeight'] = df['genWeight']
        events['Pileup_nPU'] = df['Pileup_nPU']

        events['genpart'] = ak.JaggedArray.fromcounts(
            df['nGenPart'],
            ak.Table.named(
                'particle',
                p4=TLorentzVectorArray.from_ptetaphim(
                    df['GenPart_pt'],
                    df['GenPart_eta'],
                    df['GenPart_phi'],
                    df['GenPart_mass'],
                ),
                pdgId=df['GenPart_pdgId'],
                genPartIdxMother=df['GenPart_genPartIdxMother'],
                statusFlags=df['GenPart_statusFlags'],
            ),
        )

    events['fatjets'] = ak.JaggedArray.fromcounts(
        df['nFatJet'],
        ak.Table.named(
            'fatjet',
            p4=TLorentzVectorArray.from_ptetaphim(
                df['FatJet_pt'],
                df['FatJet_eta'],
                df['FatJet_phi'],
                df['FatJet_mass'],
            ),
            msoftdrop=ak.MaskedArray(df['FatJet_msoftdrop'] <= 0, df['FatJet_msoftdrop']),
            area=df['FatJet_area'],
            n2=df['FatJet_n2b1'],
            btagDDBvL=df['FatJet_btagDDBvL'],
            btagDDCvL=df['FatJet_btagDDCvL'],
            btagDDCvB=df['FatJet_btagDDCvB'],
            jetId=df['FatJet_jetId'],
        ),
    )

    events['jets'] = ak.JaggedArray.fromcounts(
        df['nJet'],
        ak.Table.named(
            'jet',
            p4=TLorentzVectorArray.from_ptetaphim(
                df['Jet_pt'],
                df['Jet_eta'],
                df['Jet_phi'],
                df['Jet_mass'],
            ),
            deepcsvb=df['Jet_btagDeepB'],
            hadronFlavor=df['Jet_hadronFlavour'],
            jetId=df['Jet_jetId'],
        ),
    )

    events['met'] = TVector2Array.from_polar(
        df['MET_pt'],
        df['MET_phi']
    )

    events['electrons'] = ak.JaggedArray.fromcounts(
        df['nElectron'],
        ak.Table.named(
            'electron',
            p4=TLorentzVectorArray.from_ptetaphim(
                df['Electron_pt'],
                df['Electron_eta'],
                df['Electron_phi'],
                df['Electron_mass'],
            ),
            cutBased=df['Electron_cutBased'],
        ),
    )

    events['muons'] = ak.JaggedArray.fromcounts(
        df['nMuon'],
        ak.Table.named(
            'muon',
            p4=TLorentzVectorArray.from_ptetaphim(
                df['Muon_pt'],
                df['Muon_eta'],
                df['Muon_phi'],
                df['Muon_mass'],
            ),
            looseId=df['Muon_looseId'],
            pfRelIso04_all=df['Muon_pfRelIso04_all'],
        ),
    )

    events['taus'] = ak.JaggedArray.fromcounts(
        df['nTau'],
        ak.Table.named(
            'tau',
            p4=TLorentzVectorArray.from_ptetaphim(
                df['Tau_pt'],
                df['Tau_eta'],
                df['Tau_phi'],
                df['Tau_mass'],
            ),
            idDecayMode=df['Tau_idDecayMode'],
        ),
    )

    return events
