import awkward as ak
from uproot_methods import TVector2Array, TLorentzVectorArray


def buildevents(df):
    events = ak.Table()

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
            ddb=df['FatJet_btagDDBvL'],
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
