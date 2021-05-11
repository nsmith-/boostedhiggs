import awkward as ak
import numpy as np

def getBosons(genparticles):
    absid = abs(genparticles.pdgId)
    return genparticles[
        # no gluons
        (absid >= 22)
        & (absid <= 25)
        & genparticles.hasFlags(['fromHardProcess', 'isLastCopy'])
    ]

def bosonFlavor(bosons):
    childid = abs(bosons.children.pdgId)
    genflavor = ak.any(childid == 5, axis=-1) * 3 + ak.any(childid == 4, axis=-1) * 2 + ak.all(childid < 4, axis=-1) * 1
    return ak.fill_none(genflavor, 0)


def pass_json(run, luminosityBlock, lumi_mask):
    if str(run) not in lumi_mask.keys():
        return False
    for lrange in lumi_mask[str(run)]:
        if int(lrange[0]) <= luminosityBlock < int(lrange[1]):
            return True
    return False

def pass_json_array(runs, luminosityBlocks, lumi_mask):
    out = []
    for run, block in zip(runs, luminosityBlocks):
        out.append(pass_json(run, block, lumi_mask))
    return np.array(out)