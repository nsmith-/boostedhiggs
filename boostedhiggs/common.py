import awkward1 as ak


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
    genflavor = ak.any(childid == 5, axis=-1) * 3 + ak.any(childid == 4, axis=-1) * 2 + ak.any(childid < 4, axis=-1) * 1
    return ak.fill_none(genflavor, 0)
