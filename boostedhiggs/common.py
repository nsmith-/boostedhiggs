import numpy as np


def getBosons(events):
    absid = np.abs(events.GenPart.pdgId)
    return events.GenPart[
        # no gluons
        (absid >= 22)
        & (absid <= 25)
        & events.GenPart.hasFlags(['fromHardProcess', 'isLastCopy'])
    ]


def match(left, right, metric, maximum=np.inf):
    '''Matching utility

    For each item in ``left``, find closest item in ``right``, using function ``metric``.
    The function must accept two broadcast-compatible arrays and return a numeric array.
    If maximum is specified, mask matched elements where metric was greater than it.
    '''
    lr = left.cross(right, nested=True)
    mval = metric(lr.i0, lr.i1)
    idx = mval.argmin()
    if maximum < np.inf:
        matched = lr.i1[idx[mval[idx] < maximum]]
        return matched.copy(content=matched.content.pad(1)).flatten(axis=1)
    else:
        return lr.i1[idx]


def matchedBosonFlavor(candidates, bosons, maxdR=0.8):
    matched = match(candidates, bosons, lambda a, b: a.delta_r(b), maxdR)
    childid = abs(matched.children.pdgId)
    genflavor = (childid == 5).any() * 3 + (childid == 4).any() * 2 + (childid < 4).all() * 1
    return genflavor.fillna(0)
