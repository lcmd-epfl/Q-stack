import numpy as np
import itertools
from types import SimpleNamespace


defaults = SimpleNamespace(sigma2=0.05, r0=0.1, rcut=4.8, dgrid2=0.03, theta0=20.0*np.pi/180.0, sigma3=0.05, dgrid3=0.03)


def get_mbtypes(qs, qml=False):

    # all the elements
    elements = itertools.chain.from_iterable(list(i) for i in qs)
    if qml:
        # bad because sets are not ordered
        elements = np.array(list(set(elements)))
    else:
        elements = np.unique(list(elements))

    # max number of atoms of each element across mols
    max_nq_in_mol = np.max([np.count_nonzero(q==elements[:,None], axis=1) for q in qs], axis=0)

    pairs = [(q, q) for q in elements] + [*itertools.combinations(elements, 2)]

    triples = []
    for q1 in elements:
        for (q2, q3) in pairs:
            for triple in [(q1, q2, q3), (q1, q3, q2), (q2, q1, q3)]:
                if (triple not in triples) and (triple[::-1] not in triples):
                    nq_in_triple = np.count_nonzero(triple==elements[:,None], axis=1)
                    if np.all(nq_in_triple <= max_nq_in_mol):
                        triples.append(triple)
    return {1: elements, 2: pairs, 3: triples}



def get_two_body(i, mbtype, q, dist,
                 r0=defaults.r0, rcut=defaults.rcut,
                 sigma=defaults.sigma2, dgrid=defaults.dgrid2):

    ngrid = int((rcut - r0)/dgrid) + 1
    rgrid = np.linspace(r0, rcut, ngrid)

    qi = q[i]
    if qi not in mbtype:
        return np.zeros_like(rgrid)

    if qi==mbtype[0]:
        (q1, q2) = mbtype
    else:
        (q1, q2) = mbtype[::-1]

    london = q1 * q2 / rgrid**6
    delta_norm = 1.0/(sigma * np.sqrt(2*np.pi))
    deltas = np.zeros(ngrid)

    j = np.where(q==q2)[0]
    j = j[j!=i]

    dist_ij = dist[np.ix_(j,[i])]
    dist_ij = dist_ij[np.where(dist_ij<rcut)[0]]

    delta = delta_norm * np.exp(-(rgrid-dist_ij)**2 * 0.5/(sigma**2))
    deltas += np.sum(delta, axis=0)

    return 0.5 * dgrid * london * deltas



def get_three_body(j, mbtype, q, r, dist,
                   rcut=defaults.rcut, theta0=defaults.theta0,
                   sigma=defaults.sigma3, dgrid=defaults.dgrid3):

    def get_cos(a, b, c):
        v1 = r[a] - r[b]
        v2 = r[c] - r[b]
        return v1 @ v2 / (dist[a,b] * dist[b,c])

    theta1 = np.pi + theta0
    ngrid = int((theta1+theta0)/dgrid) + 1
    tgrid = np.linspace(-theta0, theta1, ngrid)
    spectrum = np.zeros_like(tgrid)

    (q1, q2, q3) = mbtype
    if q[j] != q2 or q1 not in q or q3 not in q:
        return spectrum

    delta_norm = 1.0/(sigma * np.sqrt(2*np.pi))

    for i in np.where(q==q1)[0]:
        if i==j or dist[i,j]>rcut:
            continue

        for k in np.where(q==q3)[0]:
            if k==j or k==i or dist[i,k]>rcut or dist[j,k]>rcut:
                continue

            cos_ikj = get_cos(i, k, j)
            cos_jik = get_cos(j, i, k)
            atm = (1.0 + np.cos(tgrid) * cos_ikj * cos_jik) / (dist[i,j]*dist[i,k]*dist[j,k])**3

            cos_ijk = get_cos(i, j, k)
            theta_ijk = np.arccos(cos_ijk)
            delta = delta_norm * np.exp(-(tgrid-theta_ijk)**2 * 0.5/(sigma**2))

            spectrum += delta * atm

    if q1==q3:
        spectrum *= 0.5
    return spectrum * dgrid * q1 * q2 * q3 / 3.0



def get_slatm(q, r, mbtypes, qml_compatible=True, stack_all=True,
              r0=defaults.r0, rcut=defaults.rcut, sigma2=defaults.sigma2, dgrid2=defaults.dgrid2,
              theta0=defaults.theta0, sigma3=defaults.sigma3, dgrid3=defaults.dgrid3):

    natoms = len(q)
    dist = np.zeros((natoms, natoms))
    for (i,j) in itertools.product(range(natoms), range(natoms)):
        dist[i,j] = np.linalg.norm(r[i]-r[j])

    slatm = []
    for i, qi in enumerate(q):

        # 1-body terms
        if qml_compatible:
            slatm1b = (mbtypes[1] == qi)*qi.astype(float)
        else:
            slatm1b = np.array((float(qi,)))

        # 2-body terms
        slatm2b = []
        for mbtype in mbtypes[2]:
            if (not qml_compatible) and (qi not in mbtype):
                continue
            two_body = get_two_body(i, mbtype, q, dist,
                                    sigma=sigma2, dgrid=dgrid2, r0=r0, rcut=rcut)
            slatm2b.append(two_body)

        # 3-body terms
        slatm3b = []
        for mbtype in mbtypes[3]:
            if (not qml_compatible) and (qi != mbtype[1]):
                continue
            three_body = get_three_body(i, mbtype, q, r, dist,
                                        sigma=sigma3, dgrid=dgrid3, rcut=rcut, theta0=theta0)
            slatm3b.append(three_body)

        # concatenate
        if stack_all:
            slatm2b = np.hstack(slatm2b)
            slatm3b = np.hstack(slatm3b)
            slatm.append(np.hstack((slatm1b, slatm2b, slatm3b)))
        else:
            slatm.append({1: slatm1b, 2: slatm2b, 3: slatm3b})

    if stack_all:
        slatm = np.vstack(slatm)

    return slatm



def get_slatm_for_dataset(molecules,
                          progress=False,
                          qml_mbtypes=True, qml_compatible=True, stack_all=True,
                          r0=defaults.r0, rcut=defaults.rcut, sigma2=defaults.sigma2, dgrid2=defaults.dgrid2,
                          theta0=defaults.theta0, sigma3=defaults.sigma3, dgrid3=defaults.dgrid3):

    if isinstance(molecules[0], str):
        import ase
        molecules = [ase.io.read(xyz) for xyz in molecules]

    qs = [mol.numbers for mol in molecules]
    mbtypes = get_mbtypes(qs, qml=True)

    if progress:
        import tqdm
        molecules = tqdm.tqdm(molecules)

    slatm = []
    for mol in molecules:
        slatm.append(get_slatm(mol.numbers, mol.positions, mbtypes,
                               qml_compatible=qml_compatible, stack_all=stack_all,
                               r0=r0, rcut=rcut, sigma2=sigma2, dgrid2=dgrid2,
                               theta0=theta0, sigma3=sigma3, dgrid3=dgrid3))
    if stack_all:
        slatm = np.vstack(slatm)

    return slatm