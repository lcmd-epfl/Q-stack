import os
import numpy as np
from qstack.tools import correct_num_threads
from . import utils
from .compute_rho_spahm import get_repr
from .parser import SpahmParser


def main(args=None):
    parser = SpahmParser(description='This program computes the SPAHM(b) representation for a given molecular system or a list of thereof', unified=True, bond=True)
    args = parser.parse_args(args=args)
    if args.print>0:
        print(vars(args))
    correct_num_threads()

    if args.name_out is None:
        args.name_out = os.path.splitext(args.filename)[0]

    if args.filename.endswith('xyz'):
        xyzlist = [args.filename]
        charge  = np.array([int(args.charge) if (args.charge != "None") else 0])
        spin    = np.array([int(args.spin)   if (args.spin != "None") else None])
    else:
        xyzlistfile = args.filename
        xyzlist = utils.get_xyzlist(xyzlistfile)
        if args.charge is not None:
            charge = utils.get_chsp(args.charge, len(xyzlist))
        else:
            charge = np.full(len(xyzlist), None, dtype=object)
        if args.spin is not None:
            spin = utils.get_chsp(args.spin, len(xyzlist))
        else:
            spin = np.full(len(xyzlist), None, dtype=object)

    mols = utils.load_mols(xyzlist, charge, spin, args.basis, args.print, units=args.units, ecp=args.ecp)

    reps = get_repr("bond",
        mols=mols, xyzlist=xyzlist, guess=args.guess, xc=args.xc, spin=spin,
        readdm=args.readdm, printlevel=args.print,
        pairfile=args.pairfile, dump_and_exit=args.dump_and_exit, same_basis=args.same_basis,
        bpath=args.bpath, cutoff=args.cutoff, omods=args.omod, with_symbols=args.with_symbols,
        elements=args.elements, only_m0=args.only_m0, zeros=args.zeros, split=(args.split>0), only_z=args.only_z,
    )

    if args.print > 0:
        print(reps.shape)
    if args.merge:
        if (spin == None).all():
            mod_iter = [(reps, '')]
        else:
            mod_iter = [(reps, '_'+'_'.join(args.omod))]
    else:
        mod_iter = [(modvec, '_'+omod) for modvec, omod in zip(reps, args.omod, strict=True)]

    for modvec, mod_suffix in mod_iter:
        if args.split >=2:
            for mol_i, molvec in enumerate(modvec):
                filename = xyzlist[mol_i]
                basename = os.path.splitext(os.path.basename(filename))[0]
                np.save(args.name_out + '_' + basename + mod_suffix, molvec)
        else:
            np.save(args.name_out + mod_suffix, modvec)

if __name__ == "__main__":
    main()
