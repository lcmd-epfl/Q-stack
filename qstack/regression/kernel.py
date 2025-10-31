import os
import numpy as np
from qstack.tools import correct_num_threads
from .kernel_utils import get_kernel, defaults
from .parser import RegressionParser


def kernel(X, Y=None, sigma=defaults.sigma, akernel=defaults.kernel, gkernel=defaults.gkernel, gdict=defaults.gdict):
    """ Computes a kernel between sets A and B (or A and A) using their representations.

    Args:
        X (list of arrays): Representation of A
        Y (list of arrays): Representation of B.
        sigma (): Sigma hyperparameter.
        akernel (): Kernel type (G for Gaussian, L for Laplacian, and myL for Laplacian for open-shell systems).
        gkernel (): Global kernel type (agv for average, rem for REMatch kernel, None for local kernels).
        gdict (): Dictionary like input string to initialize global kernel parameters. Defaults to {'alpha':1.0, 'normalize':1}.

    Returns:
        A numpy ndarray containing the kernel.
    """
    if Y is None:
        Y = X
    kernel = get_kernel(akernel, [gkernel, gdict])
    K = kernel(X, Y, 1.0/sigma)
    return K


def main():
    parser = RegressionParser(description='This program computes kernel.', hyperparameters_set='single')
    parser.remove_argument('prop')
    parser.remove_argument('test_size')
    parser.remove_argument('train_size')
    parser.remove_argument('readk')
    parser.remove_argument('sparse')
    parser.remove_argument('random_state')
    parser.remove_argument('eta')
    parser.add_argument('--dir',  type=str, dest='dir',  default='./',  help='directory to save the output in')
    args = parser.parse_args()
    print(vars(args))
    if(args.ll):
        correct_num_threads()
    if os.path.isfile(args.repr):
        X = np.load(args.repr)
    else:
        x_files = [os.path.join(args.repr, f) for f in os.listdir(args.repr) if os.path.isfile(os.path.join(args.repr, f))]
        X = [np.load(f, allow_pickle=True) for f in x_files]
    K = kernel(X, sigma=args.sigma, akernel=args.akernel, gkernel=args.gkernel, gdict=args.gdict)
    repr_str = os.path.splitext(os.path.basename(args.repr))[0]
    gkernel_str = '_'.join([str(v) for v in args.gdict.values()])
    fname = f"{args.dir}/K_{repr_str}_{args.akernel}_{args.gkernel}_norm{gkernel_str}_{args.sigma:e}"
    np.save(fname, K)


if __name__ == "__main__":
    main()
