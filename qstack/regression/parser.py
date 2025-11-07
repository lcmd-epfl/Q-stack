"""Command-line argument parser for regression tasks."""

import argparse
from qstack.tools import FlexParser
from .kernel_utils import defaults, ParseKwargs, local_kernels_dict, global_kernels_dict


class RegressionParser(FlexParser):
    """Custom argument parser for kernel ridge regression tasks.

    Provides pre-configured argument sets for KRR routines.

    Args:
        hyperparameters_set (str, optional): Hyperparameter mode. Options:
            - None: No hyperparameter arguments added
            - 'single': Single eta/sigma values for direct regression
            - 'array': Multiple eta/sigma values for grid search/cross-validation
            Defaults to None.
        **kwargs: Additional arguments passed to ArgumentParser.

    Attributes:
        Standard arguments added for all modes:
        - x (--x): Path to molecular representations file
            - y (--y): Path to target properties file
            - akernel (--akernel): Local/atomic kernel type (Gaussian, Laplacian, etc.)
        - gkernel (--gkernel): Global/molecular kernel type (average, REMatch)
        - gdict (--gdict): Global kernel parameters dictionary
            - test (--test): Test set fraction (0.0-1.0)
        - train (--train): Training set fraction list for learning curvers
          (0.0-1.0 where 1.0 means full training set minus test set)
        - ll (--ll): Thread correction flag for running on clusters
            - readkernel (--readkernel): Flag if input is pre-computed kernel
            - sparse (--sparse): Sparse regression basis size
            - random_state (--random_state): Random seed for reproducibility

        Additional for 'single' mode:
        - eta (--eta): Single regularization parameter
            - sigma (--sigma): Single kernel width parameter

        Additional for 'array' mode:
        - eta (--eta): Array of regularization parameters
            - sigma (--sigma): Array of kernel width parameters
            - splits (--splits): Number of k-fold cross-validation splits
            - print (--print): Verbosity level
            - ada (--ada): Adaptive sigma flag
            - name (--name): Output filename
    """
    def __init__(self, hyperparameters_set=None, **kwargs):
        super().__init__(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                **kwargs)

        parser = self
        parser.add_argument('--x',             type=str,   dest='repr',       required=True,                        help='path to the representations file')
        parser.add_argument('--y',             type=str,   dest='prop',       required=True,                        help='path to the properties file')
        if hyperparameters_set=='single':
            parser.add_argument('--eta',       type=float, dest='eta',        default=defaults.eta,                 help='eta hyperparameter')
            parser.add_argument('--sigma',     type=float, dest='sigma',      default=defaults.sigma,               help='sigma hyperparameter')
        elif hyperparameters_set=='array':
            parser.add_argument('--eta',       type=float, dest='eta',        default=defaults.etaarr,   nargs='+', help='eta array')
            parser.add_argument('--sigma',     type=float, dest='sigma',      default=defaults.sigmaarr, nargs='+', help='sigma array')
        parser.add_argument('--akernel',       type=str,   dest='akernel',    default=defaults.kernel,   choices=local_kernels_dict.keys(),
                            help='local kernel type: "G" for Gaussian, "L" for Laplacian, "dot" for dot products, "cosine" for cosine similarity. \
                                    "G_{sklearn,custom_c}", "L_{sklearn,custom_c,custom_py}" for specific implementations. \
                                    "L_custompy" is suited to open-shell systems')
        parser.add_argument('--gkernel',       type=str,   dest='gkernel',    default=defaults.gkernel,  choices=global_kernels_dict.keys(),
                            help='global kernel type: "avg" for average, "rem" for REMatch')
        parser.add_argument('--gdict', action=ParseKwargs, dest='gdict',      default=defaults.gdict,    nargs='*', help='dictionary like input string to initialize global kernel parameters')
        parser.add_argument('--test',          type=float,          dest='test_size',    default=defaults.test_size,             help='test set fraction')
        parser.add_argument('--train',         type=float,          dest='train_size',   default=defaults.train_size, nargs='+', help='training set fractions')
        parser.add_argument('--ll',            action='store_true', dest='ll',           default=False,                          help='if correct for the numper of threads')
        parser.add_argument('--readkernel',    action='store_true', dest='readk',        default=False,                          help='if X is kernel')
        parser.add_argument('--sparse',        type=int,            dest='sparse',       default=None,                           help='regression basis size for sparse learning')
        parser.add_argument('--random_state',  type=int,            dest='random_state', default=defaults.random_state,          help='seed for test / train splitting')
        if hyperparameters_set=='array':
            parser.add_argument('--splits',    type=int,            dest='splits',       default=defaults.splits,                help='k in k-fold cross validation')
            parser.add_argument('--print',     type=int,            dest='printlevel',   default=0,                              help='printlevel')
            parser.add_argument('--ada',       action='store_true', dest='adaptive',     default=False,                          help='if adapt sigma')
            parser.add_argument('--name',      type=str,            dest='nameout',      default=None,                           help='the name of the output file')
