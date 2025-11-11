import argparse
from . import GlobalKernelsDict, LocalKernelsDict, defaults


class ParseKwargs(argparse.Action):
    def __call__(self, _parser, namespace, values, _option_string=None):
        setattr(namespace, self.dest, defaults.gdict)
        for value in values:
            key, value = value.split('=')
            for t in [int, float]:
                try:
                    value = t(value)
                    break
                except ValueError:
                    continue
            getattr(namespace, self.dest)[key] = value


class RegressionParser(argparse.ArgumentParser):
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
        parser.add_argument('--akernel',       type=str,   dest='akernel',    default=defaults.kernel,   choices=LocalKernelsDict().keys(),
                            help='local kernel type: "G" for Gaussian, "L" for Laplacian, "dot" for dot products, "cosine" for cosine similarity. \
                                    "G_{sklearn,custom_c}", "L_{sklearn,custom_c,custom_py}" for specific implementations. \
                                    "L_custompy" is suited to open-shell systems')
        parser.add_argument('--gkernel',       type=str,   dest='gkernel',    default=defaults.gkernel,  choices=GlobalKernelsDict().keys(),
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


    def remove_argument(parser, arg):
        for action in parser._actions:
            opts = action.option_strings
            if (opts and opts[0] == arg) or action.dest == arg:
                parser._remove_action(action)
                break

        for action in parser._action_groups:
            for group_action in action._group_actions:
                opts = group_action.option_strings
                if (opts and opts[0] == arg) or group_action.dest == arg:
                    action._group_actions.remove(group_action)
                    return
