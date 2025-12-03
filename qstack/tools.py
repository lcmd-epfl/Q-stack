"""Utility functions and classes for Q-stack.

Provides decorators, argument parsers, and helper functions for command-line tools.
"""

import os
import sys
import time
import resource
import argparse
import numpy as np
from itertools import accumulate
if sys.version_info[1]>=10:
    from itertools import pairwise
else:
    def pairwise(iterable):
        """Implement itertools.pairwise for python<3.10.

        Taken from https://docs.python.org/3/library/itertools.html#itertools.pairwise
        """
        iterator = iter(iterable)
        a = next(iterator, None)
        for b in iterator:
            yield a, b
            a = b


def unix_time_decorator(func):
    """Measure and print execution time statistics for a function.

    Measures real, user, and system time for the decorated function.
    Thanks to https://gist.github.com/turicas/5278558

    Args:
        func (callable): Function to be decorated.

    Returns:
        callable: Wrapped function that prints timing information.
    """
    def wrapper(*args, **kwargs):
        start_time, start_resources = time.time(), resource.getrusage(resource.RUSAGE_SELF)
        ret = func(*args, **kwargs)
        end_resources, end_time = resource.getrusage(resource.RUSAGE_SELF), time.time()
        real = end_time - start_time
        user = end_resources.ru_utime - start_resources.ru_utime
        syst = end_resources.ru_stime - start_resources.ru_stime
        print(f'{func.__name__} :  real: {real:.4f}  user: {user:.4f}  sys: {syst:.4f}')
        return ret
    return wrapper


def unix_time_decorator_with_tvalues(func):
    """Measure execution time statistics and return them along with function result.

    Measures real, user, and system time for the decorated function and returns timing dict.
    Thanks to https://gist.github.com/turicas/5278558

    Args:
        func (callable): Function to be decorated.

    Returns:
        callable: Wrapped function that returns (timing_dict, result).
    """
    def wrapper(*args, **kwargs):
        start_time, start_resources = time.time(), resource.getrusage(resource.RUSAGE_SELF)
        ret = func(*args, **kwargs)
        end_resources, end_time = resource.getrusage(resource.RUSAGE_SELF), time.time()
        timing = {'real' : end_time - start_time,
                  'user' : end_resources.ru_utime - start_resources.ru_utime,
                  'sys' : end_resources.ru_stime - start_resources.ru_stime}
        return timing, ret
    return wrapper


def correct_num_threads():
    """Set MKL and OpenBLAS thread counts based on SLURM environment.

    If running under SLURM, sets MKL_NUM_THREADS and OPENBLAS_NUM_THREADS
    to match SLURM_CPUS_PER_TASK.
    """
    if "SLURM_CPUS_PER_TASK" in os.environ:
        os.environ["MKL_NUM_THREADS"] = os.environ["SLURM_CPUS_PER_TASK"]
        os.environ["OPENBLAS_NUM_THREADS"] = os.environ["SLURM_CPUS_PER_TASK"]


class FlexParser(argparse.ArgumentParser):
    """Argument parser that allows removing arguments.

    Args:
        **kwargs: Arguments passed to ArgumentParser.

    """
    def remove_argument(self, arg):
        """Remove an argument from the parser.

        Utility method for customizing parsers by removing unwanted arguments
        from the pre-configured set. Useful when deriving specialized parsers.

        Args:
            arg (str): Option destination name.

        Output:
            Modifies parser in place.
        """
        for action in self._actions:
            opts = action.option_strings
            if (opts and opts[0] == arg) or action.dest == arg:
                self._remove_action(action)
                break

        for action in self._action_groups:
            for group_action in action._group_actions:
                opts = group_action.option_strings
                if (opts and opts[0] == arg) or group_action.dest == arg:
                    action._group_actions.remove(group_action)
                    return


def slice_generator(iterable, inc=lambda x: x, i0=0):
    """Generate slices for elements in an iterable based on increments.

    Args:
        iterable (iterable): Iterable of elements to generate slices for.
        inc (callable: int->int): Function that computes increment size for each element.
                                  Defaults to identity function.
        i0 (int): Initial starting index. Defaults to 0.

    Yields:
        tuple: (element, slice) pairs for each element in the iterable.
    """
    func = func=lambda total, elem: total+inc(elem)
    starts = accumulate(iterable, func=func, initial=i0)
    starts_ends = pairwise(starts)
    for elem, (start, end) in zip(iterable, starts_ends, strict=True):
        yield elem, np.s_[start:end]


class Cursor:
    """Cursor class to manage dynamic indexing.

    Args:
        action (str): Type of indexing action ('slicer' or 'ranger').
        inc (callable: int->int): Function to determine increment size.
                                  Defaults to identity function.
        i0 (int): Initial index position. Defaults to 0.

    Attributes:
        i (int): Current index position.
        i_prev (int): Previous index position.
        cur (range or slice: Current range or slice.
        inc (callable int->int): Increment function.

    Methods:
        add(di): Advances the cursor by increment and returns current range/slice.
        __call__(di=None): Advances the cursor if di is not None,
            returns current range/slice.
    """
    def __init__(self, action='slicer', inc=lambda x: x, i0=0):
        self.i = i0
        self.i_prev = None
        self.inc = inc
        self.cur = None
        self.action = action
        self.actions_dict = {'slicer': self._slicer, 'ranger': self._ranger}

    def add(self, di):
        """Advance the cursor and return the current range or slice.

        Args:
            di: Element to determine increment size.

        Returns:
            Current range or slice after advancing.
        """
        self._add(di)
        self.cur = self.actions_dict[self.action]()
        return self.cur

    def _ranger(self):
        return range(self.i_prev, self.i)

    def _slicer(self):
        return np.s_[self.i_prev:self.i]

    def _add(self, di):
        self.i_prev = self.i
        self.i += self.inc(di)

    def __call__(self, di=None):
        """Optionally advance the cursor and return the current range or slice.

        If the argument is passed, it is used to advance the cursor.
        If not, the current value is returned.

        Args:
            di (optional): Element to determine increment size.

        Returns:
            Current range or slice (after advancing).
        """
        if di is None:
            return self.cur
        else:
            return self.add(di)

    def __str__(self):
        return str(self.i)

    def __repr__(self):
        return str(self.i)
