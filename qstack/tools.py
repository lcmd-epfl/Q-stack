"""Utility functions and classes for Q-stack.

Provides decorators, argument parsers, and helper functions for command-line tools."""

import os
import time
import resource
import argparse


def unix_time_decorator(func):
    """Decorator to measure and print execution time statistics for a function.

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
    """Decorator to measure execution time statistics and return them along with function result.

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
        """Removes an argument from the parser.

        Utility method for customizing parsers by removing unwanted arguments
        from the pre-configured set. Useful when deriving specialized parsers.

        Args:
            arg (str): Option destination name.

        Returns:
            None: Modifies parser in place.
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
