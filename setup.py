from setuptools import setup, find_packages, Extension
import subprocess

VERSION="0.0.1"

def get_git_version_hash():
    """Get tag/hash of the latest commit.
    Thanks to https://gist.github.com/nloadholtes/07a1716a89b53119021592a1d2b56db8"""
    try:
        p = subprocess.Popen(["git", "describe", "--tags", "--dirty", "--always"], stdout=subprocess.PIPE)
    except EnvironmentError:
        return VERSION + "+unknown"
    version = p.communicate()[0]
    print(version)
    return VERSION+'+'+version.strip().decode()


setup(
    name='qstack',
    version=get_git_version_hash(),
    description='Stack of codes for dedicated pre- and post-processing tasks for Quantum Machine Learning',
    url='https://github.com/lcmd-epfl/Q-stack',
    install_requires=[],
    packages=find_packages(),
    ext_modules=[Extension('manh',
                           ['qstack/regression/lib/manh.c'],
                           extra_compile_args=['-fopenmp'],
                           extra_link_args=['-lgomp'])
                ],
)

