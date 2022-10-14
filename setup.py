from setuptools import setup, find_packages, Extension

setup(
    name='qstack',
    version='0.0.0',
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

