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
                           extra_compile_args=['-fopenmp', '-std=gnu11'],
                           extra_link_args=['-lgomp'])
                ],
    include_package_data=True,
    package_data={'': ['spahm/rho/basis_opt/*.bas']},
)

