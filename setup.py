from setuptools import setup, find_packages, Extension
import subprocess
import os, tempfile, shutil

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


def check_for_openmp():
    """Check if there is OpenMP available
    Thanks to https://stackoverflow.com/questions/16549893/programatically-testing-for-openmp-support-from-a-python-setup-script"""
    omp_test = 'void main() { }'
    tmpdir = tempfile.mkdtemp()
    curdir = os.getcwd()
    os.chdir(tmpdir)
    filename = r'test.c'
    with open(filename, 'w') as file:
        file.write(omp_test)
    with open(os.devnull, 'w') as fnull:
        result = subprocess.call(['cc', '-fopenmp', '-lgomp', filename], stdout=fnull, stderr=fnull)
    os.chdir(curdir)
    shutil.rmtree(tmpdir)
    return not result


def get_requirements():
    fname = f"{os.path.dirname(os.path.realpath(__file__))}/requirements.txt"
    with open(fname) as f:
        install_requires = f.read().splitlines()
    return install_requires


if __name__ == '__main__':
    openmp_enabled = check_for_openmp()

    setup(
        name='qstack',
        version=get_git_version_hash(),
        description='Stack of codes for dedicated pre- and post-processing tasks for Quantum Machine Learning',
        url='https://github.com/lcmd-epfl/Q-stack',
        python_requires='==3.9.*',
        install_requires=get_requirements(),
        packages=setuptools.find_packages(exclude=['tests', 'examples']),
        ext_modules=[Extension('qstack.regression.lib.manh',
                               ['qstack/regression/lib/manh.c'],
                               extra_compile_args=['-fopenmp', '-std=gnu11'] if openmp_enabled else ['-std=gnu11'],
                               extra_link_args=['-lgomp'] if openmp_enabled else [])
                    ],
        include_package_data=True,
        package_data={'': ['regression/lib/manh.c', 'spahm/rho/basis_opt/*.bas']},
    )
