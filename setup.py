from setuptools import setup, Extension
import subprocess
import os
import tempfile
import shutil

# ruff: noqa: S607  # look, if people are installing qstack while using a borked PATH, that's on them
# ruff: noqa: D100  # this is a setup.py file, no docstring is needed

VERSION="0.0.1"

def get_git_version_hash():
    """Get tag/hash of the latest commit.

    Thanks to https://gist.github.com/nloadholtes/07a1716a89b53119021592a1d2b56db8
    """
    try:
        p = subprocess.Popen(["git", "describe", "--tags", "--dirty", "--always"], stdout=subprocess.PIPE)
    except OSError:
        return VERSION + "+unknown"
    version = p.communicate()[0]
    if not version.strip():
        return VERSION + "+unknown"
    print(version)
    return VERSION+'+'+version.strip().decode()


def check_for_openmp():
    """Check if there is OpenMP available.

    Thanks to https://stackoverflow.com/questions/16549893/programatically-testing-for-openmp-support-from-a-python-setup-script
    """
    omp_test = 'void main() { }'
    tmpdir = tempfile.mkdtemp()
    curdir = os.getcwd()
    os.chdir(tmpdir)
    filename = r'test.c'
    with open(filename, 'w') as file:
        file.write(omp_test)
    with open(os.devnull, 'w') as fnull:
        result = subprocess.call(['cc', '-fopenmp', '-lgomp', filename], stdout=fnull, stderr=fnull)  # noqa: S603  (filename is hard-coded earlier in this function)
    os.chdir(curdir)
    shutil.rmtree(tmpdir)
    return not result


if __name__ == '__main__':
    openmp_enabled = check_for_openmp()

    setup(
        version=get_git_version_hash(),
        ext_modules=[Extension('qstack.regression.lib.manh',
                               ['qstack/regression/lib/manh.c'],
                               extra_compile_args=['-fopenmp', '-std=gnu11'] if openmp_enabled else ['-std=gnu11'],
                               extra_link_args=['-lgomp'] if openmp_enabled else []),
                    ],
    )
