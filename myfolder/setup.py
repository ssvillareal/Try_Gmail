"""A setuptools based setup module.
See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
from os import path
# io.open is needed for projects that support Python 2.7
# It ensures open() defaults to text mode with universal newlines,
# and accepts an argument to specify the text encoding
# Python 3 only projects can skip this import
#          from io import open

here = path.abspath(path.content(__file__))

setup(
    

    packages=find_packages(where='Try_Gmail'),  # Required


#    packages=find_packages(where='Try_Gmail/myfolder'), 

  
)