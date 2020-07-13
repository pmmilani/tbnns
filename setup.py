#------------------------- setup.py file for TBNN-s package ----------------------------#
# https://stackoverflow.com/questions/1471994/what-is-setup-py/23998536

# Note: this installs the requirements needed to the package, but this needs setuptools
# to be installed to work. Make sure it is installed; you might potentially need to run
# "pip install setuptools" or an equivalent command.
from setuptools import setup

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
   name='tbnns',
   version='0.5.0',
   description='TBNN-s - Tensor Basis Neural Network for Scalar Mixing',
   license='Apache',
   long_description=long_description,
   author='Pedro M. Milani',
   author_email='pmmilani@stanford.edu',
   url="https://github.com/pmmilani/tbnns.git",
   packages=['tbnns'],  # same as name
   install_requires=['tensorflow==1.15.2', 'joblib'], # dependencies
   python_requires='>=3.6',
   classifiers=[
        "Programming Language :: Python :: 3.7"
   ],
)
