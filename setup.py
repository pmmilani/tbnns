#------------------------- setup.py file for RaFoFC package ----------------------------#
# https://stackoverflow.com/questions/1471994/what-is-setup-py/23998536

# Note: this installs the requirements needed to the package, but this needs setuptools
# to be installed to work. Make sure it is installed; you might potentially need to run
# "pip install setuptools" or an equivalent command.
from setuptools import setup

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
   name='tbnns',
   version='1.0.0',
   description='TBNN-s v1.0.0 - Tensor Basis Neural Network for Scalar Mixing',
   license='Apache',
   long_description=long_description,
   author='Pedro M. Milani',
   author_email='pmmilani@stanford.edu',
   packages=['tbnns'],  # same as name
   install_requires=['tensorflow>=1.13.1', 'joblib'], # dependencies
   include_package_data=True      
)
