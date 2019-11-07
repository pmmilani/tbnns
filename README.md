## TBNN-s v0.3.0 - Tensor Basis Neural Network for Scalar Mixing

Author: Pedro M. Milani (email: pmmilani@stanford.edu)

Last modified: 11/06/2019

Developed and tested in Python 3.7

### Installation
To install, run the following (optionally within a virtual environment): 

    pip install tbnns [--user] [--upgrade]
    
This will install the stable version from the Python Package Index. Use
the flag --user in case you do not have administrator privileges and the
flag --upgrade to get the newest version.
    
To test the program while it is being developed, run the command below
from the current directory. This is useful when you are developing the
code.

    pip install -e .
    
To uninstall, run:
    
    pip uninstall tbnns
    
The commands above will also install
some dependencies (included in the file "requirements.txt")
needed for this package.

### Examples and Testing

The folder test contains a script example_usage.py and three representative
datasets. For an example of training a TBNN-s and applying it to a test
set, run the following inside the folder test:

    python example_usage.py