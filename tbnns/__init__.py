#-------------------------- init.py file for TBNN-s package ----------------------------#

__name__ = "tbnns"


from pkg_resources import get_distribution

def printInfo():
    """
    Makes sure everything is properly installed.
    
    We print a welcome message, and the version of the package. Return 1 at the end
    if no exceptions were raised.
    """
    
    print('Welcome to TBNN-s - Tensor Basis Neural Network for Scalar Mixing package!')
    
    # Get distribution version
    dist = get_distribution('tbnns')
    print('Version: {}'.format(dist.version))    
       
    return 1 # return this if everything went ok
