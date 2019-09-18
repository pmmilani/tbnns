#--------------------------------- utils.py file ---------------------------------------#
"""
This file contains utility functions and classes that support the TBNN-s class.
This includes layers and cleaning functions.
"""

# ------------ Import statements
import os
import tensorflow as tf
import timeit
import numpy as np
from tbnns import constants

class FullyConnected(object):
    """    
    """

    def __init__(self, out_size, drop_prob, relu=True, name=""):
        """               
        """
           
        self.out_size = out_size
        self.drop_prob = drop_prob
        self.relu = relu
        self.name = name # this is used to create different variable contexts
        
    def build(self, layer_inputs):
        """
        Inputs:          

        Returns:          
        """
        with tf.variable_scope("FC_"+self.name):
        
            out = tf.contrib.layers.fully_connected(layer_inputs, self.out_size, 
                                                    activation_fn=None)
            if self.relu:                
                out = tf.nn.relu(out)
                out = tf.nn.dropout(out, rate=self.drop_prob) # Apply dropout

            return out


def downsampleIdx(n_total, downsample):
    """
    Produces a set of indices to index into a numpy array and shuffle/downsample it.
    
    Arguments:
    n_total -- int, total size of the array in that dimensionalization
    downsample -- number that controls how we downsample the data
                  before saving it to disk. If None, no downsampling or shuffling is 
                  done. If this number is more than 1, then it represents the number
                  of examples we want to save; if it is less than 1, it represents the
                  ratio of all training examples we want to save.    
                  
    Returns:
    idx -- numpy array of ints of size (n_take), which contains the indices
           that we are supposed to take for downsampling.    
    """
    
    idx_tot = np.arange(n_total)
    
    if downsample is None:
        return idx_tot
        
    np.random.shuffle(idx_tot)    
    assert downsample > 0, "downsample must be greater than 0!"    
    if int(downsample) > 1:
        n_take = int(downsample)            
        if n_take > n_total:
            print("Warning! This dataset has fewer than {} usable points. "
                  + "All of them will be taken.".format(n_take))
            n_take = n_total                 
    else:
        n_take = int(downsample * n_total)
        if n_take > n_total: n_take = n_total # catches bug where downsample = 1.1            
    
    idx = idx_tot[0:n_take]
    
    return idx 

            
def cleanDiffusivity(diff, g, test_inputs, n_std, prt_default, gamma_min, verbose=True):
    """
    This function is called to post-process the diffusivity and g produced for stability
    
    Arguments:
    diff -- numpy array of shape (n_useful, 3, 3) containing the originally predicted 
            diffusivity tensor
    g -- numpy array of shape (n_useful, NUM_BASIS) containing the factor that multiplies
         each tensor basis to produce the tensor diffusivity
    test_inputs -- numpy array of shape (n_useful, NUM_FEATURES) containing the test 
                   point features that produced the diffusivity tensor we are dealing
                   with.
    n_std -- float, number of standard deviations around the training mean that each
             test feature is allowed to be. If None, default value is read from 
             constants.py
    prt_default -- float, default value of turbulent Prandlt number used when cleaning
                   diffusivities that are outright negative. If None is passed, default
                   value is read from constants.py
    gamma_min -- float, minimum value of gamma = diffusivity/turbulent viscosity allowed.
                 Used to clean values that are positive but too close to zero. If None is
                 passed, default value is read from constants.py
    verbose -- optional argument, boolean that says whether we print some extra 
               information to the screen
    
    Returns:
    diff -- numpy array of shape (n_useful, 3, 3) containing the post-processed 
            diffusivity tensor
    g -- numpy array of shape (n_useful, NUM_BASIS) containing the post-processed
         factor that multiplies each tensor basis to produce the tensor diffusivity    
    """
        
    # Get default values if None is passed
    if n_std is None:
        n_std = constants.N_STD
    if gamma_min is None:
        gamma_min = constants.GAMMA_MIN
    
    print("Cleaning predicted diffusivity... ", end="", flush=True)
    tic = timeit.default_timer()        
    
    # (1) Here, make sure that there is no input more or less than n_std away
    # from the mean. Since test_inputs was normalized, it should not be more
    # than n_std or less than -n_std if we don't want test points too different
    # from the training data. We are cleaning points that, by a very simple measure,
    # denote extrapolation from the training set
    mask_ext = (np.amax(test_inputs,axis=1) > n_std)+(np.amin(test_inputs,axis=1) < -n_std)
    diff, g = applyMask(diff, g, mask_ext, prt_default)        
    num_ext = np.sum(mask_ext) # total number of entries affected by this step
    #--------------------------------------------------------------------       
    
    # (2) Here, make sure that all real part of eigenvalues have a minimum value.    
    # First, clear all negative ones; then add a complement to all the positive
    # but very small ones        
    eig_all, _ = np.linalg.eig(diff)
    t = np.amin(np.real(eig_all), axis=1) # minimum real part of eigenvalues       
    diff, g = applyMask(diff, g, t < 0, prt_default) # clean all the negative eigenvalues   
    
    # now, add a complement to increase the minimum eigenvalues beyond gamma_min
    complement = np.zeros_like(t)
    mask = (t>=0) * (t<=gamma_min)
    complement[mask] = gamma_min - t[mask]    
    assert (complement >= 0).all(), "Ops, that is not supposed to happen"
    diff[:,0,0] = diff[:,0,0] + complement
    diff[:,1,1] = diff[:,1,1] + complement
    diff[:,2,2] = diff[:,2,2] + complement
    g[:,0] = g[:,0] + complement    
    
    num_eig = np.sum(t <= gamma_min) # total number of entries affected by this step
    #--------------------------------------------------------------------     
    
    
    # (3) Here, make sure that no diffusivities have negative diagonals.
    # First, clean negative values
    t = np.amin([diff[:,0,0],diff[:,1,1],diff[:,2,2]],axis=0) # minimum diagonal entry
    diff, g = applyMask(diff, g, t < 0, prt_default)        
    
    # now, add a complement to increase the minimum diagonal beyond gamma_min
    complement = np.zeros_like(t)
    mask = (t>=0) * (t<=gamma_min)
    complement[mask] = gamma_min - t[mask]    
    assert (complement >= 0).all(), "Ops, that is not supposed to happen"
    diff[:,0,0] = diff[:,0,0] + complement
    diff[:,1,1] = diff[:,1,1] + complement
    diff[:,2,2] = diff[:,2,2] + complement
    g[:,0] = g[:,0] + complement 
    
    num_neg = np.sum(t <= gamma_min) # total number of entries affected by this step                 
    #--------------------------------------------------------------------        
    
    # Calculate minimum real part of eigenvalue and minimum diagonal entry 
    # after cleaning
    eig_all, _ = np.linalg.eig(diff)
    min_eig = np.amin(np.real(eig_all)) # minimum real part of any eigenvalue
    min_diag = np.amin(np.concatenate((diff[:,0,0], diff[:,1,1], diff[:,2,2])),axis=0)        
    
    # Print information        
    toc = timeit.default_timer()
    print("Done! It took {:.1f}s".format(toc-tic), flush=True)
    if verbose:            
        print("{:.2f}% of points were cleaned due to outlier inputs."\
            .format(100.0*num_ext/diff.shape[0]), flush=True)
        print("{:.2f}% of points were cleaned due to negative eigenvalues"\
            .format(100.0*num_eig/diff.shape[0]), flush=True)
        print("{:.2f}% of points were cleaned due to negative diagonal diffusivities:"\
            .format(100.0*num_neg/diff.shape[0]), flush=True)
        print("In cleaned diffusivity: min eigenvalue real part = {:g}".format(min_eig)
              + ", minimum diagonal entry = {:g}".format(min_diag), flush=True)   
              
    return diff, g

        
def applyMask(diff, g, mask, prt_default):
    """
    This simple function applies a mask to diff and g.
    
    Arguments:
    diff -- numpy array of shape (n_useful, 3, 3) containing the originally predicted 
            diffusivity tensor
    g -- numpy array of shape (n_useful, NUM_BASIS) containing the factor that multiplies
         each tensor basis to produce the tensor diffusivity
    mask -- boolean numpy array of shape (n_useful,) containing a mask which is True
            in the places we want to blank out
    prt_default -- float, default value of turbulent Prandlt number used when cleaning
                   diffusivities that are outright negative. If None is passed, default
                   value is read from constants.py
    
    Returns:
    diff -- numpy array of shape (n_useful, 3, 3) containing the post-processed 
            diffusivity tensor
    g -- numpy array of shape (n_useful, NUM_BASIS) containing the post-processed
         factor that multiplies each tensor basis to produce the tensor diffusivity    
    """

    if prt_default is None:
        prt_default = constants.PR_T
    
    diff[mask,:,:] = 0.0
    diff[mask,0,0] = 1.0/prt_default
    diff[mask,1,1] = 1.0/prt_default
    diff[mask,2,2] = 1.0/prt_default
    g[mask, :] = 0
    g[mask, 0] = 1.0/prt_default

    return diff, g
    
    
def suppressWarnings():
    """
    This function suppresses several warnings from Tensorflow.
    """
    
    import tensorflow.python.util.deprecation as deprecation
    deprecation._PRINT_DEPRECATION_WARNINGS = False
    if type(tf.contrib) != type(tf): tf.contrib._warning = None
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    os.environ['KMP_WARNINGS'] = 'off'
    