#--------------------------------- utils.py file ---------------------------------------#
"""
This file contains utility functions and classes that support the TBNN-s class.
This includes layers and cleaning functions.
"""

# ------------ Import statements
import os
import tensorflow as tf
import time
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

            
def cleanDiffusivity(diff, g, test_inputs, n_std):
        
        if n_std is None:
            n_std = constants.N_STD
        
        print("Cleaning predicted diffusivity... ", end="", flush=True)
        tic = time.time()
        
        # Here, make sure that there is no input more or less than n_std away
        # from the mean. The mean and standard deviation used are from the 
        # training data (which are set to 0 and 1 respectively)
        mask_extr = (np.amax(test_inputs, axis=1) > n_std) + (np.amin(test_inputs, axis=1) < -n_std)
        num_extr = np.sum(mask_extr) 
        diff, g = applyMask(diff, g, mask_extr)         
        
        # Here, make sure that no eigenvalues have a negative real part. 
        # If they did, an unstable model is produced.
        eig_all, _ = np.linalg.eig(diff)
        t = np.amin(np.real(eig_all), axis=1) # minimum real part of eigenvalues
        mask_eig = t < 0
        num_eig = np.sum(mask_eig)
        avg_negative_eig = 0
        if num_eig > 0:
            avg_negative_eig = np.mean(t[mask_eig])        
        diff, g = applyMask(diff, g, mask_eig)
        
        # Here, make sure that no diffusivities have negative diagonals 
        # If they did, an unstable model is produced.
        mask_neg = (diff[:,0,0] < 0) + (diff[:,1,1] < 0) + (diff[:,2,2] < 0)
        num_neg = np.sum(mask_neg)        
        avg_neg_diff_x = 0
        num_neg_diff_x = np.sum(diff[:,0,0]<0)
        if num_neg_diff_x > 0:
            avg_neg_diff_x = np.mean(diff[diff[:,0,0]<0,0,0])
        avg_neg_diff_y = 0
        num_neg_diff_y = np.sum(diff[:,1,1]<0)
        if num_neg_diff_y > 0:
            avg_neg_diff_y = np.mean(diff[diff[:,1,1]<0,1,1])
        avg_neg_diff_z = 0
        num_neg_diff_z = np.sum(diff[:,2,2]<0)
        if num_neg_diff_z > 0:
            avg_neg_diff_z = np.mean(diff[diff[:,2,2]<0,2,2])                  
        diff, g = applyMask(diff, g, mask_neg)        
        
        # Calculate minimum real part of eigenvalue and minimum diagonal entry 
        # after cleaning
        eig_all, _ = np.linalg.eig(diff)
        min_eig = np.amin(np.real(eig_all)) # minimum real part of any eigenvalue
        min_diag = np.amin(np.concatenate((diff[:,0,0], diff[:,1,1], diff[:,2,2])))        
        
        # Print information        
        toc = time.time()
        print("Done! It took {:.1f}s".format(toc-tic), flush=True)
        print("{:.3f}% of points were cleaned due to outlier inputs."\
            .format(100.0*num_extr/diff.shape[0]), flush=True)
        print("{:.3f}% of points were cleaned due to negative eigenvalues (avg: {:.2f})."\
            .format(100.0*num_eig/diff.shape[0], avg_negative_eig), flush=True)
        print("{:.3f}% of points were cleaned due to negative diagonal diffusivities:".format(100.0*num_neg/diff.shape[0]), flush=True)
        print("\t x: {:.3f}% (avg={:.2f}), y: {:.3f}% (avg={:.2f}), z: {:.3f}% (avg={:.2f})"\
            .format(100.0*num_neg_diff_x/diff.shape[0], avg_neg_diff_x,
                    100.0*num_neg_diff_y/diff.shape[0], avg_neg_diff_y,
                    100.0*num_neg_diff_z/diff.shape[0], avg_neg_diff_z), flush=True)
        print("In cleaned diffusivity: minimum eigenvalue real part = {:g}, minimum diagonal entry = {:g}"\
            .format(min_eig, min_diag), flush=True)
        
        return diff, g

        
def applyMask(diff, g, mask):
    diff[mask,:,:] = 0
    diff[mask,0,0] = 1.0/constants.PR_T
    diff[mask,1,1] = 1.0/constants.PR_T
    diff[mask,2,2] = 1.0/constants.PR_T
    g[mask, :] = 0;
    g[mask, 0] = 1.0/constants.PR_T;

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
    