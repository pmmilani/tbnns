#--------------------------------- utils.py file ---------------------------------------#
"""
This file contains utility functions and classes that support the TBNN-s class.
This includes layers and cleaning functions.
"""

# ------------ Import statements
import tensorflow as tf


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
            
            
def applyMask(diff, g, mask, PR_T):
    diff[mask,:,:] = 0
    diff[mask,0,0] = 1.0/PR_T
    diff[mask,1,1] = 1.0/PR_T
    diff[mask,2,2] = 1.0/PR_T
    g[mask, :] = 0;
    g[mask, 0] = 1.0/PR_T;

    return diff, g