#----------------------------- momentum_losses.py file ---------------------------------#
"""
This file contains the definition of the loss functions that are employed in the TBNN
class, between predicted and truth anisotropy tensors
"""

import numpy as np
import tensorflow as tf

def lossL2(b, b_predicted, loss_weight=None, tf_flag=False):
    """
    One possible loss to determine how bad b_predicted is compared to b, the L2 loss
    
    Arguments:
    b -- np.array or tensor containing the true b vector, shape [None, 3, 3]
    b_predicted -- np.array or tensor containing the predicted b vector, 
                   shape [None, 3, 3]
    loss_weight -- np.array or tensor containing the elements that are used to
                   scale the losses in each cell. Since the anisotropy b is 
                   dimensionless, this parameter is optional. If not passed, not
                   scaling is done. If passed, it must be broadcastable
                   with b, i.e. it must have a shape such as [None, 1, 1] or [None, 3, 3]
    tf_flag -- optional argument, bool which says whether we are dealing with tensorflow
               tensors or with numpy arrays.
               
    Returns:
    loss_pred -- the prediction loss for the given b_predicted (either a number or a
                 tensor shape [] depending on tf_flag)    
    """
    
    # Optional: factor to multiply the loss and make it O(1)
    FACTOR = 1
    denominator = 1    
    
    # This array is used such that we don't double count the loss in the diagonals.
    # Since b is symmetric, we zero off-dimagonal elements below the diagonal
    # So we wouldn't double count u'v' and v'u' for example.
    upper_diagonal = np.ones(shape=(1,3,3))
    upper_diagonal[:,1,0]=0; upper_diagonal[:,2,0]=0; upper_diagonal[:,2,1]=0
    
    if tf_flag:
        loss = tf.math.squared_difference(b_predicted, b) * upper_diagonal
        if loss_weight is not None:
            loss = loss * loss_weight
            denominator = 1.0/6 * tf.reduce_mean(tf.reduce_sum(loss_weight*upper_diagonal,
                                                           axis=[1,2]))
        loss_pred = FACTOR * 1.0/6 * tf.reduce_mean(tf.reduce_sum(loss, axis=[1,2]))          
        
    else:
        loss = (b_predicted - b)**2 * upper_diagonal
        if loss_weight is not None:
            if len(loss_weight.shape) == 1: 
                loss_weight=np.reshape(loss_weight, (-1, 1, 1))
            loss = loss * loss_weight
            denominator = 1.0/6 * np.mean(np.sum(loss_weight*upper_diagonal, axis=(1,2)))
        loss_pred = FACTOR * 1.0/6 * np.mean(np.sum(loss, axis=(1,2)))        
    
    return loss_pred/denominator      

        
def lossL1(b, b_predicted, loss_weight=None, tf_flag=False):
    """
    One possible loss to determine how bad b_predicted is compared to b, the L1 loss
    
    Arguments:
    b -- np.array or tensor containing the true b vector, shape [None, 3, 3]
    b_predicted -- np.array or tensor containing the predicted b vector, 
                   shape [None, 3, 3]
    loss_weight -- np.array or tensor containing the elements that are used to
                   scale the losses in each cell. Since the anisotropy b is 
                   dimensionless, this parameter is optional. If not passed, not
                   scaling is done. If passed, it must be broadcastable
                   with b, i.e. it must have a shape such as [None, 1, 1] or [None, 3, 3]
    tf_flag -- optional argument, bool which says whether we are dealing with tensorflow
               tensors or with numpy arrays.
               
    Returns:
    loss_pred -- the prediction loss for the given b_predicted (either a number or a
                 tensor shape [] depending on tf_flag)    
    """
    
    # Optional: factor to multiply the loss and make it O(1)
    FACTOR = 1
    denominator = 1
    
    # This array is used such that we don't double count the loss in the diagonals.
    # Since b is symmetric, we zero off-dimagonal elements below the diagonal
    # So we wouldn't double count u'v' and v'u' for example.
    upper_diagonal = np.ones(shape=(1,3,3))
    upper_diagonal[:,1,0]=0; upper_diagonal[:,2,0]=0; upper_diagonal[:,2,1]=0
       
    if tf_flag:
        loss = tf.math.abs(b_predicted - b) * upper_diagonal
        if loss_weight is not None:
            loss = loss * loss_weight
            denominator = 1.0/6 * tf.reduce_mean(tf.reduce_sum(loss_weight*upper_diagonal,
                                                           axis=[1,2]))
        loss_pred = FACTOR * 1.0/6 * tf.reduce_mean(tf.reduce_sum(loss, axis=[1,2]))       
        
    else:
        loss = np.abs(b_predicted - b) * upper_diagonal
        if loss_weight is not None:
            if len(loss_weight.shape) == 1: 
                loss_weight=np.reshape(loss_weight, (-1, 1, 1))
            loss = loss * loss_weight
            denominator = 1.0/6 * np.mean(np.sum(loss_weight*upper_diagonal, axis=(1,2)))
        loss_pred = FACTOR * 1.0/6 * np.mean(np.sum(loss, axis=(1,2)))        
    
    return loss_pred/denominator  