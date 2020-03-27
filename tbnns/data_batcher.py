#------------------------------ data_batcher.py file -----------------------------------#
"""
This file contains classes that help in training/testing neural networks. It defines two
classes: BatchGenerator and Batch. The latter contains a mini-batch of data that is
presented to the neural network for training, while the former takes in the whole dataset
and returns instances of Batch.
"""

# ------------ Import statements
import numpy as np


class Batch(object):
    """
    This class consolidates all information needed in a single batch
    to train/test a TBNN-s.
    """
    
    def __init__(self, x_features, tensor_basis=None,  
                 uc=None, gradc=None, eddy_visc=None,
                 loss_weight=None, gamma_desired=None):
        """
        Constructor method, which takes in all the arrays and stores them.
        
        Arguments:
        x_features -- numpy array of shape (batch_size, n_features).
        tensor_basis -- numpy array of shape (batch_size, n_basis, 3, 3). Optional,
                        only for TBNN-s model
        uc -- numpy array of shape (batch_size, 3). Optional, only at training time.
        gradc -- numpy array of shape (batch_size, 3). Optional, only at training time.
        eddy_visc -- numpy array of shape (batch_size,). Optional, only at training time.
        loss_weight -- numpy array of shape (batch_size,1). Optional, only at training 
                       time and only for specific prediction losses.        
        gamma_desired -- numpy array of shape (batch_size, ). Optional, only at training
                         time for the model when 'enforce_prt'=True. This is a value of
                         gamma (1.0/Prt) that is predicted by a separate model and that
                         will be enforced by the NN at training time.
        """
        
        self.x_features = x_features
        self.tensor_basis = tensor_basis        
        self.uc = uc       
        self.gradc = gradc        
        self.eddy_visc = eddy_visc
        self.loss_weight = loss_weight
        self.gamma_desired = gamma_desired
                

class BatchGenerator(object):
    """
    This class is initialized with all training/test data available and automates
    the processes of creating min-batches to feed to the neural network.
    """
    
    def __init__(self, batch_size, x_features, tensor_basis=None, 
                 uc=None, gradc=None, eddy_visc=None,
                 loss_weight=None, gamma_desired=None):
        """
        Constructor method, which takes in batch size and full arrays.
        
        Arguments:
        batch_size -- int, which tells the class what is the batch size
        x_features -- numpy array of shape (n_total, n_features).
        tensor_basis -- numpy array of shape (n_total, n_basis, 3, 3). Optional,
                        only for TBNN-s model
        uc -- numpy array of shape (n_total, 3). Optional, only at training time.
        gradc -- numpy array of shape (num_total, 3). Optional, only at training time.
        eddy_visc -- numpy array of shape (num_total,). Optional, only at training time.
        loss_weight -- numpy array of shape (num_total, ) or (num_total, 1). Optional,
                       only needed at training time and for specific loss types. Note
                       that this function will expand_dims if needed to make it 
                       (num_total,1)        
        gamma_desired -- numpy array of shape (num_total, ). Optional, only at training
                         time for the model when 'enforce_prt'=True. This is a value of
                         gamma (1.0/Prt) that is predicted by a separate model and that
                         will be enforced by the NN at training time.
        """
        
        # Data that will be used in the batches
        self.x_features = x_features
        self.tensor_basis = tensor_basis
        self.uc = uc
        self.gradc = gradc
        self.eddy_visc = eddy_visc        
        self.gamma_desired = gamma_desired
                
        # Expand dims if necessary
        if loss_weight is not None:
            if len(loss_weight.shape) == 1:
                self.loss_weight = np.expand_dims(loss_weight,-1)
            else:
                self.loss_weight = loss_weight
        else:
            self.loss_weight = None            
        
        # Configurations
        self.n_total = x_features.shape[0] # total number of examples
        self.batch_size = batch_size
        self.batch_num = 0 # this contains the current batch number
                
        # Contains all the indices that must be used, in originally ordered
        self.idx_tot = np.arange(self.n_total)
        
        
    def reset(self):
        """
        Resets the generator: re-shuffles the indices and set batch_num to zero.
        Must be called before each epoch to randomize order at training.
        """
        
        np.random.shuffle(self.idx_tot)
        self.batch_num = 0
    
    def nextBatch(self):
        """
        This function is called to return the next batch of data.
        
        Returns:
        Batch() -- instance of Batch class containing the data for the next batch
        """
        
        # Here, there are no more examples left in that epoch, so return None
        if self.batch_num * self.batch_size >= self.n_total:
            return None
        
        # last batch: return everything left        
        if (self.batch_num+1) * self.batch_size >= self.n_total:
            idx = self.idx_tot[self.batch_num*self.batch_size:]

        # all other batches: returns the next num_batch elements
        else:
            idx = self.idx_tot[self.batch_num*self.batch_size:\
                              (self.batch_num+1)*self.batch_size]
        
        # count extra batch
        self.batch_num += 1
        
        # return according to appropriate indices
        x = self.x_features[idx,:]
        tb = self.tensor_basis[idx,:,:,:]
                
        uc = None; gradc = None; eddy_visc = None;
        loss_weight = None; gamma_desired = None
           
        if self.uc is not None:
            uc = self.uc[idx,:]
        if self.gradc is not None:
            gradc = self.gradc[idx,:]
        if self.eddy_visc is not None:
            eddy_visc = self.eddy_visc[idx]
        if self.loss_weight is not None:
            loss_weight = self.loss_weight[idx]        
        if self.gamma_desired is not None:
            gamma_desired = self.gamma_desired[idx]
        
        # Instantiate and return a Batch
        return Batch(x, tb, uc, gradc, eddy_visc, loss_weight, gamma_desired)