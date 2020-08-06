#---------------------------- example_usage_tbnn.py ------------------------------------#
"""
Quick script showing how to import and use the tbnns package
"""

from tbnns import printInfo
from tbnns.tbnn import TBNN
import joblib
import numpy as np


def loadData():
    """
    This function loads the pre-processed data containing invariants (x),
    tensor basis (tb), mean velocity gradient (gradu), Reynolds stress
    anisotropy (b), turbulent kinetic energy (tke), specific dissipationand
    rate (omega), eddy viscosity (nut), and computational cell volume (vol).
    We have three distinct flows and 10k points per flow. They are jets in
    crossflow with 3 distinct velocity ratios r described in the following paper:
    
    Milani et al. "Enriching MRI mean flow data of inclined jets in 
    crossflow with Large Eddy Simulations". Int. J. Heat Fluid Flow 2019.
    
    Note that the data contains mostly LES variables, but the values of nut 
    and omega employed to produce the data come from a 
    k-omega SST simulation run with the LES velocity field in 
    the LES mesh. This is described in detail in my PhD thesis 
    (https://doi.org/10.13140/RG.2.2.27377.30566)  
    """

    print("Loading data from three different jet in crossflow simulations")
    print("The LES's are described in Milani et al. IJHFF 2019")
    print("This folder contains 10k points for each of the 3 flows.")
    
    print("Loading... ", end="", flush=True)    
    (x_r1, tb_r1, gradu_r1, b_r1, tke_r1, omg_r1, 
     nut_r1, vol_r1) = joblib.load("tbnn_data_jicfr1_10k.pckl")    
    (x_r1p5, tb_r1p5, gradu_r1p5, b_r1p5, tke_r1p5, omg_r1p5,
     nut_r1p5, vol_r1p5) = joblib.load("tbnn_data_jicfr1p5_10k.pckl")        
    (x_r2, tb_r2, gradu_r2, b_r2, tke_r2, omg_r2,
     nut_r2, vol_r2) = joblib.load("tbnn_data_jicfr2_10k.pckl")
    print("Done!")
    
    print("Shape of features: ", x_r1.shape)
    print("Shape of tensor basis array: ", tb_r1.shape)
    print("Shape of gradu array: ", gradu_r1.shape)
    print("Shape of anisotropy (b) array: ", b_r1.shape)
    print("Shape of tke array: ", tke_r1.shape)
    print("Shape of omega array: ", omg_r1.shape)
    print("Shape of eddy viscosity (nu_t) array: ", nut_r1.shape)
    print("Shape of cell volume array: ", vol_r1.shape)
    
    return (x_r1, tb_r1, gradu_r1, b_r1, tke_r1, omg_r1, nut_r1, vol_r1,
            x_r1p5, tb_r1p5, gradu_r1p5, b_r1p5, tke_r1p5, omg_r1p5, nut_r1p5, vol_r1p5,
            x_r2, tb_r2, gradu_r2, b_r2, tke_r2, omg_r2, nut_r2, vol_r2)    


def trainNetwork(x_train, tb_train, b_train, x_dev, tb_dev, b_dev,
                 loss_weight_train=None, loss_weight_dev=None):
    """
    This function takes in training data and validation data (aka dev set)
    and runs the training routine. We initialize parameters of the TBNN-s 
    through the dictionary FLAGS and call nn.train() for training.
    """
        
    # Flags indicating parameters of the TBNN. This is a 
    # comprehensive list, with all flags that can be prescribed
    FLAGS = {} # FLAGS is a dictionary with the following keys:
    
    FLAGS['num_features'] = 7 # number of features to be used
    FLAGS['num_basis'] = 10 # number of tensor basis in the expansion
    FLAGS['num_layers'] = 8 # number of hidden layers
    FLAGS['num_neurons'] = 30 # number of hidden units in each layer
    
    FLAGS['learning_rate'] = 1e-3 # learning rate for SGD algorithm
    FLAGS['num_epochs'] = 1000 # maximum number of epochs to run
    FLAGS['early_stop_dev'] = 20 # after this many evaluations without improvement, stop training           
    FLAGS['eval_every'] = 100 # when to evaluate losses
    FLAGS['train_batch_size'] = 50 # number of points per batch
    
    FLAGS['loss_type'] = 'l2' # loss type    
    FLAGS['c_reg'] = 1e-7 # L2 regularization strength   
    FLAGS['drop_prob'] = 0 # dropout probability at training time   
     
    # Initialize TBNN with given FLAGS
    nn = TBNN()
    nn.initializeGraph(FLAGS)   
    
    # Path to write TBNN metadata and parameters
    path_params = 'checkpoints/nn_test_tbnn.ckpt'
    path_class = 'nn_test_tbnn.pckl'
    
    # Train and save to disk
    nn.train(path_params,
             x_train, tb_train, b_train, x_dev, tb_dev, b_dev,
             train_loss_weight=loss_weight_train, dev_loss_weight=loss_weight_dev,
             detailed_losses=True)
    nn.saveToDisk("Testing TBNN", path_class)
  

def applyNetwork(x_test, tb_test, b_test, gradu_test, nut_test, tke_test):
    """
    This function takes in test data and applies previously trained TBNN.    
    """    
    
    # ----- Load meta-data and initialize parameters
    # path_class should correspond to the path on disk of an existing trained network 
    path_class = 'nn_test_tbnn.pckl' 
    nn = TBNN()
    nn.loadFromDisk(path_class, verbose=True)
    
    # Apply TBNN on the test set to get losses
    loss, loss_pred, _, _ = nn.getTotalLosses(x_test, tb_test, b_test)
    loss_pred_rans = nn.getRansLoss(b_test, gradu_test, tke_test, nut_test)   
    print("JICF r=1 LEVM loss: {:g}".format(loss_pred_rans))
    print("JICF r=1 TBNN-s loss: {:g}".format(loss_pred))
    print("")
    
    # Now, apply TBNN on the test set to get a predicted anisotropy matrix
    # This can be used in a RANS solver to get improved velocity field predictions
    b_pred, g = nn.getTotalAnisotropy(x_test, tb_test)    
    print("Predicted anisotropy shape: ", b_pred.shape)
    print("Predicted coefficients g_n shape: ", g.shape)    
    

def main():       
    
    printInfo() # simple function to print info about package
    
    # Load data to test TBNN
    (x_r1, tb_r1, gradu_r1, b_r1, tke_r1, omg_r1, nut_r1, vol_r1,
     x_r1p5, tb_r1p5, gradu_r1p5, b_r1p5, tke_r1p5, omg_r1p5, nut_r1p5, vol_r1p5,
     x_r2, tb_r2, gradu_r2, b_r2, tke_r2, omg_r2, nut_r2, vol_r2) = loadData() 

    # train the TBNN on r=2 data (with r=1.5 as the dev set)
    print("")
    print("Training TBNN on baseline jet r=2 data...")
    trainNetwork(x_r2, tb_r2, b_r2, x_r1p5, tb_r1p5, b_r1p5)    
    #trainNetwork(x_r2, tb_r2, b_r2, x_r1p5, tb_r1p5, b_r1p5, vol_r2, vol_r1)
    # the last two arguments are optional; they consist of a weight that is applied
    # to the loss function. Uncomment to apply the computational cell volume as a weight
    
    # Apply the trained network on the r=1 data
    print("")
    print("Applying trained TBNN on baseline jet r=1 data...")
    applyNetwork(x_r1, tb_r1, b_r1, gradu_r1, nut_r1, tke_r1)
    
        
if __name__ == "__main__":
    main()