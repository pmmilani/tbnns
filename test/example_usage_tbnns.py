#--------------------------- example_usage_tbnns.py ------------------------------------#
"""
Quick script showing how to import and use the tbnns package
"""

from tbnns import printInfo
from tbnns.tbnns import TBNNS
import joblib
import numpy as np


def loadData():
    """
    This function loads the pre-processed data containing invariants (x),
    tensor basis (tb), turbulent scalar flux (u'c'), gradient of mean
    scalar concentration (gradc), and eddy viscosity (nut). We have three
    distinct flows and 10k points per flow. They are jets in crossflow
    with 3 distinct velocity ratios r described in the following paper:
    
    Milani et al. "Enriching MRI mean flow data of inclined jets in 
    crossflow with Large Eddy Simulations". Int. J. Heat Fluid Flow 2019.
    
    Note that the data contains mostly LES variables, but the values of k,
    epsilon and eddy viscosity employed to produce the data come from a 
    realizable k-epsilon simulation run with the LES velocity field in 
    the LES mesh. This is described in detail in my PhD thesis 
    (https://doi.org/10.13140/RG.2.2.27377.30566)  
    """

    print("Loading data from three different jet in crossflow simulations")
    print("The LES's are described in Milani et al. IJHFF 2019")
    print("This folder contains 10k points for each of the 3 flows.")
    
    print("Loading... ", end="", flush=True)    
    x_r1, tb_r1, uc_r1, gradc_r1, nut_r1 = joblib.load("tbnns_data_jicfr1_10k.pckl")    
    x_r1p5, tb_r1p5, uc_r1p5, gradc_r1p5, nut_r1p5 = joblib.load("tbnns_data_jicfr1p5_10k.pckl")        
    x_r2, tb_r2, uc_r2, gradc_r2, nut_r2 = joblib.load("tbnns_data_jicfr2_10k.pckl")
    print("Done!")
    
    print("Shape of features: ", x_r1.shape)
    print("Shape of tensor basis array: ", tb_r1.shape)
    print("Shape of u'c' array: ", uc_r1.shape)
    print("Shape of grad c array: ", gradc_r1.shape)
    print("Shape of eddy viscosity (nu_t) array: ", nut_r1.shape)
    
    return (x_r1, tb_r1, uc_r1, gradc_r1, nut_r1,
            x_r1p5, tb_r1p5, uc_r1p5, gradc_r1p5, nut_r1p5,
            x_r2, tb_r2, uc_r2, gradc_r2, nut_r2)    


def trainNetwork(x_train, tb_train, uc_train, gradc_train, nut_train,
                 x_dev, tb_dev, uc_dev, gradc_dev, nut_dev):
    """
    This function takes in training data and validation data (aka dev set)
    and runs the training routine. We initialize parameters of the TBNN-s 
    through the dictionary FLAGS and call nn.train() for training.
    """
        
    # Flags indicating parameters of the TBNN-s. This is a 
    # comprehensive list, with all flags that can be prescribed    
    FLAGS = {} # FLAGS is a dictionary with the following keys:
    
    FLAGS['num_features'] = 15 # number of features to be used
    FLAGS['num_basis'] = 6 # number of tensor basis in the expansion
    FLAGS['num_layers'] = 8 # number of hidden layers
    FLAGS['num_neurons'] = 30 # number of hidden units in each layer
    
    FLAGS['learning_rate'] = 1e-3 # learning rate for SGD algorithm
    FLAGS['num_epochs'] = 1000 # maximum number of epochs to run
    FLAGS['early_stop_dev'] = 20 # after this many evaluations without improvement in the dev set, stop training           
    FLAGS['eval_every'] = 100 # when to evaluate losses
    FLAGS['train_batch_size'] = 50 # number of points per batch
    
    FLAGS['loss_type'] = 'log' # loss type
    FLAGS['c_psd'] = 0.1 # constant for the J_psd loss function (see thesis section 6.3.3)
    FLAGS['c_prt'] = 0 # constant for the J_prt loss function (see thesis section 6.3.3)
    FLAGS['c_neg'] = 0 # constant for the J_neg loss function (see thesis section 6.3.3)
    FLAGS['c_reg'] = 5e-3 # L2 regularization strength   
    FLAGS['drop_prob'] = 0.2 # dropout probability at training time
    
    FLAGS['enforce_prt'] = False # whether to enforce given Pr_t (see thesis section 6.5)
     
    # Initialize TBNN-s with given FLAGS
    nn = TBNNS()
    nn.initializeGraph(FLAGS)   
    
    # Path to write TBNN-s metadata and parameters
    path_params = 'checkpoints/nn_test_tbnns.ckpt'
    path_class = 'nn_test_tbnns.pckl'
    
    # Train and save to disk
    nn.train(path_params,
             x_train, tb_train, uc_train, gradc_train, nut_train,
             x_dev, tb_dev, uc_dev, gradc_dev, nut_dev,             
             detailed_losses=True)
    nn.saveToDisk("Testing TBNN-s", path_class)
  

def applyNetwork(x_test, tb_test, uc_test, gradc_test, nut_test):
    """
    This function takes in test data and applies previously trained TBNN-s.    
    """   
    
    # ----- Load meta-data and initialize parameters
    # path_class should correspond to the path on disk of an existing trained network 
    path_class = 'nn_test_tbnns.pckl' 
    nn = TBNNS()
    nn.loadFromDisk(path_class, verbose=True)
    
    # Apply TBNN-s on the test set to get losses
    loss, loss_pred, _, _, loss_prt, _, _ = nn.getTotalLosses(x_test, tb_test, uc_test, gradc_test, nut_test)
    loss_pred_rans, loss_prt_rans, _ = nn.getRansLoss(uc_test, gradc_test, nut_test)   
    print("JICF r=1 Sc_t=0.85 losses: Prediction: {:g} | Prt: {:g}".format(loss_pred_rans, loss_prt_rans))
    print("JICF r=1 TBNN-s losses: Prediction: {:g} | Prt: {:g}".format(loss_pred, loss_prt))
    print("")
    
    # Now, apply TBNN-s on the test set to get a predicted diffusivity matrix
    diff, g = nn.getTotalDiffusivity(x_test, tb_test)    
    print("Predicted diffusivity shape: ", diff.shape)
    print("Predicted coefficients g_n shape: ", g.shape)    
    

def main():       
    
    printInfo() # simple function to print info about package
    
    # Load data to test TBNN-s
    (x_r1, tb_r1, uc_r1, gradc_r1, nut_r1,
     x_r1p5, tb_r1p5, uc_r1p5, gradc_r1p5, nut_r1p5,
     x_r2, tb_r2, uc_r2, gradc_r2, nut_r2) = loadData() 

    # train the TBNN-s on r=2 data (with r=1.5 as the dev set)
    print("")
    print("Training TBNN-s on baseline jet r=2 data...")
    trainNetwork(x_r2, tb_r2, uc_r2, gradc_r2, nut_r2,
                 x_r1p5, tb_r1p5, uc_r1p5, gradc_r1p5, nut_r1p5)  
    
    # Apply the trained network on the r=1 data
    print("")
    print("Applying trained TBNN-s on baseline jet r=1 data...")
    applyNetwork(x_r1, tb_r1, uc_r1, gradc_r1, nut_r1)
    
        
if __name__ == "__main__":
    main()