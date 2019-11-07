#------------------------------ example_usage.py ---------------------------------------#
"""
Quick script showing how to import and use the tbnns package
"""

from tbnns import printInfo
from tbnns.tbnns import TBNNS
import joblib
import numpy as np


def trainNetwork():
    
    # ----- Load data from 3 different datasets (25k points from each)
    x_r1, tb_r1, uc_r1, gradc_r1, nut_r1, k_r1, eps_r1, u_r1, c_r1, Vi_r1, Xpos_r1 = joblib.load("br1_25k.pckl")    
    x_r1p5, tb_r1p5, uc_r1p5, gradc_r1p5, nut_r1p5, k_r1p5, eps_r1p5, u_r1p5, c_r1p5, Vi_r1p5, Xpos_r1p5 = joblib.load("br1p5_25k.pckl")        
    x_r2, tb_r2, uc_r2, gradc_r2, nut_r2, k_r2, eps_r2, u_r2, c_r2, Vi_r2, Xpos_r2 = joblib.load("br2_25k.pckl")
    
    
    # ----- Construct TBNN-s with specific flags
    FLAGS = {}
    FLAGS['num_epochs'] = 50
    FLAGS['early_stop_dev'] = 10    
    FLAGS['num_layers'] = 10 #Number of hidden layers
    FLAGS['num_neurons'] = 30 #Number of hidden units in each layer   
    FLAGS['drop_prob'] = 0.2 # Dropout probability at training time    
    FLAGS['learning_rate'] = 1e-3 # Initial learning rate    
    FLAGS['loss_type'] = 'log'
    FLAGS['neg_factor'] = 0.1
    FLAGS['reg_factor'] = 1e-2 # L2 regularization strength 
    FLAGS['gamma_factor'] = 0
    FLAGS['reduce_diff'] = True # whether to make gamma_loss penalize diffusivity matrix in counter-gradient diffusion regions    
    FLAGS['eval_every'] = 250 # When to evaluate losses
    FLAGS['train_batch_size'] = 50    
    nn = TBNNS()
    nn.initializeGraph(FLAGS)
    
        
    # ----- Calculate and print initial error
    x_r1_norm = (x_r1 - np.mean(x_r1, axis=0, keepdims=True))/np.std(x_r1, axis=0, keepdims=True)
    x_r1p5_norm = (x_r1p5 - np.mean(x_r1p5, axis=0, keepdims=True))/np.std(x_r1p5, axis=0, keepdims=True)
    x_r2_norm = (x_r2 - np.mean(x_r2, axis=0, keepdims=True))/np.std(x_r2, axis=0, keepdims=True)    
    loss, loss_pred, _, _, loss_gamma, _ = nn.getTotalLosses(x_r1_norm, tb_r1, uc_r1, gradc_r1, nut_r1)
    loss_pred_rans, loss_gamma_rans = nn.getRansLoss(uc_r1, gradc_r1, nut_r1)
    print("(BR=1) Prediction losses RANS: {:g} | Initial: {:g}".format(loss_pred_rans, loss_pred))
    print("(BR=1) Gamma losses RANS: {:g} | Initial: {:g}".format(loss_gamma_rans, loss_gamma))    
    loss, loss_pred, _, _, loss_gamma, _ = nn.getTotalLosses(x_r1p5_norm, tb_r1p5, uc_r1p5, gradc_r1p5, nut_r1p5)
    loss_pred_rans, loss_gamma_rans = nn.getRansLoss(uc_r1p5, gradc_r1p5, nut_r1p5)
    print("(BR=1p5) Prediction losses RANS: {:g} | Initial: {:g}".format(loss_pred_rans, loss_pred))
    print("(BR=1p5) Gamma losses RANS: {:g} | Initial: {:g}".format(loss_gamma_rans, loss_gamma))    
    loss, loss_pred, _, _, loss_gamma, _ = nn.getTotalLosses(x_r2_norm, tb_r2, uc_r2, gradc_r2, nut_r2)
    loss_pred_rans, loss_gamma_rans = nn.getRansLoss(uc_r2, gradc_r2, nut_r2)
    print("(BR=2) Prediction losses RANS: {:g} | Initial: {:g}".format(loss_pred_rans, loss_pred)) 
    print("(BR=2) Gamma losses RANS: {:g} | Initial: {:g}".format(loss_gamma_rans, loss_gamma))

    
    # ----- Train the network with BR=2 and BR=1.5 as dev set
    CASE = "test"
    
    path_params = 'checkpoints/nn_{}.ckpt'.format(CASE)
    path_class = 'nn_{}.pckl'.format(CASE)
    
    nn.train(path_params,
             x_r2, tb_r2, uc_r2, gradc_r2, nut_r2,
             x_r1p5, tb_r1p5, uc_r1p5, gradc_r1p5, nut_r1p5,             
             detailed_losses=True)
    nn.saveToDisk("Testing TBNN-s: {}".format(CASE), path_class)
  

def applyNetwork():
    
    # ----- Load data from 3 different datasets (25k points from each)
    x_r1, tb_r1, uc_r1, gradc_r1, nut_r1, k_r1, eps_r1, u_r1, c_r1, Vi_r1, Xpos_r1 = joblib.load("br1_25k.pckl")    
    x_r1p5, tb_r1p5, uc_r1p5, gradc_r1p5, nut_r1p5, k_r1p5, eps_r1p5, u_r1p5, c_r1p5, Vi_r1p5, Xpos_r1p5 = joblib.load("br1p5_25k.pckl")        
    x_r2, tb_r2, uc_r2, gradc_r2, nut_r2, k_r2, eps_r2, u_r2, c_r2, Vi_r2, Xpos_r2 = joblib.load("br2_25k.pckl")
    
    
    # ----- Load meta-data and initialize parameters
    CASE = "test"  
    path_class = 'nn_{}.pckl'.format(CASE)   
    nn = TBNNS()
    nn.loadFromDisk(path_class, verbose=True)  

    
    # ----- Apply it on the 3 datasets and print errors
    loss, loss_pred, _, _, loss_gamma, _ = nn.getTotalLosses(x_r1, tb_r1, uc_r1, gradc_r1, nut_r1)
    loss_pred_rans, loss_gamma_rans = nn.getRansLoss(uc_r1, gradc_r1, nut_r1)
    print("BR=1 RANS losses: Prediction: {:g} | Gamma: {:g}".format(loss_pred_rans, loss_gamma_rans))
    print("BR=1 TBNN-s losses: Prediction: {:g} | Gamma: {:g}".format(loss_pred, loss_gamma))    
    diff, g = nn.getTotalDiffusivity(x_r1, tb_r1)
    print("")
    
    loss, loss_pred, _, _, loss_gamma, _ = nn.getTotalLosses(x_r1p5, tb_r1p5, uc_r1p5, gradc_r1p5, nut_r1p5)
    loss_pred_rans, loss_gamma_rans = nn.getRansLoss(uc_r1p5, gradc_r1p5, nut_r1p5)
    print("BR=1.5 RANS losses: Prediction: {:g} | Gamma: {:g}".format(loss_pred_rans, loss_gamma_rans))
    print("BR=1.5 TBNN-s losses: Prediction: {:g} | Gamma: {:g}".format(loss_pred, loss_gamma))    
    diff, g = nn.getTotalDiffusivity(x_r1p5, tb_r1p5)
    print("")    
    
    loss, loss_pred, _, _, loss_gamma, _ = nn.getTotalLosses(x_r2, tb_r2, uc_r2, gradc_r2, nut_r2)
    loss_pred_rans, loss_gamma_rans = nn.getRansLoss(uc_r2, gradc_r2, nut_r2)
    print("BR=2 RANS losses: Prediction: {:g} | Gamma: {:g}".format(loss_pred_rans, loss_gamma_rans))
    print("BR=2 TBNN-s losses: Prediction: {:g} | Gamma: {:g}".format(loss_pred, loss_gamma))    
    diff, g = nn.getTotalDiffusivity(x_r2, tb_r2)
    

def main():   
    
    printInfo() # simple function to print info about package   
    trainNetwork() # shows how to train the network and save parameters to disk   
    applyNetwork() # loads previously trained network and applies it to data   
    
        
if __name__ == "__main__":
    main()