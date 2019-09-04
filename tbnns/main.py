#--------------------------------- main.py file ---------------------------------------#
"""
This file contains the definition of the TBNN-s class, which is implemented using 
tensorflow. This is a Tensor Basis Neural Network that is meant to predict a tensorial
diffusivity for turbulent mixing applications.
"""

# ------------ Import statements
import numpy as np
import os
import joblib
import time
import tensorflow as tf
from tbnns.data_batcher import Batch, BatchGenerator
from tbnns import constants
from tbnns import utils


class TBNN_S(object):
    """
    This class contains definitions and methods needed for the TBNN_S class.
    """
    
    def __init__(self, FLAGS, saver_path=None, features_mean=None, features_std=None):
        """
        Constructor class that initializes the model.
        
        Arguments:
        FLAGS -- dictionary containing different parameters for the network
        saver_path -- string, containing the (relative) path in which the model
                      parameters should be saved.
        features_mean -- mean of the training features, which is used to normalize the
                         features at training time.
        features_std -- standard deviation of the training features, which is used to 
                        normalize the features at training time.
        """
        
        print("Initializing TBNN-s Class...", end="", flush=True)
                   
        self.FLAGS = FLAGS        
        self.features_mean = features_mean
        self.features_std = features_std
        self.saver_path = saver_path
        
        self.buildGraph()   
                    
        print(" Done!", flush=True)
        
    
    def buildGraph(self):
        """
        This method builds the tensorflow graph of the network.
        """   
      
        # Add all parts of the graph
        tf.reset_default_graph()
        with tf.variable_scope("Model"):
            self.addPlaceholders()
            self.constructNN()
            self.combineBases()
            self.addLoss()        
        
        # Define optimizer and updates
        # (updates is what you need to fetch in session.run to do a gradient update)
        self.global_step = tf.Variable(0, name="global_step", trainable=False)                
        opt = tf.train.AdamOptimizer(learning_rate=self.FLAGS['learning_rate'])
        self.updates = opt.minimize(self.loss, global_step=self.global_step)
        
        # Define savers (for checkpointing)
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
        
    
    def addPlaceholders(self):
        """
        Adds all placeholders for external data into the model

        Defines:
        self.x_features -- placeholder for features (i.e. inputs to NN)
        self.tensor_basis -- placeholder for tensor basis
        self.uc -- placeholder for the labels u'c'
        self.eddy_visc -- placeholder for eddy viscosity
        self.drop_prob -- placeholder for dropout probability        
        """
        
        self.x_features = tf.placeholder(tf.float32, 
                                     shape=[None, self.FLAGS['num_features']])
        self.tensor_basis = tf.placeholder(tf.float32, 
                                          shape=[None, self.FLAGS['num_bases'], 3, 3])
        self.uc = tf.placeholder(tf.float32, shape=[None, 3])
        self.gradc = tf.placeholder(tf.float32, shape=[None, 3])
        self.eddy_visc = tf.placeholder(tf.float32, shape=[None])                
        self.drop_prob = tf.placeholder_with_default(0.0, shape=())
                    
    
    def constructNN(self):    
        """
        Creates the neural network section of the model, with fully connected layers

        Uses:          

        Defines:
          self.g: the coefficients for each of the form invariant basis
        """
        
        # Creates the first hidden state from the inputs
        fc1 = FullyConnected(self.FLAGS['num_neurons'], self.drop_prob, name="1")
        hd1 = fc1.build(self.x_features)
        
        hd_list = [hd1, ] # list of all hidden states
        
        # Creates all other hidden states
        for i in range(self.FLAGS['num_layers']-1):
            fc = FullyConnected(self.FLAGS['num_neurons'], self.drop_prob, name=str(i+2))
            hd_list.append(fc.build(hd_list[-1]))
        
        # Go from last hidden state to the outputs (g in this case)
        fc_last = FullyConnected(self.FLAGS['num_bases'], self.drop_prob, relu=False, name="last")
        self.g = fc_last.build(hd_list[-1])

    
    def combineBases(self):
        
        with tf.variable_scope("bases"):
        
            # shape of [None,num_bases,3,3]    
            mult_bas = tf.multiply(tf.reshape(self.g, shape=[-1,self.FLAGS['num_bases'],1,1]), 
                                   self.tensor_basis) 
            
            # shape of [None,3,3]            
            self.diffusivity = tf.reduce_sum(mult_bas, axis=1)
            
            # shape of [None,3,1]
            gradc = tf.expand_dims(self.gradc, -1)

            # shape of [None, 3]
            self.uc_predicted = -1.0 * tf.expand_dims(self.eddy_visc,-1) * tf.squeeze(tf.matmul(self.diffusivity, gradc))
        

    def addLoss(self):
        """
        Add loss computation to the graph.

        Uses:          

        Defines:
          self.loss_pred, self.loss_l2, self.loss: all scalars with the loss in this batch
        """
        
        with tf.variable_scope("loss"):
            
            # Average prediction loss across batch
            loss_prediction = tf.norm(self.uc - self.uc_predicted, ord=2, axis=1)/tf.norm(self.uc, ord=2, axis=1)
            self.loss_pred = tf.reduce_mean(tf.log(loss_prediction))               
            
            # Calculate the L2 regularization component of the loss. Don't regularize bias
            vars = tf.trainable_variables()
            self.loss_l2 = self.FLAGS['l2_reg'] *\
                           tf.add_n([tf.nn.l2_loss(v) for v in vars if ('bias' not in v.name)])             

            self.loss = self.loss_pred + self.loss_l2  
            

    def runTrainIter(self, session, batch):
        """
        This performs a single training iteration (forward pass, loss computation, backprop, 
        parameter update)

        Inputs:
          session: TensorFlow session
          batch: a Batch object          

        Returns:
          loss: The loss (averaged across the batch) for this batch.
          global_step: The current number of training iterations we've done
          gradient_norm: Global norm of the gradients
        """
        # Match up our input data with the placeholders
        input_feed = {}
        input_feed[self.x_features] = batch.x_features
        input_feed[self.uc] = batch.uc
        input_feed[self.gradc] = batch.gradc
        input_feed[self.tensor_basis] = batch.tensor_basis
        input_feed[self.eddy_visc] = batch.eddy_visc
        input_feed[self.drop_prob] = self.FLAGS['drop_prob'] # apply dropout
                            
        # output_feed contains the things we want to fetch.
        output_feed = [self.updates, self.loss, self.global_step]

        # Run the model
        [_, loss, global_step] = session.run(output_feed, input_feed)

        return loss, global_step
        
    
    def getLoss(self, session, batch):
        """
        This runs a single forward pass and obtains the loss
        """
        
        input_feed = {}
        input_feed[self.x_features] = batch.x_features
        input_feed[self.uc] = batch.uc
        input_feed[self.gradc] = batch.gradc
        input_feed[self.tensor_basis] = batch.tensor_basis
        input_feed[self.eddy_visc] = batch.eddy_visc
                                
        # output_feed contains the things we want to fetch.
        output_feed = [self.loss, self.loss_pred]
        
        # Run the model
        [loss, loss_pred] = session.run(output_feed, input_feed)

        return loss, loss_pred
        
        
    def getDiffusivity(self, session, batch):
        """
        This runs a single forward pass to obtain the (dimensionless) diffusivity matrix
        """
        
        input_feed = {}
        input_feed[self.x_features] = batch.x_features        
        input_feed[self.tensor_basis] = batch.tensor_basis
                        
        # output_feed contains the things we want to fetch.
        output_feed = [self.diffusivity, self.g]
        
        # Run the model
        [diff, g] = session.run(output_feed, input_feed)
        
        return diff, g
    
    
    def getTotalLoss(self, session, x_features, tensor_basis, uc, gradc,
                     eddy_visc, update_stats=True, subsample=None):
        """
        This method takes in a whole dataset and computes the average loss on it.
        """
    
        tic = time.time()
        
        # This is a constant, which determines how big our batches are.
        # Since we go through all the data exactly one and compute the loss,
        # this parameter shouldn't change the result.
        DEV_BATCH_SIZE = 10000
        
        # Initializes quantities we need to keep track of
        total_loss = 0
        total_loss_pred = 0
        N = x_features.shape[0]
        
        # This normalizes the inputs. Runs when update_stats = True
        if self.features_mean is not None and update_stats:
            x_features = (x_features - self.features_mean)/self.features_std        
        
        # Initialize batch generator (either by subsampling or not)
        batch_gen = None
        if subsample is not None:
            N = int(N*subsample)
            idx_tot = np.arange(N)
            np.random.shuffle(idx_tot)
            idx = idx_tot[0:N]
            batch_gen = BatchGenerator(DEV_BATCH_SIZE, x_features[idx,:], 
                                       tensor_basis[idx,:,:,:], 
                                       uc[idx,:], gradc[idx,:],
                                       eddy_visc[idx])
        else:
            batch_gen = BatchGenerator(DEV_BATCH_SIZE, x_features, tensor_basis, 
                                       uc, gradc, eddy_visc)       
        
        # Iterate through all batches of data
        batch = batch_gen.nextBatch()        
        while batch is not None:
            this_loss, this_loss_pred = self.getLoss(session, batch)
            total_loss += this_loss * batch.x_features.shape[0]
            total_loss_pred += this_loss_pred * batch.x_features.shape[0]  
            batch = batch_gen.nextBatch()
        
        # To get an average loss, divide by total number of dev points
        total_loss =  total_loss/N
        total_loss_pred = total_loss_pred/N
        
        toc = time.time()
        #print("Computed loss over {} examples in {:.2f}s".format(N,toc-tic))       
        
        return total_loss, total_loss_pred
     
     
    def getTotalDiffusivity(self, session, test_inputs, test_tensor_basis, 
                            update_stats=True, clean=True, N_std=20):
        """
        This method takes in a whole dataset and computes the diffusivity matrix for it.
        """
    
        tic = time.time()
        
        # This is a constant, which determines how big our batches are.
        # Since we go through all the data exactly one and compute the loss,
        # this parameter shouldn't change the result.
        TEST_BATCH_SIZE = 10000
        
        N = test_inputs.shape[0]
        total_diff = np.empty((N, 3, 3))
        total_g = np.empty((N, self.FLAGS['num_bases']))
        i = 0 # marks the index where the current batch starts       
        
        # This normalizes the inputs. Runs when update_stats = True
        if self.features_mean is not None and update_stats:
            test_inputs = (test_inputs - self.features_mean)/self.features_std
                    
        # Initialize batch generator
        batch_gen = BatchGenerator(TEST_BATCH_SIZE, test_inputs, test_tensor_basis)                                  
        
        # Iterate through all batches of data
        batch = batch_gen.nextBatch()        
        while batch is not None:
            n_batch = batch.x_features.shape[0] 
            total_diff[i:i+n_batch,:,:], total_g[i:i+n_batch] = self.getDiffusivity(session, batch)
            i += n_batch
            
            batch = batch_gen.nextBatch()       
        
        toc = time.time()
        #print("Computed diffusivity over {} examples in {:.2f}s".format(N,toc-tic))
        
        if clean:
            total_diff, total_g = self.cleanDiffusivity(total_diff, total_g, test_inputs, N_std)
        
        return total_diff, total_g
        
    
    def cleanDiffusivity(self, diff, g, test_inputs, N_std, PR_T=0.85):
        
        print("Cleaning predicted diffusivity... ", end="", flush=True)
        tic = time.time()
        
        # Here, make sure that there is no input more or less than N_std away
        # from the mean. The mean and standard deviation used are from the 
        # training data (which are set to 0 and 1 respectively)
        mask_extr = (np.amax(test_inputs, axis=1) > N_std) + (np.amin(test_inputs, axis=1) < -N_std)
        num_extr = np.sum(mask_extr) 
        diff, g = applyMask(diff, g, mask_extr, PR_T)         
        
        # Here, make sure that no eigenvalues have a negative real part. 
        # If they did, an unstable model is produced.
        eig_all, _ = np.linalg.eig(diff)
        t = np.amin(np.real(eig_all), axis=1) # minimum real part of eigenvalues
        mask_eig = t < 0
        num_eig = np.sum(mask_eig)
        avg_negative_eig = 0
        if num_eig > 0:
            avg_negative_eig = np.mean(t[mask_eig])        
        diff, g = applyMask(diff, g, mask_eig, PR_T)
        
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
        diff, g = applyMask(diff, g, mask_neg, PR_T)        
        
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
        
    
    def train(self, session, num_epochs, path_to_save,
              train_inputs, train_tensor_basis, train_uc, train_gradc, train_eddy_visc,
              dev_inputs, dev_tensor_basis, dev_uc, dev_gradc, dev_eddy_visc,
              update_stats=True, initialize=True, early_stop=10, subsample_devloss=None):
        """
        This method trains the model
        """
        
        print("Training...", flush=True)
        
        self.saver_path = path_to_save
        
        # If this is true, initialize all global variables
        if initialize:
            session.run(tf.global_variables_initializer())            
        
        # Normalizes the inputs and save the mean and standard deviation
        if update_stats:
            self.features_mean = np.mean(train_inputs, axis=0, keepdims=True)
            self.features_std = np.std(train_inputs, axis=0, keepdims=True)
            train_inputs = (train_inputs - self.features_mean)/self.features_std       
        
        # Keeps track of the best dev loss
        best_dev_loss = 1e10
        cur_iter=0
        to_break=False
        exp_loss=None # exponentially-smoothed training loss
        
        # Initialize batch generator
        batch_gen = BatchGenerator(self.FLAGS['train_batch_size'],
                                   train_inputs, train_tensor_basis, train_uc,
                                   train_gradc, train_eddy_visc)
        
        # This loop goes over the epochs
        for ep in range(num_epochs):
            tic = time.time()
        
            batch_gen.reset() # reset batch generator before each epoch
        
            # Iterates through batches of data
            batch = batch_gen.nextBatch()        
            while batch is not None:
                loss, step = self.runTrainIter(session, batch) # take one training step
                
                # Updates exponentially-smoothed training loss
                if exp_loss is None: exp_loss=loss
                else: exp_loss = 0.95*exp_loss + 0.05*loss
                
                # Do this every self.FLAGS['eval_every'] steps
                if step % self.FLAGS['eval_every'] == 0 and step != 0:
                    print("Step {}. Evaluating losses:".format(step), end="", flush=True)                    
                    loss_dev, loss_dev_pred = self.getTotalLoss(session, dev_inputs, 
                                              dev_tensor_basis, dev_uc, dev_gradc, 
                                              dev_eddy_visc, subsample=subsample_devloss)                                              
                    print(" Exp Train Loss: {:g} / Dev Loss: {:g}".format(exp_loss,loss_dev), flush=True)
                    
                    # If the dev loss beats the previous best, run this
                    if loss_dev_pred < best_dev_loss:
                        print("(*) New best prediction loss: {:g}".format(loss_dev_pred), flush=True)
                        best_dev_loss = loss_dev_pred
                        self.saver.save(session, self.saver_path)
                        cur_iter=0
                    else:
                        cur_iter += 1 # holds number of checkpoints since dev loss last improved
                    
                    # Detects early stopping
                    if cur_iter > early_stop:
                        to_break = True
                        break                        
                
                batch = batch_gen.nextBatch()
                
            toc = time.time()
            print("---------Epoch {} took {:.2f}s".format(ep,toc-tic), flush=True)

            if to_break:
                print("(**) Dev loss not changing... Training will stop early.", flush=True)
                break                
        
        print("Done!", flush=True)
        return best_dev_loss
    
    
    def loadParameters(self, session):
        """
        Invoke the saver to restore a previous set of parameters
        """
        self.saver.restore(session, self.saver_path)
        
    
    def saveToDisk(self, path):
        """
        Save configurations to disk (parameters are saved by the train() method)
        """
    
        print("Saving to disk...", end="", flush=True)
        
        list_variables = [self.FLAGS, self.saver_path, self.features_mean, self.features_std]
        joblib.dump(list_variables, path, protocol=2)
        
        print(" Done.", flush=True)        
        
    
    def getRANSLoss(self, uc, gradc, eddy_visc, PR_T=0.85):
        """
        Call this function to get a baseline loss (using fixed Pr_t)
        for a given dataset
        """
        
        uc_rans = -1.0 * (np.expand_dims(eddy_visc/PR_T, 1)) * gradc
        loss = np.mean(np.log(np.linalg.norm(uc_rans - uc, ord=2, axis=1)/np.linalg.norm(uc, ord=2, axis=1)))
        
        return loss    
    
    
    def printTrainableParams(self):
        """
        Call this function to print all trainable parameters of the model
        """
        
        # Prints all trainable parameters for sanity check
        params = tf.trainable_variables()
        print("This model has {} trainable parameters. They are:".format(len(params)))
        for i, v in enumerate(params):
            print("{}: {}".format(i, v.name))
            print("\t shape: {} size: {}".format(v.shape, np.prod(v.shape)))        
        
        
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
    diff[mask,:,:]=0
    diff[mask,0,0]=1.0/PR_T
    diff[mask,1,1]=1.0/PR_T
    diff[mask,2,2]=1.0/PR_T
    g[mask, :] = 0;
    g[mask, 0] = 1.0/PR_T;

    return diff, g