#--------------------------------- main.py file ---------------------------------------#
"""
This file contains the definition of the TBNN-s class, which is implemented using 
tensorflow. This is a Tensor Basis Neural Network that is meant to predict a tensorial
diffusivity for turbulent mixing applications.
"""

# ------------ Import statements
import tensorflow as tf
import numpy as np
import joblib
import time
from tbnns.data_batcher import Batch, BatchGenerator
from tbnns import constants
from tbnns import utils

# Run this to suppress gnarly warnings/info messages from tensorflow
utils.suppressWarnings()

class TBNNS(object):
    """
    This class contains definitions and methods needed for the TBNN-s class.
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
        
        Defines:
        self.global_step -- integer with the number of the current training step
        self.updates -- structure that instructs tensorflow to minimize the loss
        self.saver -- tf.train.Saver class responsible for saving the parameter values 
                      to disk
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
                                     shape=[None, constants.NUM_FEATURES])
        self.tensor_basis = tf.placeholder(tf.float32, 
                                          shape=[None, constants.NUM_BASIS, 3, 3])
        self.uc = tf.placeholder(tf.float32, shape=[None, 3])
        self.gradc = tf.placeholder(tf.float32, shape=[None, 3])
        self.eddy_visc = tf.placeholder(tf.float32, shape=[None])                
        self.drop_prob = tf.placeholder_with_default(0.0, shape=())
                    
    
    def constructNN(self):    
        """
        Creates the neural network section of the model, with fully connected layers            

        Defines:
        self.g -- the coefficients for each of the form invariant basis, shape (None,num_basis)
        """
        
        # Creates the first hidden state from the inputs
        fc1 = utils.FullyConnected(self.FLAGS['num_neurons'], self.drop_prob, name="1")
        hd1 = fc1.build(self.x_features)
        
        hd_list = [hd1, ] # list of all hidden states
        
        # Creates all other hidden states
        for i in range(self.FLAGS['num_layers']-1):
            fc = utils.FullyConnected(self.FLAGS['num_neurons'], self.drop_prob,
                                      name=str(i+2))
            hd_list.append(fc.build(hd_list[-1]))
        
        # Go from last hidden state to the outputs (g in this case)
        fc_last = utils.FullyConnected(constants.NUM_BASIS, self.drop_prob, 
                                       relu=False, name="last")
        self.g = fc_last.build(hd_list[-1])

    
    def combineBases(self):
        """
        Uses the coefficients g to calculate the diffusivity and the u'c'
        
        Defines:
        self.diffusivity -- diffusivity tensor, shape (None,3,3)
        self.uc_predicted -- predicted value of u'c', shape (None,3)        
        """
        
        with tf.variable_scope("bases"):
        
            # shape of [None,num_bases,3,3]    
            mult_bas = tf.multiply(tf.reshape(self.g,shape=[-1,constants.NUM_BASIS,1,1]), 
                                   self.tensor_basis) 
            
            # shape of [None,3,3]            
            self.diffusivity = tf.reduce_sum(mult_bas, axis=1)
            
            # shape of [None,3,1]
            gradc = tf.expand_dims(self.gradc, -1)

            # shape of [None, 3]
            self.uc_predicted = -1.0*tf.expand_dims(self.eddy_visc,-1)*tf.squeeze(tf.matmul(self.diffusivity, gradc))
        

    def addLoss(self):
        """
        Add loss computation to the graph.

        Defines:
        self.loss_pred -- scalar with the loss due to error between uc_predicted and uc
        self.loss_l2 -- scalar with the regularization loss
        self.loss -- scalar with the total loss (sum of the two components), which is
                     what gradient descent tries to minimize
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
        session -- current TensorFlow session
        batch -- a Batch object containing information necessary for training         

        Returns:
        loss -- the loss (averaged across the batch) for this batch.
        global_step -- the current number of training iterations we have done        
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
        This runs a single forward pass and obtains the loss.
        
        Inputs:
        session -- current TensorFlow session
        batch -- a Batch object containing information necessary for training         

        Returns:
        loss -- the loss (averaged across the batch) for this batch.
        loss_pred -- the loss just due to the error between u'c' and u'c'_predicted
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
        This runs a single forward pass to obtain the (dimensionless) diffusivity matrix.
        
        Inputs:
        session -- current TensorFlow session
        batch -- a Batch object containing information necessary for testing         

        Returns:
        diff -- the diffusivity tensor for this batch, a numpy array of shape (None,3,3)
        g -- the coefficient multiplying each tensor basis, a numpy array of 
             shape (None,num_basis)
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
                     eddy_visc, normalize=True, subsample=None):
        """
        This method takes in a whole dataset and computes the average loss on it.
        
        Inputs:
        session -- current TensorFlow session
        x_features -- numpy array containing the features in the whole dataset, of shape 
                      (num_points, num_features)
        tensor_basis -- numpy array containing the tensor basis in the whole dataset, of
                        shape (num_points, num_basis, 3, 3)
        uc -- numpy array containing the label (uc vector) in the whole dataset, of
                        shape (num_points, 3)
        gradc -- numpy array containing the gradient of c vector in the whole dataset, of
                        shape (num_points, 3)
        eddy_visc -- numpy array containing the eddy viscosity in the whole dataset, of
                        shape (num_points)
        normalize -- optional argument, boolean flag saying whether to normalize the 
                     features before feeding them to the neural network. True by default.
        subsample -- optional argument, ratio of points to use of the overall dataset.
                     If None, deactivate (and use all points). Must be between 0 and 1 
                     otherwise.
        Returns:
        mean_loss -- a scalar, the average total loss for the whole dataset
        mean_loss_pred -- a scalar, the average prediction loss for the whole dataset
        """
    
        # Initializes quantities we need to keep track of
        total_loss = 0
        total_loss_pred = 0
        num_points = x_features.shape[0]
        
        # This normalizes the inputs. Runs when normalize = True
        if self.features_mean is not None and normalize:
            x_features = (x_features - self.features_mean)/self.features_std        
        
        # Initialize batch generator (either by subsampling or not)
        batch_gen = None
        if subsample is not None:
            assert subsample > 0 and subsample <= 1.0, \
                                "subsample must be between 0 and 1!"
            idx_tot = np.arange(num_points)
            num_points = int(num_points*subsample)            
            np.random.shuffle(idx_tot)
            idx = idx_tot[0:num_points]
            batch_gen = BatchGenerator(constants.TEST_BATCH_SIZE, x_features[idx,:], 
                                       tensor_basis[idx,:,:,:], uc[idx,:], 
                                       gradc[idx,:], eddy_visc[idx])
        else:
            batch_gen = BatchGenerator(constants.TEST_BATCH_SIZE, x_features, 
                                       tensor_basis, uc, gradc, eddy_visc)       
        
        # Iterate through all batches of data
        batch = batch_gen.nextBatch()        
        while batch is not None:
            this_loss, this_loss_pred = self.getLoss(session, batch)
            total_loss += this_loss * batch.x_features.shape[0]
            total_loss_pred += this_loss_pred * batch.x_features.shape[0]  
            batch = batch_gen.nextBatch()
        
        # To get an average loss, divide by total number of dev points
        total_loss =  total_loss/num_points
        total_loss_pred = total_loss_pred/num_points              
        
        return total_loss, total_loss_pred
     
     
    def getTotalDiffusivity(self, session, test_x_features, test_tensor_basis, 
                            normalize=True, clean=True, n_std=None):
        """
        This method takes in a whole test set and computes the diffusivity matrix on it.
        
        Inputs:
        session -- current TensorFlow session
        test_x_features -- numpy array containing the features in the whole dataset, of
                           shape (num_points, num_features)
        test_tensor_basis -- numpy array containing the tensor basis in the whole
                             dataset, of shape (num_points, num_basis, 3, 3)        
        normalize -- optional argument, boolean flag saying whether to normalize the 
                     features before feeding them to the neural network. True by default.
        clean -- optional argument, whether to clean the output diffusivity according
                 to the function defined in utils.py. True by default.
        n_std -- number of standard deviations around the mean which is the threshold to
                 characterize a point as an outlier. This is passed to the cleaning 
                 function, which sets a default value for all outlier points. By default
                 it is None, which means that the value in constants.py is used instead
        
        Returns:
        total_diff -- dimensionless diffusivity predicted, numpy array of shape 
                      (num_points, 3, 3)
        total_g -- coefficients that multiply each of the tensor basis predicted, a
                   numpy array of shape (num_points, num_basis)
        """
    
        num_points = test_x_features.shape[0]
        total_diff = np.empty((num_points, 3, 3))
        total_g = np.empty((num_points, constants.NUM_BASIS))
        i = 0 # marks the index where the current batch starts       
        
        # This normalizes the inputs. Runs when normalize = True
        if self.features_mean is not None and normalize:
            test_x_features = (test_x_features - self.features_mean)/self.features_std
                    
        # Initialize batch generator. We do not call reset() to avoid shuffling the batch
        batch_gen = BatchGenerator(constants.TEST_BATCH_SIZE, test_x_features, 
                                   test_tensor_basis)                                  
        
        # Iterate through all batches of data
        batch = batch_gen.nextBatch()        
        while batch is not None:
            n_batch = batch.x_features.shape[0] # number of points in this batch
            total_diff[i:i+n_batch,:,:], total_g[i:i+n_batch] = self.getDiffusivity(session, batch)
            i += n_batch            
            batch = batch_gen.nextBatch()      
        
        # Clean the resulting diffusivity and g arrays
        if clean:
            total_diff, total_g = utils.cleanDiffusivity(total_diff, total_g,
                                                         test_x_features, n_std)
        
        return total_diff, total_g   
        
    
    def train(self, session, num_epochs, path_to_saver,
              train_inputs, train_tensor_basis, train_uc, train_gradc, train_eddy_visc,
              dev_inputs, dev_tensor_basis, dev_uc, dev_gradc, dev_eddy_visc,
              update_stats=True, initialize=True, early_stop=10, subsample_devloss=None):
        """
        This method trains the model
        """
        
        print("Training...", flush=True)
        
        self.saver_path = path_to_saver
        
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
        
    
    def getRANSLoss(self, uc, gradc, eddy_visc, prt=None):
        """
        Call this function to get a baseline loss (using fixed Pr_t)
        for a given dataset
        """
        
        if prt is None:
            prt = constants.PR_T
        
        uc_rans = -1.0 * (np.expand_dims(eddy_visc/prt, 1)) * gradc
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
        
        
