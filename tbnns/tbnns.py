#--------------------------------- tbnns.py file ---------------------------------------#
"""
This file contains the definition of the TBNN-s class, which is implemented using 
tensorflow. This is a Tensor Basis Neural Network that is meant to predict a tensorial
diffusivity for turbulent mixing applications.
"""

# ------------ Import statements
import tensorflow as tf
import numpy as np
import joblib
import timeit
from tbnns.data_batcher import Batch, BatchGenerator
from tbnns import constants, utils, layers

# Run this to suppress gnarly warnings/info messages from tensorflow
utils.suppressWarnings()


class TBNNS():
    """
    This class contains definitions and methods needed for the TBNN-s class.
    """
    
    def __init__(self):
        """
        Constructor class that initializes the model. Sets instance variables to None.        
        """
        self._tfsession = None
        self.FLAGS = None 
        self.features_mean = None
        self.features_std = None
        self.saver_path = None    
    
    
    def saveToDisk(self, description, path, compress=True, protocol=-1):
        """
        Save model meta-data to disk, which is used to restore it later.
        Note that trainable parameters are directly saved by the train() function.
        
        Arguments:
        description -- string, containing he description of the model being saved
        path -- string containing the path on disk in which the model is saved
        compress -- optional, boolean that is passed to joblib determining whether to
                    compress the data saved to disk.
        protocol -- optional, int containing protocol passed to joblib for saving data
                    to disk.        
        """
    
        print("Saving to disk...", end="", flush=True)
        
        list_variables = [self.FLAGS, self.saver_path, 
                          self.features_mean, self.features_std]
        joblib.dump([description, list_variables], path, 
                    compress=compress, protocol=protocol)
        
        print(" Done.", flush=True)        
    
    
    def loadFromDisk(self, path_class, verbose=False, fn_modify=None):
        """
        Invoke the saver to restore a previous set of parameters
        
        Arguments:        
        path_class -- string containing path where file is located.
        verbose -- boolean flag, whether to print more details about the model.
                   By default it is False.
        fn_modify -- function, optional argument. If this is not None, it must be a 
                     function that takes is a string and outputs another string,
                     str2 = fn_modify(str1). It is applied to the path where the 
                     model parameters are saved. This is done to allow for relative
                     paths when loadfromDisk is called. 
        """
        
        # Loading file with metadata from disk
        description, list_vars = joblib.load(path_class)
        
        FLAGS, saved_path, feat_mean, feat_std = list_vars # unpack list_vars
        self.initializeGraph(FLAGS, feat_mean, feat_std) # initialize
        if fn_modify is not None: saved_path = fn_modify(saved_path)       
        self._saver.restore(self._tfsession, saved_path) # restore previous parameters
        
        if verbose:
            print("Model loaded successfully! Description: {}".format(description))            
            self.printModelInfo()
        
        return description
    
    
    def initializeGraph(self, FLAGS, features_mean=None, features_std=None):
        """
        This method builds the tensorflow graph of the network.
        
        Arguments:
        FLAGS -- dictionary containing different parameters for the network        
        features_mean -- mean of the training features, which is used to normalize the
                         features at training time.
        features_std -- standard deviation of the training features, which is used to 
                        normalize the features at training time.
        
        Defines:
        self.global_step -- integer with the number of the current training step
        self.updates -- structure that instructs tensorflow to minimize the loss
        self._saver -- tf.train.Saver class responsible for saving the parameter values 
                      to disk
        """   
        
        # Initializes appropriate instance variables
        self.FLAGS = FLAGS        
        self.features_mean = features_mean
        self.features_std = features_std
        
        # checks to see if the parameters FLAGS is consistent
        self.checkFlags()
                
        # Add all parts of the graph
        tf.reset_default_graph()
        
        # Builds the graph by calling appropriate function
        self.constructPlaceholders()                 
        self.constructNet()       
        self.combineBasis()
        self.constructLoss()
                
        # Define optimizer and updates
        # (updates is what you need to fetch in session.run to do a gradient update)
        self.global_step = tf.Variable(0, name="global_step", trainable=False)                
        opt = tf.train.AdamOptimizer(learning_rate=self.FLAGS['learning_rate'])
        self.updates = opt.minimize(self.loss, global_step=self.global_step)
        
        # Define savers (for checkpointing)
        self._saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)        
        
        # Creates session and initializes global variables
        self._tfsession = tf.Session()
        self._tfsession.run(tf.global_variables_initializer())
        
    
    def constructPlaceholders(self):
        """
        Adds all placeholders for external data into the model

        Defines:
        self.x_features -- placeholder for features (i.e. inputs to NN)
        self.tensor_basis -- placeholder for tensor basis
        self.uc -- placeholder for the labels u'c'
        self.eddy_visc -- placeholder for eddy viscosity
        self.loss_weight -- placeholder for the loss weight (which is multiplied by
                            the L2 prediction loss element-wise)
        self.log_gamma -- placeholder for the log of gamma=1/Prt which is employed
                          when FLAGS.loss_gamma is True
        self.drop_prob -- placeholder for dropout probability        
        """
        
        self.x_features = tf.placeholder(tf.float32, 
                                     shape=[None, constants.NUM_FEATURES])
        self.tensor_basis = tf.placeholder(tf.float32, 
                                          shape=[None, constants.NUM_BASIS, 3, 3])
        self.uc = tf.placeholder(tf.float32, shape=[None, 3])
        self.gradc = tf.placeholder(tf.float32, shape=[None, 3])
        self.eddy_visc = tf.placeholder(tf.float32, shape=[None])
        self.loss_weight = tf.placeholder(tf.float32, shape=[None, 1])
        self.log_gamma = tf.placeholder(tf.float32, shape=[None])        
        self.drop_prob = tf.placeholder_with_default(0.0, shape=())
                    
    
    def constructNet(self):    
        """
        Creates the neural network that predicts g in the model            

        Defines:
        self.g -- the coefficients for each of the form invariant basis, of
                  shape (None,num_basis)
        """
        
        with tf.variable_scope("model"):        
            # Creates the first hidden state from the inputs
            fc1 = layers.FullyConnected(self.FLAGS['num_neurons'], self.drop_prob, 
                                        name="1")
            hd1 = fc1.build(self.x_features)            
            hd_list = [hd1, ] # list of all hidden states
            
            # Creates all other hidden states
            for i in range(self.FLAGS['num_layers']-1):
                fc = layers.FullyConnected(self.FLAGS['num_neurons'], self.drop_prob,
                                          name=str(i+2))
                hd_list.append(fc.build(hd_list[-1]))
            
            # Go from last hidden state to the outputs (g in this case)
            fc_last = layers.FullyConnected(constants.NUM_BASIS, self.drop_prob, 
                                           relu=False, name="last")
            self.g = fc_last.build(hd_list[-1])
        
    
    def combineBasis(self):
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
            
            # diffusivity matrix, shape [None, 3, 3]
            self.diffusivity = tf.reduce_sum(mult_bas, axis=1)         
            
            # shape of [None,3,1]
            gradc_ext = tf.expand_dims(self.gradc, -1)                  

            # shape of [None, 3], full u'c' vector
            self.uc_predicted = -1.0*( tf.expand_dims(self.eddy_visc,-1) *
                                     tf.squeeze(tf.matmul(self.diffusivity, gradc_ext)) )
        

    def constructLoss(self):
        """
        Add loss computation to the graph. 

        Defines:
        self.loss_pred -- scalar with the loss due to error between uc_predicted and uc
        self.loss_reg -- scalar with the regularization loss
        self.loss_neg -- scalar with the component of the loss due to numerically 
                         unstable negative diffusivities
        self.loss -- scalar with the total loss (sum of the two components), which is
                     what gradient descent tries to minimize
        """
        
        with tf.variable_scope("losses"):
                       
            if self.FLAGS['loss_type'] == 'log':
                self.loss_pred = layers.lossLog(self.uc, self.uc_predicted, tf_flag=True)
            if self.FLAGS['loss_type'] == 'l2_k':
                self.loss_pred = layers.lossL2k(self.uc, self.uc_predicted, 
                                                self.loss_weight, tf_flag=True)
            if self.FLAGS['loss_type'] == 'l2':
                self.loss_pred = layers.lossL2(self.uc, self.uc_predicted,
                                               self.loss_weight, tf_flag=True)
            if self.FLAGS['loss_type'] == 'l1':
                self.loss_pred = layers.lossL1(self.uc, self.uc_predicted,
                                               self.loss_weight, tf_flag=True)
            if self.FLAGS['loss_type'] == 'cos':
                self.loss_pred = layers.lossCos(self.uc, self.uc_predicted, tf_flag=True)            
            
            # Calculate the L2 regularization component of the loss            
            if self.FLAGS['reg_factor'] == 0: self.loss_reg = tf.constant(0.0)
            else:
                vars = tf.trainable_variables()
                self.loss_reg = \
                     tf.add_n([tf.nn.l2_loss(v) for v in vars if ('bias' not in v.name)]) 
            
            # negative diffusivity loss            
            if self.FLAGS['neg_factor'] == 0: self.loss_neg = tf.constant(0.0)
            else:
                diff_sym = 0.5*(self.diffusivity + 
                                tf.linalg.matrix_transpose(self.diffusivity))
                                
                # e contains eigenvalues of symmetric part, in non-decreasing order
                e, _ = tf.linalg.eigh(diff_sym) 
                self.loss_neg = tf.reduce_mean(tf.maximum(-tf.reduce_min(e,axis=1),0))               
                                                       
            # gamma loss (either due to mismatch of gamma or in negative diff regions)                            
            if self.FLAGS['gamma_factor'] == 0: self.loss_gamma = tf.constant(0.0)
            elif self.FLAGS['reduce_diff'] == False:
                log_gamma_implied = utils.calculateLogGamma(self.uc_predicted,
                                                            self.gradc, self.eddy_visc,
                                                            tf_flag=True)
                self.loss_gamma = \
                             tf.reduce_mean(tf.abs(log_gamma_implied-self.log_gamma))
            else:
                # appropriate norm of the diffusivity matrix
                diff_norm = tf.norm(self.diffusivity, ord=2, axis=[1,2])

                # flag that indicates if the given points have counter gradient transport
                ctr_grad_flag = tf.math.greater_equal(tf.reduce_sum(self.uc*self.gradc,
                                                                    axis=1), 0)                
                # take the mean of the diffusivity norm in locations where there is 
                # counter gradient transport. tf.cond is used to return zero if no points
                # contain counter gradient transport
                self.loss_gamma = tf.cond(tf.reduce_any(ctr_grad_flag), 
                                  lambda: tf.reduce_mean(tf.boolean_mask(diff_norm,
                                                                         ctr_grad_flag)),
                                  lambda: tf.constant(0.0) )                
            
            # Loss is the sum of different components
            self.loss = (self.loss_pred 
                         + self.FLAGS['reg_factor']*self.loss_reg 
                         + self.FLAGS['neg_factor']*self.loss_neg
                         + self.FLAGS['gamma_factor']*self.loss_gamma)
            

    def runTrainIter(self, batch):
        """
        This performs a single training iteration (forward pass, loss computation,
        backprop, parameter update)

        Inputs:        
        batch -- a Batch object containing information necessary for training         

        Returns:
        loss -- the loss (averaged across the batch) for this batch.
        global_step -- the current number of training iterations we have done        
        """
        # Match up our input data with the placeholders
        input_feed = {}
        input_feed[self.x_features] = batch.x_features
        input_feed[self.tensor_basis] = batch.tensor_basis
        input_feed[self.uc] = batch.uc
        input_feed[self.gradc] = batch.gradc        
        input_feed[self.eddy_visc] = batch.eddy_visc       
        input_feed[self.drop_prob] = self.FLAGS['drop_prob'] # apply dropout
        
        if batch.loss_weight is not None:
            input_feed[self.loss_weight] = batch.loss_weight
        if batch.log_gamma is not None: 
            input_feed[self.log_gamma] = batch.log_gamma
                                    
        # output_feed contains the things we want to fetch.
        output_feed = [self.updates, self.loss, self.global_step]

        # Run the model
        [_, loss, global_step] = self._tfsession.run(output_feed, input_feed)

        return loss, global_step
        
    
    def getLoss(self, batch):
        """
        This runs a single forward pass and obtains the loss.
        
        Inputs:        
        batch -- a Batch object containing information necessary for training         

        Returns:
        loss -- the loss (averaged across the batch) for this batch.
        loss_pred -- the loss just due to the error between u'c' and u'c'_predicted
        loss_reg -- the regularization component of the loss
        loss_neg -- the component of the loss due to negative diffusivity
        loss_gamma -- the component of the loss due to mismatch of gamma 
        """
        
        input_feed = {}
        input_feed[self.x_features] = batch.x_features
        input_feed[self.tensor_basis] = batch.tensor_basis
        input_feed[self.uc] = batch.uc
        input_feed[self.gradc] = batch.gradc
        
        input_feed[self.eddy_visc] = batch.eddy_visc
        
        if batch.loss_weight is not None:
            input_feed[self.loss_weight] = batch.loss_weight
        if batch.log_gamma is not None: 
            input_feed[self.log_gamma] = batch.log_gamma        
                                
        # output_feed contains the things we want to fetch.
        output_feed = [self.loss, self.loss_pred, self.loss_reg, self.loss_neg,
                       self.loss_gamma]        
        
        # Run the model
        [loss, loss_pred, loss_reg, 
        loss_neg, loss_gamma] = self._tfsession.run(output_feed, input_feed)

        return loss, loss_pred, loss_reg, loss_neg, loss_gamma
        
        
    def getDiffusivity(self, batch):
        """
        This runs a single forward pass to obtain the (dimensionless) diffusivity matrix.
        
        Inputs:        
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
        [diff, g] = self._tfsession.run(output_feed, input_feed)
        
        return diff, g
    
    
    def getTotalLosses(self, x_features, tensor_basis, uc, gradc, eddy_visc,
                       loss_weight=None, normalize=True, downsample=None, 
                       report_psd=False):
        """
        This method takes in a whole dataset and computes the average loss on it.
        
        Inputs:        
        x_features -- numpy array containing the features in the whole dataset, of shape 
                      (num_points, num_features).
        tensor_basis -- numpy array containing the tensor basis in the whole dataset, of
                        shape (num_points, num_basis, 3, 3).
        uc -- numpy array containing the label (uc vector) in the whole dataset, of
                        shape (num_points, 3).
        gradc -- numpy array containing the gradient of c vector in the whole dataset, of
                        shape (num_points, 3).
        eddy_visc -- numpy array containing the eddy viscosity in the whole dataset, of
                        shape (num_points).
        loss_weight -- numpy array of shape (num_points). Optional, only needed for
                       specific loss types.        
        normalize -- optional argument, boolean flag saying whether to normalize the 
                     features before feeding them to the neural network. True by default.
        downsample -- optional argument, ratio of points to use of the overall dataset.
                     If None, deactivate (and use all points). Must be between 0 and 1 
                     otherwise.
        report_psd -- optional argument, boolean flag containing whether to return 
                      the ratio of non-psd matrices in the predicted diffusivities
        
        Returns:
        total_loss -- a scalar, the average total loss for the whole dataset
        total_loss_pred -- a scalar, the average prediction loss for the whole dataset
        total_loss_reg -- a scalar, the average regularization loss for the whole dataset
        total_loss_neg -- a scalar, the average negative diffusivity loss for the whole
                          dataset
        total_loss_gamma -- a scalar, the average gamma loss for the whole dataset
        ratio_eig -- a scalar, the ratio of non-PSD matrices in the output of the model.
                     This is only calculated if report_psd==True; otherwise, it just 
                     contains 0.
        """
        
        # Make sure it is only None if appropriate flags are set
        self.assertArguments(loss_weight)

        # Calculates log(gamma) based on quantities received
        log_gamma = utils.calculateLogGamma(uc, gradc, eddy_visc)        
        
        # Initializes quantities we need to keep track of
        total_loss = 0
        total_loss_pred = 0
        total_loss_reg = 0
        total_loss_neg = 0
        total_loss_gamma = 0
        num_points = x_features.shape[0]
        
        # This normalizes the inputs. Runs when normalize = True
        if self.features_mean is not None and normalize:
            x_features = (x_features - self.features_mean)/self.features_std        
        
        # Initialize batch generator, downsampling only if not None        
        idx = utils.downsampleIdx(num_points, downsample)
        if loss_weight is not None: loss_weight = loss_weight[idx]                
        batch_gen = BatchGenerator(constants.TEST_BATCH_SIZE, x_features[idx,:], 
                                   tensor_basis[idx,:,:,:], uc[idx,:], gradc[idx,:],
                                   eddy_visc[idx], loss_weight, log_gamma[idx])             
        
        # Iterate through all batches of data
        batch = batch_gen.nextBatch()        
        while batch is not None:
            this_l, this_lpred, this_lreg, this_lneg, this_lgamma = self.getLoss(batch)
            total_loss += this_l * batch.x_features.shape[0]
            total_loss_pred += this_lpred * batch.x_features.shape[0]
            total_loss_reg += this_lreg * batch.x_features.shape[0]
            total_loss_neg += this_lneg * batch.x_features.shape[0]
            total_loss_gamma += this_lgamma * batch.x_features.shape[0]
            batch = batch_gen.nextBatch()
        
        # To get an average loss, divide by total number of dev points
        total_loss =  total_loss/num_points
        total_loss_pred = total_loss_pred/num_points
        total_loss_reg = total_loss_reg/num_points
        total_loss_neg = total_loss_neg/num_points
        total_loss_gamma = total_loss_gamma/num_points        
        
        # Report the number of non-PSD matrices
        if report_psd:
            diff, _, = self.getTotalDiffusivity(x_features, tensor_basis, 
                                                normalize=False, clean=False)
            diff_sym = 0.5*(diff+np.transpose(diff,axes=(0,2,1)))
            eig_all, _ = np.linalg.eigh(diff_sym)
            eig_min = np.amin(eig_all, axis=1)
            ratio_eig = np.sum(eig_min < 0) / eig_min.shape[0]
        else: ratio_eig = None           
        
        return (total_loss, total_loss_pred, total_loss_reg, total_loss_neg,
                total_loss_gamma, ratio_eig)
     
     
    def getTotalDiffusivity(self, test_x_features, test_tensor_basis, 
                            normalize=True, clean=True, n_std=None, prt_default=None, 
                            gamma_min=None):
        """
        This method takes in a whole test set and computes the diffusivity matrix on it.
        
        Inputs:        
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
        prt_default -- optional argument, default turbulent Prandtl number to use
                       whenever the output diffusivity is cleaned. If None, it will read
                       from constants.py
        gamma_min -- optional argument, minimum value of gamma = diffusivity/turbulent 
                     viscosity allowed. Used to clean values that are positive but too
                     close to zero. If None is passed, default value is read from 
                     constants.py
        
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
            total_diff[i:i+n_batch], total_g[i:i+n_batch] = self.getDiffusivity(batch)                       
            i += n_batch
            batch = batch_gen.nextBatch()
        
        # Clean the resulting diffusivity
        if clean:
            total_diff, total_g = utils.cleanDiffusivity(total_diff, total_g, 
                                                         test_x_features, n_std,
                                                         prt_default, gamma_min)        
        return total_diff, total_g  
        
    
    def train(self, path_to_saver,
              train_x_features, train_tensor_basis, train_uc, train_gradc, train_eddy_visc,
              dev_x_features, dev_tensor_basis, dev_uc, dev_gradc, dev_eddy_visc, 
              train_loss_weight=None, dev_loss_weight=None, 
              early_stop_dev=None, update_stats=True, 
              downsample_devloss=None, detailed_losses=False):
        """
        This method trains the model.
        
        Inputs:        
        path_to_saver -- string containing the location in disk where the model 
                         parameters will be saved after it is trained. 
        train_x_features -- numpy array containing the features in the training set,
                            of shape (num_train, num_features)
        train_tensor_basis -- numpy array containing the tensor basis in the training 
                              set, of shape (num_train, num_basis, 3, 3)
        train_uc -- numpy array containing the label (uc vector) in the training set, of
                    shape (num_train, 3)
        train_gradc -- numpy array containing the gradient of c vector in the training
                       set, of shape (num_train, 3)
        train_eddy_visc -- numpy array containing the eddy viscosity in the training set
                           dataset, of shape (num_train)
        dev_x_features -- numpy array containing the features in the dev set,
                          of shape (num_dev, num_features)
        dev_tensor_basis -- numpy array containing the tensor basis in the dev 
                            set, of shape (num_dev, num_basis, 3, 3)
        dev_uc -- numpy array containing the label (uc vector) in the dev set, of
                  shape (num_dev, 3)
        dev_gradc -- numpy array containing the gradient of c vector in the dev
                     set, of shape (num_dev, 3)
        dev_eddy_visc -- numpy array containing the eddy viscosity in the dev set
                         dataset, of shape (num_dev)
        train_loss_weight -- optional argument, numpy array containing the loss_weight
                             for the training data, which is used in some types of 
                             prediction losses. Shape (num_train)
        dev_loss_weight -- optional argument, numpy array containing the loss_weight
                           for the dev data, which is used in some types of 
                           prediction losses. Shape (num_dev)        
        update_stats -- bool, optional argument. Whether to normalize features and update
                        the value of mean and std of features given this training set. By
                        default is True.
        early_stop_dev -- int, optional argument. How many iterations to wait for the dev
                          loss to improve before breaking. If this is activated, then we
                          save the model that generates the best prediction dev loss even
                          if the loss went up later. If this is activated, training stops
                          early if the dev loss is not going down anymore. Note that this
                          quantities number of times we measure the dev loss, i.e.,
                          early_stop_dev * FLAGS['eval_every'] iterations. If this is
                          zero, then it is deactivated. If it is None, read from FLAGS
        downsample_devloss -- optional argument, controls whether and how much to
                            subsample the dev set to calculate losses. If the dev set
                            is very big, calculating the full loss can be slow, so you
                            can set this parameter to only use part of it. If this is
                            less than 1, it indicates a ratio (0.1 means 10% of points
                            at random are used); if this is more than 1, it indicates abs
                            number (10000 means 10k points are used at random). None
                            deactivates subsampling, which is default behavior.
        
        
        Returns:
        best_dev_loss -- The best (prediction) loss throughout training in the dev set
        end_dev_loss -- The final (prediction) loss throughout training in the dev set
        step_list -- A list containing iteration numbers at which losses are returned
        train_loss_list -- A list containing training losses throughout training
        dev_loss_list -- A list containing dev losses throughout training
        """
        
        # Make sure it is only None if appropriate flags are set
        self.assertArguments(train_loss_weight)
        self.assertArguments(dev_loss_weight)
        
        # Calculates log(gamma) based on quantities received
        train_log_gamma = utils.calculateLogGamma(train_uc, train_gradc, train_eddy_visc)       
                
        # If early_stop_dev is None, get value from FLAGS
        if early_stop_dev is None:
            early_stop_dev = self.FLAGS['early_stop_dev']
        
        print("Training...", flush=True)        
        self.saver_path = path_to_saver
                
        # Normalizes the inputs and save the mean and standard deviation
        if update_stats:
            self.features_mean = np.mean(train_x_features, axis=0, keepdims=True)
            self.features_std = np.std(train_x_features, axis=0, keepdims=True)
            train_x_features = (train_x_features - self.features_mean)/self.features_std       
        
        # Keeps track of the best dev loss
        best_dev_loss=1e10 # very high initial best loss
        cur_iter=0
        to_break=False
        exp_loss=None # exponentially-smoothed training loss
        
        # Lists of losses to keep track and plot later
        step_list = []
        train_loss_list = []
        dev_loss_list = [] # list of list of different types of losses
        
        # Initialize batch generator
        batch_gen = BatchGenerator(self.FLAGS['train_batch_size'],
                                   train_x_features, train_tensor_basis, train_uc,
                                   train_gradc, train_eddy_visc,
                                   train_loss_weight, train_log_gamma)
        
        # This loop goes over the epochs        
        for ep in range(self.FLAGS['num_epochs']):
            tic = timeit.default_timer()        
                  
            # Iterates through batches of data
            batch_gen.reset() # reset batch generator before each epoch
            batch = batch_gen.nextBatch()        
            while batch is not None:
                loss, step = self.runTrainIter(batch) # take one training step
                
                # Updates exponentially-smoothed training loss
                if exp_loss is None: exp_loss=loss
                else: exp_loss = 0.95*exp_loss + 0.05*loss
                
                # Do this every self.FLAGS['eval_every'] steps
                if step % self.FLAGS['eval_every'] == 0 and step != 0:
                    # Evaluate dev loss
                    print("Step {}. Evaluating losses:".format(step), end="", flush=True)
                    
                    if detailed_losses:
                        losses = self.getTotalLosses(dev_x_features, dev_tensor_basis,
                                                     dev_uc, dev_gradc, dev_eddy_visc,
                                                     dev_loss_weight,
                                                     downsample=downsample_devloss,
                                                     report_psd=True)          
                        (loss_dev, loss_dev_pred, loss_dev_reg, loss_dev_neg,
                         loss_dev_gamma, ratio_psd) = losses
                        
                        print(" Exp Train: {:g} | Dev: {:g}".format(exp_loss, loss_dev)
                              + " ({:.3f}% non-PSD)".format(100*ratio_psd), flush=True)
                        print("->Breakdown prediction: {:g} |".format(loss_dev_pred)
                              + " regularize: {:g} |".format(loss_dev_reg)
                              + " negative diff: {:g} |".format(loss_dev_neg)
                              + " gamma: {:g}".format(loss_dev_gamma), flush=True)                  
                    
                    else:
                        losses = self.getTotalLosses(dev_x_features, dev_tensor_basis,
                                                     dev_uc, dev_gradc, dev_eddy_visc,
                                                     dev_loss_weight,
                                                     downsample=downsample_devloss,
                                                     report_psd=False)          
                        (loss_dev, loss_dev_pred, _, loss_dev_neg, 
                         loss_dev_gamma, _) = losses                                   
                        print(" Exp Train: {:g} | Dev: {:g}".format(exp_loss, loss_dev),
                              flush=True)
                    
                    # Append to lists
                    step_list.append(step)
                    train_loss_list.append(exp_loss)
                    dev_loss_list.append(losses)                    
                    
                    # If the dev loss beats the previous best, run this
                    if loss_dev_pred < best_dev_loss:
                        print("(*) New best prediction loss: {:g}".format(loss_dev_pred),
                              flush=True)
                        best_dev_loss = loss_dev_pred
                        self._saver.save(self._tfsession, self.saver_path)
                        cur_iter_dev = 0
                    else:
                        cur_iter_dev += 1 # number of checks since dev loss last improved
                    
                    # Detects early stopping in the dev set
                    if (early_stop_dev > 0 and cur_iter_dev > early_stop_dev):
                        to_break = True
                        break                        
                
                batch = batch_gen.nextBatch()
                
            toc = timeit.default_timer()
            print("---------Epoch {} took {:.2f}s".format(ep,toc-tic), flush=True)

            if to_break:
                print("(***) Dev loss not changing... Training will stop early.", 
                      flush=True)
                break                
        
        # Calculate last dev loss 
        _, end_dev_loss, _, _, _, _ = self.getTotalLosses(dev_x_features, 
                                                          dev_tensor_basis, 
                                                          dev_uc, dev_gradc,
                                                          dev_eddy_visc, dev_loss_weight,
                                                          downsample=downsample_devloss)
        
        # save the last model if early stopping is deactivated
        if early_stop_dev == 0:
            print("Saving model with dev prediction loss {:g}... ".format(end_dev_loss), 
                  end="", flush=True)
            self._saver.save(self._tfsession, self.saver_path)
        else:
            print("End model has dev prediction loss {:g}... ".format(end_dev_loss), 
                  end="", flush=True)            
        
        print("Done!", flush=True)
        
        return best_dev_loss, end_dev_loss, step_list, train_loss_list, dev_loss_list    
           
    
    def getRansLoss(self, uc, gradc, eddy_visc, loss_weight=None, prt_default=None):
        """
        This function provides a baseline loss from a fixed turbulent Pr_t assumption.
        
        Arguments:
        uc -- numpy array, shape (n_points, 3) containing the true (LES) u'c' vector
        gradc -- numpy array, shape (n_points, 3) containing the gradient of scalar
                 concentration
        eddy_visc -- numpy array, shape (n_points,) containing the eddy viscosity
                     (with units of m^2/s) from RANS calculation
        loss_weight -- numpy array of shape (num_points). Optional, only needed for
                       specific loss types.        
        prt_default -- optional, number containing the fixed turbulent Prandtl number
                       to use. If None, use the value specified in constants.py
               
        Returns:
        loss - the mean value of the loss from using the fixed Pr_t assumption
        """
        
        self.assertArguments(loss_weight)
        
        # Calculates log(gamma) based on quantities received
        log_gamma = utils.calculateLogGamma(uc, gradc, eddy_visc)
        
        # uc_rans calculated with fixed Pr_t
        if prt_default is None:
            prt_default = constants.PR_T        
        uc_rans = -1.0 * (np.expand_dims(eddy_visc/prt_default, 1)) * gradc
        
        # return appropriate loss here        
        if self.FLAGS['loss_type'] == 'log':
            loss_pred = layers.lossLog(uc, uc_rans)
        if self.FLAGS['loss_type'] == 'l2_k':
            loss_pred = layers.lossL2k(uc, uc_rans, loss_weight)
        if self.FLAGS['loss_type'] == 'l2':
            loss_pred = layers.lossL2(uc, uc_rans, loss_weight)
        if self.FLAGS['loss_type'] == 'l1':
            loss_pred = layers.lossL1(uc, uc_rans, loss_weight)
        if self.FLAGS['loss_type'] == 'cos':
            loss_pred = layers.lossCos(uc, uc_rans)        
        
        # Calculate loss gamma
        if self.FLAGS['gamma_factor'] == 0: loss_gamma = 0
        elif self.FLAGS['reduce_diff']: loss_gamma = 1.0/prt_default        
        else: loss_gamma = np.mean(np.abs(log_gamma-np.log(1.0/prt_default)))        
        
        return loss_pred, loss_gamma
    
            
    def checkFlags(self):
        """
        This function checks some key flags in the dictionary self.FLAGS to make
        sure they are valid. If they have not been passed in, this also sets them
        to default values.
        """
        
        # List of all properties that have to be non-negative
        list_keys = ['num_layers', 'num_neurons', 'num_epochs', 'early_stop_dev',
                     'train_batch_size', 'eval_every', 'learning_rate', 'reg_factor', 
                     'neg_factor', 'gamma_factor']
        list_defaults = [constants.NUM_LAYERS, constants.NUM_NEURONS,
                         constants.EARLY_STOP_DEV, constants.TRAIN_BATCH_SIZE,
                         constants.LEARNING_RATE, constants.EVAL_EVERY,
                         constants.REG_FACTOR, constants.NEG_FACTOR,
                         constants.GAMMA_FACTOR]                         
        for key, default in zip(list_keys, list_defaults):
            if key in self.FLAGS:
                assert self.FLAGS[key] >= 0, "FLAGS['{}'] can't be negative!".format(key)
            else:
                self.FLAGS[key] = default
        
        # Check if a loss type has been passed, if not use default
        if 'loss_type' in self.FLAGS:
            assert self.FLAGS['loss_type'] == 'log' or \
                   self.FLAGS['loss_type'] == 'l2_k' or \
                   self.FLAGS['loss_type'] == 'l2' or \
                   self.FLAGS['loss_type'] == 'l1' or \
                   self.FLAGS['loss_type'] == 'cos', "FLAGS['loss_type'] is not valid!"
        else:            
            self.FLAGS['loss_type'] = constants.LOSS_TYPE
            
        # Check dropout rate
        if 'drop_prob' in self.FLAGS:
            assert self.FLAGS['drop_prob'] >= 0 and self.FLAGS['drop_prob'] <= 1, \
                  "FLAGS['drop_prob'] must be between 0 and 1"
        else:
            self.FLAGS['drop_prob'] = constants.DROP_PROB
        
        # Reduce diffusivity flag
        if 'reduce_diff' in self.FLAGS:
            assert self.FLAGS['reduce_diff'] is True or \
                   self.FLAGS['reduce_diff'] is False, \
                        "FLAGS['reduce_diff'] must be True or False!"
        else:
            self.FLAGS['reduce_diff'] = False
    
    
    def printModelInfo(self):
        """
        Call this function to print all flags and trainable parameters of the model.
        """
        # Prints all flags used to train the model
        print("FLAGS employed:")
        for key in self.FLAGS:
            print(" {} --- {}".format(key, self.FLAGS[key]))
               
        # Prints all trainable parameters     
        params = tf.trainable_variables()
        print("This model has {} trainable parameters. They are:".format(len(params)))
        for i, v in enumerate(params):
            print("{}: {}".format(i, v.name))
            print("\t shape: {} size: {}".format(v.shape, np.prod(v.shape)))
            
        
    def assertArguments(self, loss_weight):
        """
        This function makes sure that loss_weight passed in is
        appropriate to the flags we have, i.e., they are only None if they are
        not needed.
        
        Arguments:
        loss_weight -- either None or a numpy array of shape (num_points). We need
                       to make sure it is not None when we need it.
        """
        
        # Check to see if we have all we need for the current configuration                   
        if self.FLAGS['loss_type'] == 'l2' or self.FLAGS['loss_type'] == 'l2_k' or \
           self.FLAGS['loss_type'] == 'l1':
            msg = "loss_weight cannot be None since loss_type=l2 or l2_k or l1"
            assert loss_weight is not None, msg  