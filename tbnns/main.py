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
import timeit
from tbnns.data_batcher import Batch, BatchGenerator
from tbnns import constants
from tbnns import utils
from pkg_resources import get_distribution

utils.suppressWarnings()


def printInfo():
    """
    Makes sure everything is properly installed.
    
    We print a welcome message, and the version of the package. Return 1 at the end
    if no exceptions were raised.
    """
    
    print('Welcome to TBNN-s - Tensor Basis Neural Network for Scalar Mixing package!')
    
    # Get distribution version
    dist = get_distribution('tbnns')
    print('Version: {}'.format(dist.version))    
       
    return 1 # return this if everything went ok
    

# Run this to suppress gnarly warnings/info messages from tensorflow
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
        self.g -- the coefficients for each of the form invariant basis, of
                  shape (None,num_basis)
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
            self.uc_predicted = -1.0 * ( tf.expand_dims(self.eddy_visc,-1) *
                                         tf.squeeze(tf.matmul(self.diffusivity, gradc)) )
        

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
            loss_prediction = ( tf.norm(self.uc - self.uc_predicted, ord=2, axis=1) /
                                tf.norm(self.uc, ord=2, axis=1) )
            
            self.loss_pred = tf.reduce_mean(tf.log(loss_prediction))               
            
            # Calculate the L2 regularization component of the loss
            vars = tf.trainable_variables()
            self.loss_l2 = self.FLAGS['l2_reg'] *\
                     tf.add_n([tf.nn.l2_loss(v) for v in vars if ('bias' not in v.name)])             
            
            # Loss is the sum of different components
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
                     eddy_visc, normalize=True, downsample=None):
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
        downsample -- optional argument, ratio of points to use of the overall dataset.
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
        idx = utils.downsampleIdx(num_points, downsample)
        batch_gen = BatchGenerator(constants.TEST_BATCH_SIZE, x_features[idx,:], 
                                   tensor_basis[idx,:,:,:], uc[idx,:], 
                                   gradc[idx,:], eddy_visc[idx])             
        
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
                            normalize=True, clean=True, n_std=None, prt_default=None, 
                            gamma_min=None):
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
            total_diff[i:i+n_batch,:,:], total_g[i:i+n_batch] = self.getDiffusivity(session,
                                                                                    batch)
            i += n_batch
            batch = batch_gen.nextBatch()
        
        # Clean the resulting diffusivity and g arrays
        if clean:
            total_diff, total_g = utils.cleanDiffusivity(total_diff, total_g,
                                                         test_x_features, n_std, 
                                                         prt_default, gamma_min)
        
        return total_diff, total_g   
        
    
    def train(self, session, num_epochs, path_to_saver,
              train_x_features, train_tensor_basis, train_uc, train_gradc, train_eddy_visc,
              dev_x_features, dev_tensor_basis, dev_uc, dev_gradc, dev_eddy_visc,
              update_stats=True, early_stop_dev=0, subsample_devloss=None):
        """
        This method trains the model.
        
        Inputs:
        session -- current TensorFlow session
        num_epochs -- int, contains max number of epochs to train the model for
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
                          zero, then it is deactivated.
        subsample_devloss -- optional argument, controls whether and how much to
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
        
        print("Training...", flush=True)        
        self.saver_path = path_to_saver
        session.run(tf.global_variables_initializer()) # initializes parameters
        
        # Normalizes the inputs and save the mean and standard deviation
        if update_stats:
            self.features_mean = np.mean(train_x_features, axis=0, keepdims=True)
            self.features_std = np.std(train_x_features, axis=0, keepdims=True)
            train_x_features = (train_x_features - self.features_mean)/self.features_std       
        
        # Keeps track of the best dev loss
        best_dev_loss=1e10
        cur_iter=0
        to_break=False
        exp_loss=None # exponentially-smoothed training loss
        
        # Lists of losses to keep track and plot later
        step_list = []
        train_loss_list = []
        dev_loss_list = []
        
        # Initialize batch generator
        batch_gen = BatchGenerator(self.FLAGS['train_batch_size'],
                                   train_x_features, train_tensor_basis, train_uc,
                                   train_gradc, train_eddy_visc)
        
        # This loop goes over the epochs        
        for ep in range(num_epochs):
            tic = timeit.default_timer()
        
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
                    # Evaluate dev loss
                    print("Step {}. Evaluating losses:".format(step), end="", flush=True)                    
                    loss_dev, loss_dev_pred = self.getTotalLoss(session, dev_x_features, 
                                                                dev_tensor_basis, dev_uc, 
                                                                dev_gradc, dev_eddy_visc,
                                                            downsample=subsample_devloss)                                              
                    print(" Exp Train Loss: {:g} / Dev Loss: {:g}".format(exp_loss,
                          loss_dev), flush=True)
                    
                    # Append to lists
                    step_list.append(step)
                    train_loss_list.append(exp_loss)
                    dev_loss_list.append(loss_dev)                    
                    
                    # If the dev loss beats the previous best, run this
                    if loss_dev_pred < best_dev_loss:
                        print("(*) New best prediction loss: {:g}".format(loss_dev_pred),
                              flush=True)
                        best_dev_loss = loss_dev_pred
                        self.saver.save(session, self.saver_path)
                        cur_iter_dev = 0
                    else:
                        cur_iter_dev += 1 # number of checks since dev loss last improved
                    
                    # Detects early stopping in the dev set
                    if early_stop_dev > 0 and cur_iter_dev > early_stop_dev:
                        to_break = True
                        break                        
                
                batch = batch_gen.nextBatch()
                
            toc = timeit.default_timer()
            print("---------Epoch {} took {:.2f}s".format(ep,toc-tic), flush=True)

            if to_break:
                print("(**) Dev loss not changing... Training will stop early.", 
                      flush=True)
                break                
        
        # Calculate last dev loss 
        _, end_dev_loss = self.getTotalLoss(session, dev_x_features, 
                                            dev_tensor_basis, dev_uc, dev_gradc,
                                            dev_eddy_visc, downsample=subsample_devloss)       
        if early_stop_dev == 0: # save the last model if early stopping is deactivated           
            print("Saving model with dev prediction loss {:g}... ".format(end_dev_loss), 
                  end="", flush=True)
            self.saver.save(session, self.saver_path)
        
        print("Done!", flush=True)
        
        return best_dev_loss, end_dev_loss, step_list, train_loss_list, dev_loss_list
    
    
    def loadParameters(self, session, saver_path=None):
        """
        Invoke the saver to restore a previous set of parameters
        
        Arguments:
        session -- current TensorFlow session
        saver_path -- optional argument, path where file is located. If None (default),
                      use the current value of self.saver_path
        """
        
        if saver_path is not None:
            self.saver_path = saver_path
        
        self.saver.restore(session, self.saver_path)
        
    
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
        
    
    def getRANSLoss(self, uc, gradc, eddy_visc, prt=None):
        """
        This function provides a baseline loss from a fixed turbulent Pr_t assumption.
        
        Arguments:
        uc -- numpy array, shape (n_points, 3) containing the true u'c' vector
        gradc -- numpy array, shape (n_points, 3) containing the gradient of scalar
                 concentration
        eddy_visc -- numpy array, shape (n_points,) containing the eddy viscosity
                     (with units of m^2/s) from RANS calculation
        prt -- optional, number containing the fixed turbulent Prandtl number to use.
               If None, use the value specified in contants.py
               
        Returns:
        loss - the mean value of the loss from using the fixed Pr_t assumption
        """
        
        if prt is None:
            prt = constants.PR_T
        
        uc_rans = -1.0 * (np.expand_dims(eddy_visc/prt, 1)) * gradc
        loss = np.mean( ( np.log(np.linalg.norm(uc_rans - uc, ord=2, axis=1)/
                         np.linalg.norm(uc, ord=2, axis=1)) ) )
        
        return loss    
    
    
    def printTrainableParams(self):
        """
        Call this function to print all trainable parameters of the model.
        """
        
        # Prints all trainable parameters for sanity check
        params = tf.trainable_variables()
        print("This model has {} trainable parameters. They are:".format(len(params)))
        for i, v in enumerate(params):
            print("{}: {}".format(i, v.name))
            print("\t shape: {} size: {}".format(v.shape, np.prod(v.shape)))       
