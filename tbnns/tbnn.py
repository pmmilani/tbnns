#--------------------------------- tbnn.py file ---------------------------------------#
"""
This file contains the definition of the TBNN class, which is implemented using 
tensorflow. This is similar to the TBNN-s, and proposed originally by Ling et al. in
J. Fluid Mech. (2016). This is a Tensor Basis Neural Network that is meant to predict 
the Reynolds stresses anisotropy tensor.
"""

# ------------ Import statements
import tensorflow as tf
import numpy as np
import joblib
import timeit
from tbnns.data_batcher import Batch, BatchGenerator
from tbnns import constants, utils, nn, momentum_losses


class TBNN(nn.NNBasic):
    """
    This class contains definitions and methods needed for the TBNN class.
    """        
    
    def constructPlaceholders(self):
        """
        Adds all placeholders for external data into the model

        Defines:
        self.x_features -- placeholder for features (i.e. inputs to NN)
        self.tensor_basis -- placeholder for tensor basis
        self.b -- placeholder for the labels, the anisotropy tensor
        self.loss_weight -- placeholder for the loss weight (which is multiplied by
                            the L2 prediction loss element-wise)        
        self.drop_prob -- placeholder for dropout probability        
        """
        
        self.x_features = tf.compat.v1.placeholder(tf.float32, 
                                     shape=[None, self.FLAGS['num_features']])
        self.tensor_basis = tf.compat.v1.placeholder(tf.float32, 
                                          shape=[None, self.FLAGS['num_basis'], 3, 3])
        self.b = tf.compat.v1.placeholder(tf.float32, shape=[None, 3, 3])
        self.loss_weight = tf.compat.v1.placeholder_with_default(tf.ones([1, 1, 1]),
                                                                shape=[None, None, None])
        self.drop_prob = tf.compat.v1.placeholder_with_default(0.0, shape=())
                    
    
    def constructNet(self):    
        """
        Creates the neural network that predicts g in the model            

        Defines:
        self.g -- the coefficients for each of the form invariant basis, of
                  shape (None,num_basis)
        """
        
        with tf.compat.v1.variable_scope("model"):
            
            # constant g model
            if self.FLAGS['num_layers'] == -1:
                g_initial = np.zeros((1, self.FLAGS['num_basis']), dtype=np.float32)
                g_initial[0, 0] = 1.0/constants.PRT_DEFAULT
                self.g = tf.compat.v1.Variable(initial_value=g_initial,trainable=True,
                                               name="g_const")
            
            # 0 hidden layers, so linear regression from features to g
            elif self.FLAGS['num_layers'] == 0:
                fc1 = nn.FullyConnected(self.FLAGS['num_basis'], self.drop_prob,
                                            relu=False, name="linear")
                self.g = fc1.build(self.x_features)
            
            # regular deep learning model
            else:
                # Creates the first hidden state from the inputs
                fc1 = nn.FullyConnected(self.FLAGS['num_neurons'], self.drop_prob,
                                            name="1")
                hd1 = fc1.build(self.x_features)
                hd_list = [hd1, ] # list of all hidden states
                
                # Creates all other hidden states
                for i in range(self.FLAGS['num_layers']-1):
                    fc = nn.FullyConnected(self.FLAGS['num_neurons'], self.drop_prob,
                                              name=str(i+2))
                    hd_list.append(fc.build(hd_list[-1]))
                
                # Go from last hidden state to the outputs (g in this case)
                fc_last = nn.FullyConnected(self.FLAGS['num_basis'], self.drop_prob, 
                                               relu=False, name="last")
                self.g = fc_last.build(hd_list[-1])
        
    
    def combineBasis(self):
        """
        Uses the coefficients g to calculate the Reynolds stress anisotropy b
        
        Defines:        
        self.b_predicted -- predicted value of anisotropy, shape (None,3,3)        
        """
        
        with tf.compat.v1.variable_scope("bases"):        
            # shape of [None,num_bases,3,3]    
            mult_bas = tf.multiply(tf.reshape(self.g,shape=[-1,self.FLAGS['num_basis'],1,1]), 
                                   self.tensor_basis) 
            
            # diffusivity matrix, shape [None, 3, 3]
            self.b_predicted = tf.reduce_sum(mult_bas, axis=1)       
        

    def constructLoss(self):
        """
        Add loss computation to the graph. 

        Defines:
        self.loss_pred -- scalar with the loss due to error between b_predicted and b
        self.loss_reg -- scalar with the regularization loss        
        self.loss -- scalar with the total loss (sum of all components), which is
                     what gradient descent tries to minimize
        """
        
        with tf.compat.v1.variable_scope("losses"):
            
                    
            if self.FLAGS['loss_type'] == 'l2':
                self.loss_pred = momentum_losses.lossL2(self.b, self.b_predicted,
                                               self.loss_weight, tf_flag=True)
            if self.FLAGS['loss_type'] == 'l1':
                self.loss_pred = momentum_losses.lossL1(self.b, self.b_predicted,
                                               self.loss_weight, tf_flag=True)                 
            
            # Calculate the L2 regularization component of the loss            
            if self.FLAGS['c_reg'] == 0: self.loss_reg = tf.constant(0.0)
            else:
                vars = tf.compat.v1.trainable_variables()
                self.loss_reg = \
                     tf.add_n([tf.nn.l2_loss(v) for v in vars if ('bias' not in v.name)])            
                            
            
            # Loss is the sum of different components
            self.loss = (self.loss_pred 
                         + self.FLAGS['c_reg']*self.loss_reg)
            
    
    def getLoss(self, batch):
        """
        This runs a single forward pass and obtains the loss.
        
        Inputs:        
        batch -- a Batch object containing information necessary for training         
        
        Returns:
        loss -- the loss (averaged across the batch) for this batch.
        loss_pred -- the loss just due to the error between b and b_predicted
        loss_reg -- the regularization component of the loss
        """
        
        input_feed = {}
        input_feed[self.x_features] = batch.x_features
        input_feed[self.tensor_basis] = batch.tensor_basis
        input_feed[self.b] = batch.b        
        
        if batch.loss_weight is not None:
            while len(batch.loss_weight.shape) < 3:
                batch.loss_weight = np.expand_dims(batch.loss_weight,-1)
            input_feed[self.loss_weight] = batch.loss_weight       
                                
        # output_feed contains the things we want to fetch.
        output_feed = [self.loss, self.loss_pred, self.loss_reg]        
        
        # Run the model
        [loss, loss_pred, loss_reg] = self._tfsession.run(output_feed, input_feed)

        return loss, loss_pred, loss_reg
        
    
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
        input_feed[self.b] = batch.b
        input_feed[self.drop_prob] = self.FLAGS['drop_prob'] # apply dropout
        
        if batch.loss_weight is not None:
            while len(batch.loss_weight.shape) < 3:
                batch.loss_weight = np.expand_dims(batch.loss_weight,-1)
            input_feed[self.loss_weight] = batch.loss_weight        
                                            
        # output_feed contains the things we want to fetch.
        output_feed = [self.updates, self.loss, self.global_step]

        # Run the model
        [_, loss, global_step] = self._tfsession.run(output_feed, input_feed)

        return loss, global_step    
    
    
    def getTotalLosses(self, x_features, tensor_basis, b,
                       loss_weight=None, normalize=True, downsample=None,
                       report_real=False):
        """
        This method takes in a whole dataset and computes the average loss on it.
        
        Inputs:        
        x_features -- numpy array containing the features in the whole dataset, of shape 
                      (num_points, num_features).
        tensor_basis -- numpy array containing the tensor basis in the whole dataset, of
                        shape (num_points, num_basis, 3, 3).
        b -- numpy array containing the label (anisotropy tensor, b_ij) in the whole
             dataset, of shape (num_points, 3, 3).        
        loss_weight -- numpy array of shape (num_points) or (num_points, 1, 1) or 
                       (num_points, 3, 3). This weights each point and possibly each 
                       component differently when assessing the predicted anisotropy
                       tensor. Optional, no weighting applied when this is None.        
        normalize -- optional argument, boolean flag saying whether to normalize the 
                     features before feeding them to the neural network. True by default.
        downsample -- optional argument, ratio of points to use of the overall dataset.
                     If None, deactivate (and use all points). Must be between 0 and 1 
                     otherwise.
        report_real -- optional argument, boolean flag containing whether to return 
                       the ratio of non-realizable anisotropy tensors
        
        Returns:
        total_loss -- a scalar, the average total loss for the whole dataset
        total_loss_pred -- a scalar, the average prediction loss for the whole dataset
        total_loss_reg -- a scalar, the average regularization loss for the whole dataset
        ratio_eig -- a scalar, the ratio of non-realizable anisotropy tensors in the
                     output of the model. This is only calculated if report_real==True;
                     otherwise, it just contains None.
        """
        
        # Make sure flags are consistent with arguments passed in
        self.assertArguments()

        # Initializes quantities we need to keep track of
        total_loss = 0
        total_loss_pred = 0
        total_loss_reg = 0            
        num_points = x_features.shape[0]
        
        # This normalizes the inputs. Runs when normalize = True
        if self.features_mean is not None and normalize:
            x_features = (x_features - self.features_mean)/self.features_std        
        
        # Initialize batch generator, downsampling only if not None        
        idx = utils.downsampleIdx(num_points, downsample)
        if loss_weight is not None: loss_weight = loss_weight[idx]        
        batch_gen = BatchGenerator(constants.TEST_BATCH_SIZE, x_features[idx,:], 
                                   tensor_basis[idx,:,:,:], b=b[idx,:,:], 
                                   loss_weight=loss_weight)             
        
        # Iterate through all batches of data
        batch = batch_gen.nextBatch()        
        while batch is not None:
            l, l_pred, l_reg = self.getLoss(batch)
            total_loss += l * batch.x_features.shape[0]
            total_loss_pred += l_pred * batch.x_features.shape[0]
            total_loss_reg += l_reg * batch.x_features.shape[0]                      
            batch = batch_gen.nextBatch()
        
        # To get an average loss, divide by total number of dev points
        total_loss =  total_loss/num_points
        total_loss_pred = total_loss_pred/num_points
        total_loss_reg = total_loss_reg/num_points

        # Report the number of non-realizable matrices        
        if report_real:
            b_pred, _ = self.getTotalAnisotropy(x_features, tensor_basis,                                                
                                                normalize=False, clean=False)            
            mask = np.zeros(b_pred.shape[0], dtype=bool)
            for i in range(3):
                for j in range(3):
                    if i == j:
                        mask = mask + (b_pred[:,i,j] > 2.0/3)
                        mask = mask + (b_pred[:,i,j] < -1.0/3)
                    else:
                        mask = mask + (b_pred[:,i,j] > 0.5)
                        mask = mask + (b_pred[:,i,j] < -0.5)
            eig_all, _ = np.linalg.eigh(b_pred)
            mask = mask + (eig_all[:, 2] < (3.0*np.abs(eig_all[:, 1]) - eig_all[:, 1])/2.0)
            mask = mask + (eig_all[:, 2] > 1.0/3 - eig_all[:, 1])
            n_fail = np.sum(mask)
            ratio_real = 1.0*n_fail/(x_features.shape[0])          
        else: ratio_real = None
        
        return total_loss, total_loss_pred, total_loss_reg, ratio_real
      
    
    def train(self, path_to_saver,
              train_x_features, train_tensor_basis, train_b,
              dev_x_features, dev_tensor_basis, dev_b, 
              train_loss_weight=None, dev_loss_weight=None,              
              early_stop_dev=None, update_stats=True, 
              downsample_devloss=None, detailed_losses=False,
              description=None, path=None):
        """
        This method trains the model.
        
        Inputs:        
        path_to_saver -- string containing the location in disk where the model 
                         parameters will be saved after it is trained. 
        train_x_features -- numpy array containing the features in the training set,
                            of shape (num_train, num_features)
        train_tensor_basis -- numpy array containing the tensor basis in the training 
                              set, of shape (num_train, num_basis, 3, 3)
        train_b -- numpy array containing the label (anisotropy tensor) in the training
                   set, of shape (num_train, 3, 3)        
        dev_x_features -- numpy array containing the features in the dev set,
                          of shape (num_dev, num_features)
        dev_tensor_basis -- numpy array containing the tensor basis in the dev 
                            set, of shape (num_dev, num_basis, 3, 3)
        dev_b -- numpy array containing the label (b tensor) in the dev set, of
                  shape (num_dev, 3, 3)        
        train_loss_weight -- numpy array of shape (num_points) or (num_points, 1, 1) or 
                             (num_points, 3, 3). This contains the loss weight for the 
                             training data; it is used to weight each point and possibly
                             each component differently when assessing the predicted
                             anisotropy tensor. Optional, no weighting applied when this
                             is None.     
        dev_loss_weight -- numpy array of shape (num_points) or (num_points, 1, 1) or 
                           (num_points, 3, 3). This contains the loss weight for the 
                           validation data; it is used to weight each point and possibly
                           each component differently when assessing the predicted
                           anisotropy tensor. Optional, no weighting applied when this
                           is None.
        early_stop_dev -- int, optional argument. How many iterations to wait for the dev
                          loss to improve before breaking. If this is activated, then we
                          save the model that generates the best prediction dev loss even
                          if the loss went up later. If this is activated, training stops
                          early if the dev loss is not going down anymore. Note that this
                          quantities number of times we measure the dev loss, i.e.,
                          early_stop_dev * FLAGS['eval_every'] iterations. If this is
                          zero, then it is deactivated. If it is None, read from FLAGS
        update_stats -- bool, optional argument. Whether to normalize features and update
                        the value of mean and std of features given this training set. By
                        default is True.        
        downsample_devloss -- optional argument, controls whether and how much to
                            subsample the dev set to calculate losses. If the dev set
                            is very big, calculating the full loss can be slow, so you
                            can set this parameter to only use part of it. If this is
                            less than 1, it indicates a ratio (0.1 means 10% of points
                            at random are used); if this is more than 1, it indicates abs
                            number (10000 means 10k points are used at random). None
                            deactivates subsampling, which is default behavior.
        detailed_losses -- optional argument, boolean that determines whether the output
                           to the screen, as the model is being trained, shows detailed 
                           information. By default, it is False. 
        description -- optional argument. String, containing he description of the model 
                       that is going to be saved to disk.
        path -- optional argument. String, containing the path on disk in which the model
                metadata is going to be saved using self.saveToDisk. This is the path that
                must be fed to the self.loadFromDisk function later to recover the trained
                model.
        
        Returns:
        best_dev_loss -- The best (prediction) loss throughout training in the dev set
        end_dev_loss -- The final (prediction) loss throughout training in the dev set
        step_list -- A list containing iteration numbers at which losses are returned
        train_loss_list -- A list containing training losses throughout training
        dev_loss_list -- A list containing dev losses throughout training
        """
        
        # Make sure flags are consistent with arguments passed in
        self.assertArguments()
                               
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

        # Save to disk before starting train so we can recover everything if the code
        # stops running in the middle of training
        if path is not None:
            self.saveToDisk(description=description, path=path)
        
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
                                   train_x_features, train_tensor_basis, b=train_b,
                                   loss_weight=train_loss_weight)
        
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
                    
                    losses = self.getTotalLosses(dev_x_features, dev_tensor_basis,
                                                 dev_b, dev_loss_weight,
                                                 downsample=downsample_devloss, 
                                                 report_real=detailed_losses)          
                    loss_dev, loss_dev_pred, loss_dev_reg, ratio_real = losses
                    
                    if detailed_losses:
                        print(" Exp Train: {:.3f} | Dev: {:.3f}".format(exp_loss, loss_dev)
                              + " ({:.3f}% non-realizable matrices)".format(100*ratio_real),
                              flush=True)
                        print("Dev breakdown -> predict: {:.3f} |".format(loss_dev_pred)
                              + " regularize: {:.3f} |".format(loss_dev_reg), flush=True) 
                    else:
                        print(" Exp Train: {:.3f} | Dev: {:.3f}".format(exp_loss, loss_dev),
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
                    
                    print("", flush=True) # Print empty line after an evaluation round
                    
                batch = batch_gen.nextBatch()
                
            toc = timeit.default_timer()
            print("---------Epoch {} took {:.2f}s".format(ep,toc-tic), flush=True)

            if to_break:
                print("(***) Dev loss not changing... Training will stop early.", 
                      flush=True)
                break                
        
        # Calculate last dev loss 
        _, end_dev_loss, _, _ = self.getTotalLosses(dev_x_features, 
                                                    dev_tensor_basis, 
                                                    dev_b, dev_loss_weight,                                                          
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
           
    
    def getRansLoss(self, b, gradu, tke, eddy_visc, loss_weight=None):
        """
        This function provides a baseline loss from the linear eddy viscosity model.
        
        Arguments:
        b -- numpy array, shape (n_points, 3, 3) containing the true (LES) anisotropy
             tensor
        gradu -- numpy array, shape (n_points, 3, 3) containing the gradient of the 
                 mean velocity
        tke -- numpy array, shape (n_points,) containing the turbulent kinetic energy 
             (with units of m^2/s^2) from RANS calculation
        eddy_visc -- numpy array, shape (n_points,) containing the eddy viscosity
                     (with units of m^2/s) from RANS calculation
        loss_weight -- numpy array of shape (num_points) or (num_points,1,1) or 
                       (num_points,3,3). Optional, only needed when a specific weight
                       is desired for the loss       
               
        Returns:
        loss_pred -- the prediction loss from using the linear eddy viscosity model        
        """        
        
        # b_rans calculated with LEVM        
        Sij = 0.5*(gradu + np.transpose(gradu, (0,2,1)))
        b_rans = -1.0 * np.reshape(eddy_visc/tke, (-1,1,1)) * Sij
        
        # return appropriate loss here
        if self.FLAGS['loss_type'] == 'l2':
            loss_pred = momentum_losses.lossL2(b, b_rans, loss_weight)
        if self.FLAGS['loss_type'] == 'l1':
            loss_pred = momentum_losses.lossL1(b, b_rans, loss_weight)
        
        return loss_pred
        
    
    def checkFlags(self):
        """
        This function checks some key flags in the dictionary self.FLAGS to make
        sure they are valid. If they have not been passed in, this also sets them
        to default values.
        """
        
        # List of all properties that have to be non-negative
        list_keys = ['num_basis', 'num_features', 'num_neurons', 'num_epochs',
                     'early_stop_dev', 'train_batch_size', 'eval_every', 'learning_rate',
                     'c_reg']
        list_defaults = [constants.NUM_BASIS, constants.NUM_FEATURES, 
                         constants.NUM_NEURONS, constants.NUM_EPOCHS, 
                         constants.EARLY_STOP_DEV, constants.TRAIN_BATCH_SIZE, 
                         constants.EVAL_EVERY, constants.LEARNING_RATE, constants.C_REG]       
        for key, default in zip(list_keys, list_defaults):
            if key in self.FLAGS:                
                assert self.FLAGS[key] >= 0, "FLAGS['{}'] can't be negative!".format(key)
            else:                
                self.FLAGS[key] = default
        
        # Check num layers, which can be -1, 0, or positive
        list_keys = ['num_layers',]
        list_defaults = [constants.NUM_LAYERS,]        
        for key, default in zip(list_keys, list_defaults):
            if key in self.FLAGS:                
                assert self.FLAGS[key] == -1 or self.FLAGS[key] >= 0, \
                   "FLAGS['{}'] is invalid! It should either be -1, 0, or >0".format(key)
            else:                
                self.FLAGS[key] = default      
        
        # List of all properties that must be True or False
        list_keys = ['enforce_prt',]
        list_defaults = [constants.ENFORCE_PRT,]
        for key, default in zip(list_keys, list_defaults):
            if key in self.FLAGS:
                assert self.FLAGS[key] is True or self.FLAGS[key] is False, \
                            "FLAGS['{}'] must be True or False!".format(key)
            else:
                self.FLAGS[key] = default        
                
        # Check if a loss type has been passed, if not use default
        if 'loss_type' in self.FLAGS:
            assert self.FLAGS['loss_type'] == 'l2' or \
                   self.FLAGS['loss_type'] == 'l1', "FLAGS['loss_type'] is not valid!"
        else:            
            self.FLAGS['loss_type'] = constants.LOSS_TYPE_TBNN
            
        # Check dropout rate
        if 'drop_prob' in self.FLAGS:
            assert self.FLAGS['drop_prob'] >= 0 and self.FLAGS['drop_prob'] <= 1, \
                  "FLAGS['drop_prob'] must be between 0 and 1"
        else:
            self.FLAGS['drop_prob'] = constants.DROP_PROB       
    
        
    def assertArguments(self):
        """
        This function makes sure that loss_weight passed in is
        appropriate to the flags we have, i.e., they are only None if they are
        not needed.
        
        Arguments:        
        """        
        pass
        
    
    def getAnisotropy(self, batch):
        """
        This runs a single forward pass to obtain the (dimensionless) anisotropy matrix.
        
        Inputs:        
        batch -- a Batch object containing information necessary for testing         

        Returns:
        b -- the diffusivity tensor for this batch, a numpy array of shape (None,3,3)
        g -- the coefficient multiplying each tensor basis, a numpy array of 
             shape (None,num_basis)        
        """
        
        input_feed = {}
        input_feed[self.x_features] = batch.x_features        
        input_feed[self.tensor_basis] = batch.tensor_basis       
                        
        # output_feed contains the things we want to fetch.
        output_feed = [self.b_predicted, self.g]
        
        # Run the model
        [b, g] = self._tfsession.run(output_feed, input_feed)
        
        return b, g
        
    
    def getTotalAnisotropy(self, test_x_features, test_tensor_basis, 
                           normalize=True, clean=True, b_default_rans=None):
        """
        This method takes in a whole test set and computes the anisotropy tensor on it.
        
        Inputs:        
        test_x_features -- numpy array containing the features in the whole dataset, of
                           shape (num_points, num_features)
        test_tensor_basis -- numpy array containing the tensor basis in the whole
                             dataset, of shape (num_points, num_basis, 3, 3)        
        normalize -- optional argument, boolean flag saying whether to normalize the 
                     features before feeding them to the neural network. True by default.
        clean -- optional argument, whether to clean the output diffusivity according
                 to the function defined in utils.py. True by default.
        b_default_rans -- numpy array containing the default anisotropy tensor in RANS.
                          Optional, this is only used to clean the anisotropy. Shape is
                          (num_points, 3, 3)
        
        Returns:
        total_b -- dimensionless diffusivity predicted, numpy array of shape 
                      (num_points, 3, 3)
        total_g -- coefficients that multiply each of the tensor basis predicted, a
                   numpy array of shape (num_points, num_basis)        
        """
    
        num_points = test_x_features.shape[0]
        total_b = np.empty((num_points, 3, 3))
        total_g = np.empty((num_points, self.FLAGS['num_basis']))        
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
            total_b[i:i+n_batch], total_g[i:i+n_batch] = self.getAnisotropy(batch)                       
            i += n_batch
            batch = batch_gen.nextBatch()
        
        # Clean the resulting diffusivity
        if clean:
            total_b = utils.cleanAnisotropy(total_b, test_x_features, 
                                            b_default_rans=b_default_rans)
        
        return total_b, total_g
    
            
    