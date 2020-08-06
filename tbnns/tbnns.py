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
from tbnns import constants, utils, nn, scalar_losses


class TBNNS(nn.NNBasic):
    """
    This class contains definitions and methods needed for the TBNN-s class. This
    inherits from the NNBasic class, which defines the skeleton of the class.
    """    
    
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
        self.prt_desired -- placeholder for Pr_t which is enforced exactly in
                            the predicted diffusivity when FLAGS['enforce_prt'] is True
        self.drop_prob -- placeholder for dropout probability        
        """
        
        self.x_features = tf.compat.v1.placeholder(tf.float32, 
                                     shape=[None, self.FLAGS['num_features']])
        self.tensor_basis = tf.compat.v1.placeholder(tf.float32, 
                                          shape=[None, self.FLAGS['num_basis'], 3, 3])
        self.uc = tf.compat.v1.placeholder(tf.float32, shape=[None, 3])
        self.gradc = tf.compat.v1.placeholder(tf.float32, shape=[None, 3])
        self.eddy_visc = tf.compat.v1.placeholder(tf.float32, shape=[None])
        self.loss_weight = tf.compat.v1.placeholder(tf.float32, shape=[None, None])
        self.prt_desired = tf.compat.v1.placeholder(tf.float32, shape=[None])
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
        Uses the coefficients g to calculate the diffusivity and the u'c'
        
        Defines:
        self.diffusivity -- diffusivity tensor, shape (None,3,3)
        self.uc_predicted -- predicted value of u'c', shape (None,3)        
        """
        
        with tf.compat.v1.variable_scope("bases"):        
            # shape of [None,num_bases,3,3]    
            mult_bas = tf.multiply(tf.reshape(self.g,shape=[-1,self.FLAGS['num_basis'],1,1]), 
                                   self.tensor_basis) 
            
            # diffusivity matrix, shape [None, 3, 3]
            self.diffusivity = tf.reduce_sum(mult_bas, axis=1)
            
            # shape of [None,3,1]
            gradc_ext = tf.expand_dims(self.gradc, -1) 
            
            # Here, we enforce a given pr_t to the diffusivity matrix (this is passed
            # in as gamma, potentially provided by a Random Forest model)
            if self.FLAGS['enforce_prt']:
                uc = tf.squeeze(tf.matmul(self.diffusivity, gradc_ext))
                gamma_implied = ( tf.reduce_sum(self.gradc*uc, axis=1) / 
                                  tf.reduce_sum(self.gradc*self.gradc, axis=1) )                
                
                # clip gamma_implied
                gamma_implied = tf.maximum(gamma_implied, constants.GAMMA_MIN)
                gamma_implied = tf.minimum(gamma_implied, 1.0/constants.GAMMA_MIN)
                
                # Edited diffusivity matrix, shape [None, 3, 3]
                gamma_desired = 1.0/self.prt_desired
                factor = tf.reshape(gamma_desired/gamma_implied, shape=[-1,1,1])                
                diff_edited = self.diffusivity * factor
                
                # shape of [None, 3], full u'c' vector
                self.uc_predicted = -1.0*( tf.expand_dims(self.eddy_visc,-1) *
                                     tf.squeeze(tf.matmul(diff_edited, gradc_ext)) )
            else:
                # shape of [None, 3], full u'c' vector
                self.uc_predicted = -1.0*( tf.expand_dims(self.eddy_visc,-1) *
                                     tf.squeeze(tf.matmul(self.diffusivity, gradc_ext)) )
        

    def constructLoss(self):
        """
        Add loss computation to the graph. 

        Defines:
        self.loss_pred -- scalar with the loss due to error between uc_predicted and uc
        self.loss_reg -- scalar with the regularization loss
        self.loss_psd -- scalar with the component of the loss that penalizes non-PSD
                         diffusivity matrices, J_PSD
        self.loss_prt -- scalar with the component of the loss that tries to match
                         implied Pr_t with the LES-extracted Pr_t
        self.loss_neg -- scalar with the component of the loss that penalizes magnitude
                         of the turbulent diffusivity matrix in regions of counter
                         gradient diffusion
        self.loss -- scalar with the total loss (sum of all components), which is
                     what gradient descent tries to minimize
        """
        
        with tf.compat.v1.variable_scope("losses"):
            
            # Calculate the prediction loss (i.e., how bad the predicted u'c' is)
            if self.FLAGS['loss_type'] == 'log':
                self.loss_pred = scalar_losses.lossLog(self.uc, self.uc_predicted, 
                                                       tf_flag=True)            
            if self.FLAGS['loss_type'] == 'l2':
                self.loss_pred = scalar_losses.lossL2(self.uc, self.uc_predicted,
                                                      self.loss_weight, tf_flag=True)
            if self.FLAGS['loss_type'] == 'l1':
                self.loss_pred = scalar_losses.lossL1(self.uc, self.uc_predicted,
                                                      self.loss_weight, tf_flag=True)
            if self.FLAGS['loss_type'] == 'l2k':
                self.loss_pred = scalar_losses.lossL2k(self.uc, self.uc_predicted, 
                                                       self.loss_weight, tf_flag=True)
            if self.FLAGS['loss_type'] == 'cos':
                self.loss_pred = scalar_losses.lossCos(self.uc, self.uc_predicted,
                                                       tf_flag=True)            
            
            # Calculate the L2 regularization component of the loss            
            if self.FLAGS['c_reg'] == 0: self.loss_reg = tf.constant(0.0)
            else:
                vars = tf.compat.v1.trainable_variables()
                self.loss_reg = \
                     tf.add_n([tf.nn.l2_loss(v) for v in vars if ('bias' not in v.name)]) 
            
            # Calculate J_PSD, the loss that enforces predicted diffusivity is PSD            
            if self.FLAGS['c_psd'] == 0: self.loss_psd = tf.constant(0.0)
            else:
                diff_sym = 0.5*(self.diffusivity + 
                                tf.linalg.matrix_transpose(self.diffusivity))
                                
                # e contains eigenvalues of symmetric part, in non-decreasing order
                e, _ = tf.linalg.eigh(diff_sym) 
                self.loss_psd = tf.reduce_mean(tf.maximum(-tf.reduce_min(e,axis=1),0))               
            
            # Calculate J_PRT, the loss that tries to match implied Pr_t with LES Pr_t
            if self.FLAGS['c_prt'] == 0: self.loss_prt = tf.constant(0.0)
            else:
                log_gamma_implied = utils.calculateLogGamma(self.uc_predicted,
                                                            self.gradc, self.eddy_visc,
                                                            tf_flag=True)
                log_gamma_les = utils.calculateLogGamma(self.uc, self.gradc, 
                                                        self.eddy_visc, tf_flag=True)
                self.loss_prt = tf.reduce_mean((log_gamma_implied-log_gamma_les)**2)                
            
            # Calculate J_NEG, the loss that penalizes the diffusivity magnitude in 
            # regions of negative diffusivity
            if self.FLAGS['c_neg'] == 0: self.loss_neg = tf.constant(0.0)            
            else:
                # appropriate norm of the diffusivity matrix
                diff_norm = tf.norm(self.diffusivity, ord=2, axis=[1,2])

                # flag that indicates if the given points have counter gradient transport
                ctr_grad_flag = tf.math.greater_equal(tf.reduce_sum(self.uc*self.gradc,
                                                                    axis=1), 0)                
                # take the mean of the diffusivity norm in locations where there is 
                # counter gradient transport. tf.cond is used to return zero if no points
                # contain counter gradient transport
                self.loss_neg = tf.cond(tf.reduce_any(ctr_grad_flag), 
                                  lambda: tf.reduce_mean(tf.boolean_mask(diff_norm,
                                                                         ctr_grad_flag)),
                                  lambda: tf.constant(0.0) )                
            
            # Loss is the sum of different components
            self.loss = (self.loss_pred 
                         + self.FLAGS['c_reg']*self.loss_reg
                         + self.FLAGS['c_psd']*self.loss_psd
                         + self.FLAGS['c_prt']*self.loss_prt
                         + self.FLAGS['c_neg']*self.loss_neg)
      
    
    def getLoss(self, batch):
        """
        This runs a single forward pass and obtains the loss.
        
        Inputs:        
        batch -- a Batch object containing information necessary for training         
        
        Returns:
        loss -- the loss (averaged across the batch) for this batch.
        loss_pred -- the loss just due to the error between u'c' and u'c'_predicted
        loss_reg -- the regularization component of the loss
        loss_psd -- component of the loss that penalizes non-PSD
                         diffusivity matrices, J_PSD
        loss_prt -- component of the loss that tries to match
                         implied Pr_t with the LES-extracted Pr_t
        loss_neg -- component of the loss that penalizes magnitude
                         of the turbulent diffusivity matrix in regions of counter
                         gradient diffusion 
        """
        
        input_feed = {}
        input_feed[self.x_features] = batch.x_features
        input_feed[self.tensor_basis] = batch.tensor_basis
        input_feed[self.uc] = batch.uc
        input_feed[self.gradc] = batch.gradc        
        input_feed[self.eddy_visc] = batch.eddy_visc
        
        if batch.loss_weight is not None:
            input_feed[self.loss_weight] = batch.loss_weight        
        if batch.prt_desired is not None: 
            input_feed[self.prt_desired] = batch.prt_desired
                                
        # output_feed contains the things we want to fetch.
        output_feed = [self.loss, self.loss_pred, self.loss_reg, self.loss_psd,
                       self.loss_prt, self.loss_neg]        
        
        # Run the model
        [loss, loss_pred, loss_reg, loss_psd,  
        loss_prt, loss_neg] = self._tfsession.run(output_feed, input_feed)

        return loss, loss_pred, loss_reg, loss_psd, loss_prt, loss_neg
       
    
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
        if batch.prt_desired is not None: 
            input_feed[self.prt_desired] = batch.prt_desired
                                    
        # output_feed contains the things we want to fetch.
        output_feed = [self.updates, self.loss, self.global_step]

        # Run the model
        [_, loss, global_step] = self._tfsession.run(output_feed, input_feed)

        return loss, global_step
    
    
    def getTotalLosses(self, x_features, tensor_basis, uc, gradc, eddy_visc,
                       loss_weight=None, prt_desired=None, normalize=True, 
                       downsample=None, report_psd=False):
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
        loss_weight -- numpy array of shape (num_points) or (num_points, 1) or 
                       (num_points, 3). This weights each point and possibly each 
                       component differently when assessing the predicted turbulent
                       scalar flux. Optional, only needed for specific loss types.
        prt_desired -- numpy array of shape(num_points) containing the desired Pr_t to
                       enforce exactly at each point when FLAGS['enforce_prt']=True.
                       Optional, only required when FLAGS['enforce_prt']=True.
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
        total_loss_psd -- a scalar, the average PSD diffusivity loss for the whole
                          dataset
        total_loss_prt -- a scalar, the average Pr_t diffusivity loss for the whole
                          dataset
        total_loss_neg -- a scalar, the average negative diffusivity loss for the whole
                          dataset
        total_loss_gamma -- a scalar, the average gamma loss for the whole dataset
        ratio_eig -- a scalar, the ratio of non-PSD matrices in the output of the model.
                     This is only calculated if report_psd==True; otherwise, it just 
                     contains None.
        """
        
        # Make sure it is only None if appropriate flags are set
        self.assertArguments(loss_weight, prt_desired)

        # Initializes quantities we need to keep track of
        total_loss = 0
        total_loss_pred = 0
        total_loss_reg = 0
        total_loss_psd = 0
        total_loss_prt = 0
        total_loss_neg = 0        
        num_points = x_features.shape[0]
        
        # This normalizes the inputs. Runs when normalize = True
        if self.features_mean is not None and normalize:
            x_features = (x_features - self.features_mean)/self.features_std        
        
        # Initialize batch generator, downsampling only if not None        
        idx = utils.downsampleIdx(num_points, downsample)
        if loss_weight is not None: loss_weight = loss_weight[idx]
        if prt_desired is not None: prt_desired = prt_desired[idx]
        batch_gen = BatchGenerator(constants.TEST_BATCH_SIZE, x_features[idx,:], 
                                   tensor_basis[idx,:,:,:], uc=uc[idx,:], 
                                   gradc=gradc[idx,:], eddy_visc=eddy_visc[idx], 
                                   loss_weight=loss_weight, prt_desired=prt_desired)             
        
        # Iterate through all batches of data
        batch = batch_gen.nextBatch()        
        while batch is not None:
            l, l_pred, l_reg, l_psd, l_prt, l_neg = self.getLoss(batch)
            total_loss += l * batch.x_features.shape[0]
            total_loss_pred += l_pred * batch.x_features.shape[0]
            total_loss_reg += l_reg * batch.x_features.shape[0]
            total_loss_psd += l_psd * batch.x_features.shape[0]
            total_loss_prt += l_prt * batch.x_features.shape[0]
            total_loss_neg += l_neg * batch.x_features.shape[0]            
            batch = batch_gen.nextBatch()
        
        # To get an average loss, divide by total number of dev points
        total_loss =  total_loss/num_points
        total_loss_pred = total_loss_pred/num_points
        total_loss_reg = total_loss_reg/num_points
        total_loss_psd = total_loss_psd/num_points
        total_loss_prt = total_loss_prt/num_points
        total_loss_neg = total_loss_neg/num_points                
        
        # Report the number of non-PSD matrices
        if report_psd:
            diff, _, = self.getTotalDiffusivity(x_features, tensor_basis,                                                
                                                normalize=False, clean=False)
            diff_sym = 0.5*(diff+np.transpose(diff,axes=(0,2,1)))
            eig_all, _ = np.linalg.eigh(diff_sym)
            eig_min = np.amin(eig_all, axis=1)
            ratio_eig = np.sum(eig_min < 0) / eig_min.shape[0]
        else: ratio_eig = None           
        
        return (total_loss, total_loss_pred, total_loss_reg, total_loss_psd,
                total_loss_prt, total_loss_neg, ratio_eig)
        
    
    def train(self, path_to_saver,
              train_x_features, train_tensor_basis, train_uc, train_gradc, train_eddy_visc,
              dev_x_features, dev_tensor_basis, dev_uc, dev_gradc, dev_eddy_visc, 
              train_loss_weight=None, dev_loss_weight=None,
              train_prt_desired=None, dev_prt_desired=None,
              early_stop_dev=None, update_stats=True, 
              downsample_devloss=None, detailed_losses=False,
              path=None, description=None):
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
                             prediction losses. Shape (num_train) or (num_train, 1) or
                             (num_train, 3).
        dev_loss_weight -- optional argument, numpy array containing the loss_weight
                           for the dev data, which is used in some types of 
                           prediction losses. Shape (num_dev) or (num_dev, 1) or 
                           (num_dev, 3)        
        train_prt_desired -- numpy array of shape (num_train) containing the desired Pr_t
                             to enforce exactly in the training data when 
                             FLAGS['enforce_prt']=True. Optional, only required when 
                             FLAGS['enforce_prt']=True.
        dev_prt_desired -- numpy array of shape (num_dev) containing the desired Pr_t
                           to enforce exactly in the dev set when 
                           FLAGS['enforce_prt']=True. Optional, only required when 
                           FLAGS['enforce_prt']=True.
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
        path -- optional argument. String, containing the path on disk in which the model
                metadata is going to be saved using self.saveToDisk. This is the path that
                must be fed to the self.loadFromDisk function later to recover the trained
                model.
        description -- optional argument. String, containing he description of the model 
                       that is going to be saved to disk.
        
        Returns:
        best_dev_loss -- The best (prediction) loss throughout training in the dev set
        end_dev_loss -- The final (prediction) loss throughout training in the dev set
        step_list -- A list containing iteration numbers at which losses are returned
        train_loss_list -- A list containing training losses throughout training
        dev_loss_list -- A list containing dev losses throughout training
        """
        
        # Make sure it is only None if appropriate flags are set
        self.assertArguments(train_loss_weight, train_prt_desired)
        self.assertArguments(dev_loss_weight, dev_prt_desired)
                       
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
                                   train_x_features, train_tensor_basis, uc=train_uc,
                                   gradc=train_gradc, eddy_visc=train_eddy_visc,
                                   loss_weight=train_loss_weight, 
                                   prt_desired=train_prt_desired)
        
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
                                                 dev_uc, dev_gradc, dev_eddy_visc,
                                                 dev_loss_weight, dev_prt_desired,
                                                 downsample=downsample_devloss,
                                                 report_psd=detailed_losses)          
                    (loss_dev, loss_dev_pred, loss_dev_reg, loss_dev_psd,
                     loss_dev_prt, loss_dev_neg, ratio_psd) = losses
                    
                    if detailed_losses:
                        print(" Exp Train: {:.3f} | Dev: {:.3f}".format(exp_loss, loss_dev)
                              + " ({:.3f}% non-PSD matrices)".format(100*ratio_psd), flush=True)
                        print("Dev breakdown -> predict: {:.3f} |".format(loss_dev_pred)
                              + " regularize: {:.3f} |".format(loss_dev_reg)
                              + " PSD: {:g} |".format(loss_dev_psd)
                              + " PRT: {:g} |".format(loss_dev_prt)
                              + " NEG: {:g}".format(loss_dev_neg), flush=True) 
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
        _, end_dev_loss, _, _, _, _, _ = self.getTotalLosses(dev_x_features, 
                                                          dev_tensor_basis, 
                                                          dev_uc, dev_gradc,
                                                          dev_eddy_visc, dev_loss_weight,
                                                          dev_prt_desired,
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
        loss_pred -- the prediction loss from using the fixed Pr_t assumption
        loss_prt -- the baseline Pr_t loss (J_PRT) with GDH and fixed Pr_t
        loss_neg -- the baseline negative diffusivity loss (J_NEG) with GDH and fixed Pr_t
        """
        
        # Calculates log(gamma) based on quantities received
        log_gamma = utils.calculateLogGamma(uc, gradc, eddy_visc)
        
        # uc_rans calculated with fixed Pr_t
        if prt_default is None:
            prt_default = constants.PRT_DEFAULT        
        uc_rans = -1.0 * (np.expand_dims(eddy_visc/prt_default, 1)) * gradc
        
        # return appropriate loss here        
        if self.FLAGS['loss_type'] == 'log':
            loss_pred = scalar_losses.lossLog(uc, uc_rans)        
        if self.FLAGS['loss_type'] == 'l2':
            loss_pred = scalar_losses.lossL2(uc, uc_rans, loss_weight)
        if self.FLAGS['loss_type'] == 'l1':
            loss_pred = scalar_losses.lossL1(uc, uc_rans, loss_weight)
        if self.FLAGS['loss_type'] == 'l2k':
            loss_pred = scalar_losses.lossL2k(uc, uc_rans, loss_weight)
        if self.FLAGS['loss_type'] == 'cos':
            loss_pred = scalar_losses.lossCos(uc, uc_rans)
        
        # pr_t loss
        loss_prt = np.mean((log_gamma-np.log(1.0/prt_default))**2)
        
        # negative diffusivity loss
        loss_neg = 1.0/prt_default              
        
        return loss_pred, loss_prt, loss_neg
    
            
    def checkFlags(self):
        """
        This function checks some key flags in the dictionary self.FLAGS to make
        sure they are valid. If they have not been passed in, this also sets them
        to default values.
        """
        
        # List of all properties that have to be non-negative
        list_keys = ['num_basis', 'num_features', 'num_neurons', 'num_epochs',
                     'early_stop_dev', 'train_batch_size', 'eval_every', 'learning_rate',
                     'c_reg', 'c_psd', 'c_prt', 'c_neg']
        list_defaults = [constants.NUM_BASIS, constants.NUM_FEATURES, 
                         constants.NUM_NEURONS, constants.NUM_EPOCHS, 
                         constants.EARLY_STOP_DEV, constants.TRAIN_BATCH_SIZE, 
                         constants.EVAL_EVERY, constants.LEARNING_RATE, constants.C_REG,
                         constants.C_PSD, constants.C_PRT, constants.C_NEG]        
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
            assert self.FLAGS['loss_type'] == 'log' or \
                   self.FLAGS['loss_type'] == 'l2' or \
                   self.FLAGS['loss_type'] == 'l1' or \
                   self.FLAGS['loss_type'] == 'l2k' or \
                   self.FLAGS['loss_type'] == 'cos', "FLAGS['loss_type'] is not valid!"
        else:            
            self.FLAGS['loss_type'] = constants.LOSS_TYPE
            
        # Check dropout rate
        if 'drop_prob' in self.FLAGS:
            assert self.FLAGS['drop_prob'] >= 0 and self.FLAGS['drop_prob'] <= 1, \
                  "FLAGS['drop_prob'] must be between 0 and 1"
        else:
            self.FLAGS['drop_prob'] = constants.DROP_PROB

        # Correct some values: c_psd and c_reg must be zero if constant g model is fitted
        if self.FLAGS['num_layers'] == -1:
            self.FLAGS['c_psd'] = 0
            self.FLAGS['c_reg'] = 0
            
        
    def assertArguments(self, loss_weight, prt_desired):
        """
        This function makes sure that loss_weight passed in is
        appropriate to the flags we have, i.e., they are only None if they are
        not needed.
        
        Arguments:
        loss_weight -- either None or a numpy array of shape (num_points) or 
                       (num_points, 1) or (num_points, 3). This weights each point and
                       possibly each component differently when assessing the predicted
                       turbulent scalar flux. This function makes sure the quantity is
                       not None when we need it.
        prt_desired -- either None or numpy array of shape(num_points) containing the
                       desired Pr_t to enforce exactly at each point when 
                       FLAGS['enforce_prt']=True. This function makes sure the quantity
                       is not None when we need it.
        """
        
        # Check to see if we have all we need for the current configuration                   
        if self.FLAGS['loss_type'] == 'l2' or self.FLAGS['loss_type'] == 'l2k' or \
           self.FLAGS['loss_type'] == 'l1':
            msg = "loss_weight cannot be None since loss_type=l2 or l2k or l1"
            assert loss_weight is not None, msg
            msg = "loss_weight must be always positive!"
            assert (loss_weight > 0).all(), msg

        # Check to see if we have all we need for the current configuration                   
        if self.FLAGS['enforce_prt'] == True:
            msg = "prt_desired cannot be None since enforce_prt=True"
            assert prt_desired is not None, msg
            msg = "prt_desired must be always positive!"
            assert (prt_desired > 0).all(), msg
            
    
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
        
        
    def getTotalDiffusivity(self, test_x_features, test_tensor_basis, 
                            normalize=True, clean=True, bump_diffusivity=True, 
                            n_std=None, prt_default=None, gamma_min=None):
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
        bump_diffusivity -- bool, optional argument. This decides whether we bump the
                            diagonal elements of the matrix in case the eigenvalues are 
                            positive but small. This helps with stability, so it's True
                            by default.
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
            total_diff[i:i+n_batch], total_g[i:i+n_batch] = self.getDiffusivity(batch)                       
            i += n_batch
            batch = batch_gen.nextBatch()
        
        # Clean the resulting diffusivity
        if clean:
            total_diff, total_g = utils.cleanDiffusivity(total_diff, total_g, 
                                                         test_x_features, n_std,
                                                         prt_default, gamma_min,
                                                         bump_diff=bump_diffusivity)        
        return total_diff, total_g  