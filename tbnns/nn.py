#--------------------------------- nn.py file ---------------------------------------#
"""
This file contains the definition of the NNBasic class, which is a building block for
either TBNN or TBNN-s classes. It also contains some basic neural network components
previously in a layers.py file: FullyConnected layer.
"""

# ------------ Import statements
import tensorflow as tf
import numpy as np
import joblib
from tbnns import constants, utils

# Run this to suppress warnings/info messages from tensorflow
utils.suppressWarnings()


class NNBasic:
    """
    This super-class contains definitions and methods needed for a basic NN class
    used by both TBNN and TBNN-s classes
    """
    
    def __init__(self):
        """
        Constructor class that initializes the model. Sets instance variables to None.

        Defines:
        self._tfsession -- instance variable holding the current tensorflow session
        self.FLAGS -- Python dictionary containing hyper-paremeter settings for the model
        self.features_mean -- numpy array of shape (num_features,1) containing the mean
                              of the features used to train the model
        self.features_std -- numpy array of shape (num_features,1) containing the 
                             standard deviation of the features used to train the model
        self.saver_path -- staring containing the file location where the tf.Saver class
                           is saved to disk
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
            print("Model loaded successfully!")           
            self.printModelInfo()
        
        return description
    
        
    def printModelInfo(self, print_values=False):
        """
        Call this function to print all flags and trainable parameters of the model.
        
        Arguments:
        print_values -- optional, this bool tells the function whether the value
                        of the trained parameters should be printed. By default it is 
                        False.
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
            if print_values:
                v_str = np.array_str(v.eval(self._tfsession), precision=4, 
                                     suppress_small=True)
                v_str = v_str.replace("\n", "\n\t ")
                print("\t {}".format(v_str))
    
    
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
        tf.compat.v1.reset_default_graph()
        
        # Builds the graph by calling appropriate function
        self.constructPlaceholders()
        self.constructNet()       
        self.combineBasis()
        self.constructLoss()
                
        # Define optimizer and updates
        # (updates is what you need to fetch in session.run to do a gradient update)
        self.global_step = tf.Variable(0, name="global_step", trainable=False)                
        opt = tf.compat.v1.train.AdamOptimizer(learning_rate=self.FLAGS['learning_rate'])
        self.updates = opt.minimize(self.loss, global_step=self.global_step)
        
        # Define savers (for checkpointing)
        self._saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(),
                                               max_to_keep=1)
        
        # Creates session and initializes global variables
        self._tfsession = tf.compat.v1.Session()
        self._tfsession.run(tf.compat.v1.global_variables_initializer())
        
    
    def constructPlaceholders(self):
        """
        Adds all placeholders for external data into the model  
        """        
        print("self.constructPlaceholders not implemented yet!")
        pass
                    
    
    def constructNet(self):    
        """
        Creates the neural network that predicts g in the model
        """
        print("self.constructNet not implemented yet!")
        pass
        
    
    def combineBasis(self):
        """
        Uses the coefficients g to calculate the eventual quantity of interest
        """        
        print("self.combineBasis not implemented yet!")
        pass      
        

    def constructLoss(self):
        """
        Add loss computation to the graph
        """
        print("self.constructLoss not implemented yet!")
        pass 
        
        
    def getLoss(self):
        """
        This runs a single forward pass and obtains the loss.
        """
        print("self.getLoss not implemented yet!")
        pass
            

    def runTrainIter(self):
        """
        This performs a single training iteration (forward pass, loss computation,
        backprop, parameter update)   
        """
        print("self.runTrainIter not implemented yet!")
        pass
        
        
    def getTotalLosses(self):
        """
        This method takes in a whole dataset and computes the average loss on it.
        """
        print("self.getTotalLosses not implemented yet!")
        pass    
    

    def train(self):
        """
        This method trains the model by running runTrainIter across the whole data.        
        """
        print("self.train not implemented yet!")
        pass
    
    
    def getRansLoss(self):
        """
        This function provides a baseline loss from a typical RANS assumption.
        """
        print("self.getRansLoss not implemented yet!")
        pass  
    
            
    def checkFlags(self):
        """
        This function checks some key flags in the dictionary self.FLAGS to make
        sure they are valid. If they have not been passed in, this also sets them
        to default values.
        """ 
        print("self.checkFlags not implemented yet!")
        pass   
            
        
    def assertArguments(self):
        """
        This function makes sure that loss_weight passed in is
        appropriate to the flags we have, i.e., they are only None if they are
        not needed.
        """
        print("self.assertArguments not implemented yet!")
        pass


class FullyConnected(object):
    """
    Simple fully connected layer. To use:
    
    fc = FullyConnected(output_size, drop_prob, relu, name)
    x2 = fc.build(x1) # x1 is the input tensor, x2 is the output tensor
    """

    def __init__(self, out_size, drop_prob, relu=True, name=""):
        """
        Constructor method, instantiates the class with key properties

        Arguments:
        out_size -- integer, number of neurons in the output layer
        drop_prob -- float, drop probability to use for dropout (0 deactivates it)
        relu -- optional argument, whether to use a relu activation function. If False,
                use a linear activation
        name -- string, add to the scope of the variables defined here
        """
           
        self.out_size = out_size
        self.drop_prob = drop_prob
        self.relu = relu
        self.name = name # this is used to create different variable contexts
        
    def build(self, layer_inputs):
        """
        Takes in the layer inputs and produces the output tensor          

        Arguments:
        layer_inputs -- tensor of inputs, shape [None, previous_layer_size]
        
        Returns:
        out -- tensor of outputs, after fully connected + relu + dropout is applied,
               shape [None, out_size]
        """
        with tf.compat.v1.variable_scope("FC_"+self.name):
        
            out = tf.contrib.layers.fully_connected(layer_inputs, self.out_size, 
                                                    activation_fn=None)
            if self.relu:                
                out = tf.nn.relu(out)
                out = tf.nn.dropout(out, rate=self.drop_prob) # Apply dropout

            return out
