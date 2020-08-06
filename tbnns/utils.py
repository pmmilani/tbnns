#--------------------------------- utils.py file ---------------------------------------#
"""
This file contains utility functions and classes that support the TBNN-s class.
This includes cleaning and processing functions.
"""

# ------------ Import statements
import os
import timeit
import numpy as np
from tbnns import constants
import tensorflow as tf
       
            
def downsampleIdx(n_total, downsample):
    """
    Produces a set of indices to index into a numpy array and shuffle/downsample it.
    
    Arguments:
    n_total -- int, total size of the array in that dimensionalization
    downsample -- number that controls how we downsample the data
                  before saving it to disk. If None, no downsampling or shuffling is 
                  done. If this number is more than 1, then it represents the number
                  of examples we want to save; if it is less than 1, it represents the
                  ratio of all training examples we want to save.    
                  
    Returns:
    idx -- numpy array of ints of size (n_take), which contains the indices
           that we are supposed to take for downsampling.    
    """
    
    idx_tot = np.arange(n_total)
    
    if downsample is None:
        return idx_tot
        
    np.random.shuffle(idx_tot)    
    assert downsample > 0, "downsample must be greater than 0!"    
    if int(downsample) > 1:
        n_take = int(downsample)            
        if n_take > n_total:
            print("Warning! This dataset has fewer than {} usable points. "
                  + "All of them will be taken.".format(n_take))
            n_take = n_total                 
    else:
        n_take = int(downsample * n_total)
        if n_take > n_total: n_take = n_total # catches bug where downsample = 1.1            
    
    idx = idx_tot[0:n_take]
    
    return idx 

            
def cleanDiffusivity(diff, g=None, test_inputs=None, n_std=None, 
                     prt_default=None, gamma_min=None, clip_elements=False,
                     bump_diff=True, verbose=True):
    """
    This function is called to post-process the diffusivity and g produced for stability
    
    Arguments:
    diff -- numpy array of shape (n_useful, 3, 3) containing the originally predicted 
            diffusivity tensor
    g -- numpy array of shape (n_useful, NUM_BASIS) containing the factor that multiplies
         each tensor basis to produce the tensor diffusivity. Optional, if None it is not
         used.
    test_inputs -- numpy array of shape (n_useful, NUM_FEATURES) containing the test 
                   point features that produced the diffusivity tensor we are dealing
                   with. Optional, if None it is not used.
    n_std -- float, number of standard deviations around the training mean that each
             test feature is allowed to be. Optional, if None, default value is read from 
             constants.py
    prt_default -- float, default value of turbulent Prandlt number used when cleaning
                   diffusivities that are outright negative. Optional, if None is passed,
                   default value is read from constants.py
    gamma_min -- float, minimum value of gamma = diffusivity/turbulent viscosity allowed.
                 Used to clean values that are positive but too close to zero. Optional, 
                 if None is passed, default value is read from constants.py
    clip_elements -- bool, optional argument. This decides whether we clip elements of
                     the matrix to make sure they are not too big or too small. By 
                     default, this is False, so no clipping is applied.
    bump_diff -- bool, optional argument. This decides whether we bump the
                 diagonal elements of the matrix in case the eigenvalues are 
                 positive but small. This helps with stability, so it's True by default.
    verbose -- optional argument, boolean that says whether we print some extra 
               information to the screen.
    
    Returns:
    diff -- numpy array of shape (n_useful, 3, 3) containing the post-processed 
            diffusivity tensor
    g -- numpy array of shape (n_useful, NUM_BASIS) containing the post-processed
         factor that multiplies each tensor basis to produce the tensor diffusivity.
         If g=None is the argument, then this is NOT returned.
    """
        
    # Get default values if None is passed
    if n_std is None:
        n_std = constants.N_STD
    if prt_default is None:
        prt_default = constants.PRT_DEFAULT
    if gamma_min is None:
        gamma_min = constants.GAMMA_MIN
    
    print("Cleaning predicted diffusivity... ", end="", flush=True)
    tic = timeit.default_timer()        
    
    # (1) Here, make sure that there is no input more or less than n_std away
    # from the mean. Since test_inputs was normalized, it should not be more
    # than n_std or less than -n_std if we don't want test points too different
    # from the training data. We are cleaning points that, by a very simple measure,
    # denote extrapolation from the training set
    mask_ext = np.zeros(diff.shape[0], dtype=bool)
    if test_inputs is not None:
        mask_ext = (np.amax(test_inputs,axis=1) > n_std) + \
                   (np.amin(test_inputs,axis=1) < -n_std)
    diff, g = applyMask(diff, g, mask_ext, prt_default)        
    num_ext = np.sum(mask_ext) # total number of entries affected by this step
    #--------------------------------------------------------------------   
    
    # (2) Here, make sure all matrix entries are bounded according to gamma_min.
    # Diagonal elements must be positive, between 1/gamma_min and gamma_min.
    # Off-diagonal can be negative, but must lie between -1/gamma_min and 1/gamma_min.
    num_clip = 0
    if clip_elements:
        for i in range(3):
            for j in range(3):
                if i == j: min = gamma_min
                else: min = -1.0/gamma_min
                max = 1.0/gamma_min
                diff, num = clipElements(diff, i, j, min, max)
                num_clip += num
    #--------------------------------------------------------------------                 
    
    # (3) Here, make sure that all matrices are positive-semidefinite. For a general real
    # matrix, this holds iff its symmetric part is positive semi-definite
    diff_sym = 0.5*(diff+np.transpose(diff,axes=(0,2,1))) # symmetric part of diff
    eig_all, _ = np.linalg.eigh(diff_sym) 
    eig_min = np.amin(eig_all, axis=1)          
    diff, g = applyMask(diff, g, eig_min < 0, prt_default)
    num_eig = np.sum(eig_min < 0) # number of entries with negative eigenvalue
    #--------------------------------------------------------------------  
    
    # (4) Add a complement to increase the minimum eigenvalues beyond gamma_min
    if bump_diff:
        complement = np.zeros_like(eig_min)
        mask = (eig_min >= 0) * (eig_min < gamma_min)
        complement[mask] = gamma_min - eig_min[mask]
        assert (complement >= 0).all(), "Negative complement. Something is wrong..."
        diff[:,0,0] = diff[:,0,0] + complement
        diff[:,1,1] = diff[:,1,1] + complement
        diff[:,2,2] = diff[:,2,2] + complement
        if g is not None: g[:,0] = g[:,0] + complement   
        num_bump = np.sum((eig_min<gamma_min)*(eig_min>=0)) # small, positive eigenvalue       
    #--------------------------------------------------------------------     
    
    # (5) Here, make sure that no diffusivities have negative diagonals.
    # After cleaning non-PSD, all diagonals should be positive. Throw error if a negative
    # one is found
    t = np.amin([diff[:,0,0],diff[:,1,1],diff[:,2,2]],axis=0) # minimum diagonal entry
    assert (t >= 0).all(), "Negative diagonal detected. That's not supposed to happen"                    
    #--------------------------------------------------------------------        
    
    # Calculate minimum real part of eigenvalue and minimum diagonal entry 
    # after cleaning
    diff_sym = 0.5*(diff+np.transpose(diff,axes=(0,2,1)))
    eig_all, _ = np.linalg.eigh(diff_sym)
    min_eig = np.amin(eig_all)    
    min_diag = np.amin(np.concatenate((diff[:,0,0], diff[:,1,1], diff[:,2,2])),axis=0)
    
    # Print information        
    toc = timeit.default_timer()
    print("Done! It took {:.1f}s".format(toc-tic), flush=True)
    if verbose:
        if test_inputs is not None:
            print("{} points ({:.2f}% of total) were cleaned due to outlier inputs."\
                  .format(num_ext, 100.0*num_ext/diff.shape[0]), flush=True)
        if clip_elements:
            print("{} entries ({:.2f}% of total) were clipped due to extreme values."\
                  .format(num_clip, 100.0*num_clip/(9*diff.shape[0])), flush=True)        
        print("{} points ({:.2f}% of total) were cleaned due to non-PSD matrices."\
              .format(num_eig, 100.0*num_eig/diff.shape[0]), flush=True)
        if bump_diff:            
            print("{} points ({:.2f}% of total) were almost non-PSD and got fixed."\
                  .format(num_bump, 100.0*num_bump/diff.shape[0]), flush=True)
        print("In cleaned diffusivity: min eigenvalue of "
              + "symmetric part = {:g}".format(min_eig)
              + ", minimum diagonal entry = {:g}".format(min_diag), flush=True)   
    
    if g is None: return diff
    else: return diff, g


def clipElements(diff, i, j, min, max):
    """
    Clips one entry of the diffusivity matrix diff
    
    Arguments:
    diff -- numpy array of shape (n_useful, 3, 3) containing the originally predicted 
            diffusivity tensor
    i -- first index over which we clip, 0 <= i <= 2
    j -- second index over which we clip, 0 <= i <= 2
    min -- minimum value that clipped entry can have
    max -- maximum value that clipped entry can have
    
    Returns:
    diff -- numpy array of shape (n_useful, 3, 3) containing the clipped 
            diffusivity tensor
    num -- total number of entries affected by this clipping  
    """
    
    # counts how many elements we are modifying
    num = np.sum(diff[:,i,j]<min) + np.sum(diff[:,i,j]>max)
    
    diff[diff[:,i,j]<min,i,j] = min
    diff[diff[:,i,j]>max,i,j] = max
    
    return diff, num

    
def applyMask(diff, g, mask, prt_default):
    """
    This simple function applies a mask to diff and g.
    
    Arguments:
    diff -- numpy array of shape (n_useful, 3, 3) containing the originally predicted 
            diffusivity tensor
    g -- numpy array of shape (n_useful, NUM_BASIS) containing the factor that multiplies
         each tensor basis to produce the tensor diffusivity. Can be None.   
    mask -- boolean numpy array of shape (n_useful,) containing a mask which is True
            in the places we want to blank out
    prt_default -- float, default value of turbulent Prandlt number used when cleaning
                   diffusivities that are outright negative. If None is passed, default
                   value is read from constants.py
    
    Returns:
    diff -- numpy array of shape (n_useful, 3, 3) containing the post-processed 
            diffusivity tensor
    g -- numpy array of shape (n_useful, NUM_BASIS) containing the post-processed
         factor that multiplies each tensor basis to produce the tensor diffusivity.
         If g=None in the argument, then None is returned.
    """

    if prt_default is None:
        prt_default = constants.PRT_DEFAULT
    
    diff[mask,:,:] = 0.0
    diff[mask,0,0] = 1.0/prt_default
    diff[mask,1,1] = 1.0/prt_default
    diff[mask,2,2] = 1.0/prt_default
    
    if g is not None:
        g[mask, :] = 0
        g[mask, 0] = 1.0/prt_default    

    return diff, g
    

def calculateLogGamma(uc, gradc, nu_t, tf_flag=False):
    """
    This function calculates log(gamma), where gamma=1/Pr_t, given u'c', gradc, and 
    eddy viscosity
    
    Arguments:
    uc -- np.array or tensor containing the u'c' vector, shape [None, 3]
    gradc -- np.array or tensor containing the concentration gradient, shape [None, 3]
    nu_t -- np.array or tensor containing the eddy viscosity, shape [None,]
    tf_flag -- optional argument, bool which says whether we are dealing with tensorflow
               tensors or with numpy arrays.    
    
    Returns:
    log(gamma) -- np.array or tensor containing the natural log of gamma, shape [None,]
    """
    
    if tf_flag:
        alpha_t = tf.reduce_sum(-1.0*uc*gradc, axis=1)/tf.reduce_sum(gradc*gradc, axis=1)
        gamma = alpha_t/nu_t
        gamma = tf.maximum(gamma, constants.GAMMA_MIN)
        gamma = tf.minimum(gamma, 1.0/constants.GAMMA_MIN)
        return tf.log(gamma)
        
    else:
        alpha_t = np.sum(-1.0*uc*gradc, axis=1) / np.sum(gradc**2, axis=1)
        gamma = alpha_t/nu_t
        gamma[gamma<constants.GAMMA_MIN] = constants.GAMMA_MIN
        gamma[gamma>1.0/constants.GAMMA_MIN] = 1.0/constants.GAMMA_MIN        
        return np.log(gamma)
    

def cleanAnisotropy(b, test_inputs=None, b_default_rans=None, n_std=None, verbose=True):
    """
    This function is called to post-process the anisotropy b for realizability
    
    Arguments:
    b -- numpy array of shape (n_useful, 3, 3) containing the originally predicted 
         anisotropy tensor    
    test_inputs -- numpy array of shape (n_useful, NUM_FEATURES) containing the test 
                   point features that produced the anisotropy tensor we are dealing
                   with. Optional, if None it is not used. 
    b_default_rans -- numpy array containing the default anisotropy tensor in RANS.
                      Shape is (num_points, 3, 3)
    n_std -- float, number of standard deviations around the training mean that each
             test feature is allowed to be. Optional, if None, default value is read from 
             constants.py
    verbose -- optional argument, boolean that says whether we print some extra 
               information to the screen.
    
    Returns:
    b -- numpy array of shape (n_useful, 3, 3) containing the post-processed
         anisotropy tensor    
    """
    
    
    # Get default values if None is passed
    if n_std is None:
        n_std = constants.N_STD    
    
    print("Cleaning predicted anisotropy tensor... ", end="", flush=True)
    tic = timeit.default_timer()        
    
    # (1) Here, make sure that there is no input more or less than n_std away
    # from the mean. Since test_inputs was normalized, it should not be more
    # than n_std or less than -n_std if we don't want test points too different
    # from the training data. We are cleaning points that, by a very simple measure,
    # denote extrapolation from the training set
    mask_ext = np.zeros(b.shape[0], dtype=bool)
    if test_inputs is not None:
        mask_ext = (np.amax(test_inputs,axis=1) > n_std) + \
                   (np.amin(test_inputs,axis=1) < -n_std)        
    num_ext = np.sum(mask_ext) # total number of entries affected by this step
    if num_ext > 0: b[mask_ext, :, :] = b_default_rans[mask_ext, :, :]
    #--------------------------------------------------------------------   
    
    # (2) Here, make sure all matrix entries are realizable.
    b, n_real, _, = makeRealizableFast(b)    
    #--------------------------------------------------------------------                 
    
    # Make sure no tensor entry disobeys the criterium in Ling et al. JFM 2016
    mask = np.zeros(b.shape[0], dtype=bool)
    for i in range(3):
        for j in range(3):
            if i == j:
                mask = mask + (b[:,i,j] > 2.0/3)
                mask = mask + (b[:,i,j] < -1.0/3)
            else:
                mask = mask + (b[:,i,j] > 0.5)
                mask = mask + (b[:,i,j] < -0.5)
    eig_all, _ = np.linalg.eigh(b)
    mask = mask + (eig_all[:, 2] < (3.0*np.abs(eig_all[:, 1]) - eig_all[:, 1])/2.0)
    mask = mask + (eig_all[:, 2] > 1.0/3 - eig_all[:, 1])
    n_fail = np.sum(mask)    
    
    # Print information        
    toc = timeit.default_timer()
    print("Done! It took {:.1f}s".format(toc-tic), flush=True)
    if verbose:
        if test_inputs is not None:
            print("{} points ({:.2f}% of total) were cleaned due to outlier inputs."\
                  .format(num_ext, 100.0*num_ext/b.shape[0]), flush=True)               
        print("{} points ({:.2f}% of total) were cleaned due to non-realizable tensors."\
              .format(n_real, 100.0*n_real/b.shape[0]), flush=True)
        print("In cleaned anisotropy tensor: {} violations detected".format(n_fail))
    
    return b


def makeRealizableFast(b):
    """
    This function takes in Reynolds stress anisotropy tensor and makes it realizable
    
    This function is based on code written by Dr. Andrew Banko. It checks that each
    point represents a realizable turbulence state by verifying that it lies within 
    the barycentric map (c.f. Banerjee et al., Journal of Turbulence, 2017 and
    Emory and Iaccarino, CTR Research Briefs, 2014). If a point lies outside the 
    triangle, it is moved to the nearest point on the boundary of the triangle. 
    
    Arguments:  
    b -- a numpy array of shape (num_points, 3, 3) containing the Reynolds
         stress anisotropy components
    
    Returns:
    b -- corrected anisotropy tensor, shape (num_points, 3, 3)
    n_fails -- number of points failing the realizability test
    ind_fail -- numpy array of shape (num_points,) containing 1 if the original data
                was not realizable, and 0 if it was realizable
    """

    # Functions to transform between eigenvalues, barycentric weights, and barycentric coordinates
    def eval_to_C(evalues):
        C1 = evalues[0] - evalues[1]
        C2 = 2.*(evalues[1] - evalues[2])
        C3 = 3.*evalues[2] + 1.
        C3 = 1. - C1 - C2 # Uncomment to verify sum(Ci) = 1
        return C1, C2, C3

    def C_to_x(C1, C2, C3):
        x = np.zeros(2)
        x[0] = C1 + 0.5*C3
        x[1] = C3*np.sqrt(3)/2.
        return x

    def x_to_C(x):
        C3 = x[1]*2./np.sqrt(3)
        C1 = x[0] - 0.5*C3
        C2 = 1. - C1 - C3
        return C1, C2, C3

    def C_to_eval(C1, C2, C3):
        evalues = np.zeros(3)
        evalues[0] = C1 + 0.5*C2 + C3/3. - 1./3.
        evalues[1] = 0.5*C2 + C3/3. - 1./3.
        evalues[2] = C3/3. - 1./3.
        return evalues            

    # Get number of points, initialize number of realizability violations
    num_points = b.shape[0]
    n_fails = 0
    ind_fail = np.zeros((num_points))

    # Unit vectors defining edges opposite vertices 1, 2, 3
    e1 = np.array([0.5, np.sqrt(3)/2.])
    e1 = e1 / np.linalg.norm(e1)
    e2 = e1
    e2[0] = -1.*e2[0]
    e3 = np.array([1., 0.])
    e1perp = np.array([e1[1], -1.*e1[0]])
    e2perp = np.array([-1.*e2[1], e2[0]])
    e3perp = np.array([e3[1], e3[0]])
    eperp_mat = np.transpose(np.array([e1perp, e2perp, e3perp]))
    pert = 1.e-8
    # Array of vertices 1, 2, 3: i.e. v1 = v[:, 0]
    v = np.array([[1., 0., 0.5], [0., 0., np.sqrt(3)/2.]]) 
    
    # Verify that b is symmetric with zero trace
    b = 0.5 * (b + np.transpose(b, (0, 2, 1)))
    trace_b = np.trace(b, axis1=1, axis2=2)
    for i in range(3):
        b[:, i, i] = b[:, i, i] - 1./3. * trace_b

    # Loop over points to check realizability and project to barycentric map if necessary
    for i in range(num_points):
        # Initialize point's realizability indicator
        fail_ID = 0
        
        # Compute eigenvalues and eigenvectors
        this_b = b[i, :, :]
        evalues, evectors = np.linalg.eig(this_b)
        sort = np.argsort(evalues)
        sort = np.flip(sort,axis=0)
        evalues = evalues[sort]
        evectors = evectors[:, sort]
        
        # Check realizability, correct if necessary
        # Correction procedure: for each vertex, determine if outside triangle
        # opposite edge - if yes, project to nearest point on that edge.
        # At end, if still outside triangle, go to nearest vertex.
        C1, C2, C3 = eval_to_C(evalues)
        x = C_to_x(C1, C2, C3)
        if C1 < 0.0:
            fail_ID = 1
            x = np.dot(x, e1)*e1 + pert*e1perp 
            C1, C2, C3 = x_to_C(x)
        if C2 < 0.0:
            fail_ID = 1
            x = np.dot(x - e3, e2)*e2 + e3 + pert*e2perp
            C1, C2, C3 = x_to_C(x)
        if C3 < 0.0:
            fail_ID = 1
            x = np.dot(x, e3)*e3 + pert*e3perp
            C1, C2, C3 = x_to_C(x)
        if (C1 < 0.0) or (C2 < 0.0) or (C3 < 0.0):
            delta_v = np.zeros(3)
            for j in range(3):
                v_temp = v[:, j]
                delta_v[j] = np.linalg.norm(x - v_temp)
            ind_min = np.argmin(delta_v)
            x = v[:, ind_min] + pert*(eperp_mat[:, np.mod(ind_min-1, 3)] + eperp_mat[:, np.mod(ind_min+1, 3)])

        # Compute corrected anisotropy tensor
        if fail_ID == 1:
            evalues = C_to_eval(C1, C2, C3)
            this_b = np.dot(np.dot(evectors, np.diag(evalues)), np.linalg.inv(evectors))
            b[i, :, :] = this_b
            n_fails += 1

        ind_fail[i] = fail_ID

    # Verify that b is symmetric with zero trace
    b = 0.5 * (b + np.transpose(b, (0, 2, 1)))
    trace_b = np.trace(b, axis1=1, axis2=2)
    for i in range(3):
        b[:, i, i] = b[:, i, i] - 1./3. * trace_b   

    return b, n_fails, ind_fail

    
def suppressWarnings():
    """
    This function suppresses several warnings from Tensorflow.
    """
    
    if type(tf.contrib) != type(tf): tf.contrib._warning = None
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    os.environ['KMP_WARNINGS'] = 'off'    