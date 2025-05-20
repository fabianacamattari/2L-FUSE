
import numpy as np
import random
import torch

from Dictionary import param_definition
from sklearn.model_selection import train_test_split
from KernelDefinition import Matern, Gaussian
from Optimization import OptimizedKernel
from defineReducedFeatures import build_new_features
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def create_dataset(n_points=50000, dataset_dim=35, relevant_dims=6, flag_print=True):

    """
    Function to create the desired dataset. The default values of the arguments allow to reproduce the f1 example.

    Args:
        n_points (int): A positive integer that specifies the number of examples to generate. Default is 50000.
        dataset_dim (int): A positive integer that specifies the number of features to generate. Default is 35.
        relevant_dims (int): A positive integer that specifies the number of relevant features to be considered in the target function expression. Default is 6.
        flag_print (bool): A boolean value that determines whether to silence prints. Default is True.

    Returns:
        data_points (array): An array of generated data points.
        target_function (array): An array of corresponding target values for each data point.
    """

    # Set seed for reproducibility
    set_seed()
    
    # Build data points
    data_points = np.random.uniform(0, 1, size=(n_points, dataset_dim))
    
    if flag_print:
        print(f'Built {data_points.shape[0]} data points in dimension {data_points.shape[1]}.')

    if relevant_dims>20:
        raise ValueError(f"The number of relevant dimensions cannot exceede dataset dimension. Got relevant_dims = {relevant_dims} but dataset_dim = {dataset_dim}.")
    
    # Build target function
    target_function = np.exp(- np.sum(data_points[:, :relevant_dims] - .5 * np.ones_like(data_points[:, :relevant_dims]), axis=1, keepdims=True) ** 2)

    if flag_print:
        print(f'Built target function with {relevant_dims} relevant dimensions.')
    
    if target_function.ndim==1:
        target_function = target_function.reshape(target_function.shape[0],1)    
    
    return data_points, target_function


def get_train_test_data(X_data, Y_data, test_size, flag_print=True):

    """
    Function to split the dataset into training and test sets according to a chosen percentage.

    Args:
        X_data (array-like): The dataset of input points. Each row corresponds to a data point, each column to a feature.
        Y_data (array-like): The dataset of corresponding target values.
        test_size (float): A float between 0 and 1 that represent the proportion of the dataset to to be allocated to the test set.
        flag_print (bool): A boolean value that determines whether to silence prints. Default is True.
        
    Returns:
        X_TRAIN (array): The training data (features).
        X_TEST (array): The test data (features).
        Y_TRAIN (array): The target values for the training data.
        Y_TEST (array): The target values for the test data.
        idx_TRAIN (array): The indices of the training data samples.
        idx_TEST (array): The indices of the test data samples.
    """

    X_TRAIN, X_TEST = None, None
    Y_TRAIN, Y_TEST = None, None
        
    idxs_hour = np.arange(X_data.shape[0])
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST, idx_TRAIN, idx_TEST = train_test_split(X_data, Y_data, idxs_hour, test_size=test_size, random_state=0, shuffle=False)

    if flag_print:
      print(f'Split the starting dataset into {int((1-test_size)*100)}% TRAIN and {int((test_size)*100)}% TEST.')

    return X_TRAIN, X_TEST, Y_TRAIN, Y_TEST, idx_TRAIN, idx_TEST



def run_A_optimization(data_points, target_function, flag_diag_A, test_size=0.2, kernel_type='mat0', flag_print=True):

    """
    Function to build the new data (mapped points) from the analysis of the optimized matrix A and visualize eigenvalues plots.
    The default values of the arguments allow to reproduce the f1 example.

    Reference:
        Tizian Wenzel, Francesco Marchetti, and Emma Perracchione. 
        "Data-driven kernel designs for optimized greedy schemes: A machine learning perspective." 
        SIAM J. Sci. Comput., 46(1):C101–C126, 2024.
        https://doi.org/10.1137/23M1551201.

    Adapted with modifications.

    Args:
        data_points (array-like): The dataset of input points. Each row corresponds to a data point, each column to a feature.
        target_function (array-like): An array of corresponding target values for each data point.
        flag_diag_A (boolean): A boolean indicating whether to optimize a diagonal matrix or not.
        test_size (float): A float between 0 and 1 that represent the proportion of the dataset to to be allocated to the test set. Default is 0.2.
        kernel_type (str): A string specifying the name of the kernel to be used. Supported strings are 'gaussian', 'mat2' and 'mat0'. Default is 'mat0'. 
        flag_print (bool): A boolean value that determines whether to silence prints. Default is True.

    Returns:
        A_optimized (array): The optimized matrix A.
        X_train (array): The training data (features).
        y_train (array): The target values for the training data.
        X_test (array): The test data (features).
        y_test (array): The target values for the test data.
        idx_TRAIN (array): The indices of the training data samples.
        idx_TEST (array): The indices of the test data samples.
    """

    set_seed()

    hyperparameter, reg_para, learning_rate, n_epochs, batch_size, n_folds, flag_initialize_diagonal, flag_symmetric_A = param_definition()

    if kernel_type=='mat0':
        k_matern = 0
        kernel_str = 'MATERN C0 (M0)'
        shape_parameter = hyperparameter.shape_para_matern0
        matern_kernel = Matern(k=k_matern, ep=shape_parameter)
        my_kernel = matern_kernel
        
    elif kernel_type=='mat2':
        k_matern = 2
        kernel_str = 'MATERN C2 (M2)'
        shape_parameter = hyperparameter.shape_para_matern2
        matern_kernel = Matern(k=k_matern, ep=shape_parameter)
        my_kernel = matern_kernel

    elif kernel_type=='gaussian':
        kernel_str = 'GAUSSIAN (GA)'
        shape_parameter = hyperparameter.shape_para_gaussian
        gaussian_kernel = Gaussian(ep=shape_parameter)
        my_kernel = gaussian_kernel

    else:
        raise ValueError("Supported strings for param kernel_type are 'gaussian', 'mat2' and 'mat0'.")

    if flag_print:
        print(f'Preparing for A optimization with {kernel_str} KERNEL')
        case = 'DIAGONAL' if flag_diag_A else 'NON-DIAGONAL'
        print(f'Selected case is {case}')

    X_train, X_test, y_train, y_test, idx_TRAIN, idx_TEST = get_train_test_data(data_points, target_function, test_size)
    X_train_torch, y_train_torch = torch.from_numpy(X_train).type(torch.float), torch.from_numpy(y_train).type(torch.float)

    if flag_print:
        print('*** Starting optimization ***')

    ## The optimization of A is based thanks to the Python package freely available at https://gitlab.mathematik.uni-stuttgart.de/pub/ians-anm/paper-2023-data-driven-kernel-designs
    # --------------------------------------------------------------------------
    model = OptimizedKernel(kernel=my_kernel, dim=X_train_torch.shape[1],
                            reg_para=reg_para,
                            learning_rate=learning_rate,
                            n_epochs=n_epochs,
                            batch_size=batch_size,
                            n_folds=n_folds,
                            flag_initialize_diagonal=flag_initialize_diagonal,
                            flag_symmetric_A=flag_symmetric_A,
                            flag_diag_A=flag_diag_A)

    model.optimize(X_train_torch, y_train_torch, flag_optim_verbose=True)
    A_optimized = model.A.detach().numpy()
    # --------------------------------------------------------------------------

    if flag_print:
        print('*** Optimization ended ***')

    return A_optimized, X_train, y_train, X_test, y_test, idx_TRAIN, idx_TEST


def build_new_data(data_points, features, A_optimized, flag_diag_A, plot_color, marker, reduced_dim=1, threshold_eigenvalues=1e-05):

    """
    Function to build the new data (mapped points) from the analysis of the optimized matrix A and visualize eigenvalues plots.
    The default values of the arguments allow to reproduce the f1 example.

    Args:
        data_points (array-like): The dataset of input points. Each row corresponds to a data point, each column to a feature.
        features (list): The list of feature names (or labels).
        A_optimized (array-like): The the previously optimized matrix to map the points.
        flag_diag_A (boolean): A boolean indicating whether it is the diagonal case. 
        plot_color (str): A string specifying the color to be used for plots (e.g., 'r', 'blue').
        marker (str): A string specifying the marker to be used for plots (e.g., 'o', '*').
        reduced_dim (int): An positive integer indicating the number of linear combination to build for the non-diagonal case. Not used if flag_diag_A = True. Default is 1.
        threshold_eigenvalues (float): A float number to define the threshold to filter significant eigenvalues for the diagonal case. Not used if flag_diag_A = False. Default is 1e-05.
        
    Returns:
        X_new (array-like): The new data points.
    """

    ll = len(features)

    # Check
    if not flag_diag_A:
        if data_points.shape[1] != ll:
            raise ValueError(f'Dimension of data_points should be {ll}, got {data_points.shape[1]} instead.')   
        if reduced_dim > ll:
            raise ValueError(f'Reduced dimension should be {ll} at most, got {reduced_dim} instead.')

    if flag_diag_A:

        # Diagonal case: use the diagonal of A_optimized
        M_eigenvalues_ = np.diag(A_optimized)**2

        print('Eigenvalues of M:\n', M_eigenvalues_)

        # Select features based on significant eigenvalues of M
        indices = np.where(M_eigenvalues_ > threshold_eigenvalues)[0]
        selected_features = [features[i] for i in indices]

        print('Threshold eigenvalues:', threshold_eigenvalues)
        print('Selected features:\n', selected_features)

        # Reduce dimension with new selected features
        X_new = np.sqrt(M_eigenvalues_[indices])*data_points[:,indices]

        order_idxs = np.argsort(M_eigenvalues_)[::-1]
        features_labels = range(1,ll+1)
        M_eigenvalues_ordered = M_eigenvalues_[order_idxs]
        xaxis = range(ll)

        M_eigenvalues_ordered = [eigen if eigen >= 2.2200e-16 else 2.2200e-16 for eigen in M_eigenvalues_ordered]

        plt.figure(figsize=(20, int(20/3)))
        plt.plot(xaxis,M_eigenvalues_ordered,f'{marker}-',color=plot_color,markersize=8,linewidth=0.8,alpha=0.8)
        plt.xticks(ticks=xaxis,labels=[features_labels[i] for i in order_idxs],rotation=45,fontsize=16,color=plot_color)
        plt.yticks(fontsize=16)
        plt.xlabel('Feature Labels',fontsize=20)
        plt.ylabel('Eigenvalues',fontsize=20)
        plt.yscale('log')
        plt.ylim(bottom=1e-17)
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        plt.close()

    else:

        # Eigenvalues of M = A^T@A or M^T = A@A^T (same eigenvalues)
        M_ = np.matmul(A_optimized,np.transpose(A_optimized))
        M_eigenvalues_, M_eigenvectors_ = np.linalg.eig(M_)

        print('Eigenvalues of M:\n', M_eigenvalues_)
        print('\n')

        # Create new feature(s) and new dataset: new feature & SYM-H
        X_new = build_new_features(data_points, M_eigenvalues_, M_eigenvectors_, reduced_dim=reduced_dim)

        order_idxs = np.argsort(M_eigenvalues_)[::-1]
        M_eigenvalues_ordered = M_eigenvalues_[order_idxs]
        xaxis = range(len(features))

        M_eigenvalues_ordered = [eigen if eigen >= 2.2200e-16 else 2.2200e-16 for eigen in M_eigenvalues_ordered]

        plt.figure(figsize=(20, int(20/3)))
        plt.plot(xaxis,M_eigenvalues_ordered,f'{marker}-',color=plot_color,markersize=8,linewidth=0.8,alpha=0.8)
        plt.xticks(ticks=xaxis,labels=[f'{x+1}' for x in xaxis],rotation=45,fontsize=16)
        plt.yticks(fontsize=16)
        plt.xlabel('Number of linear combinations',fontsize=20)
        plt.ylabel('Eigenvalues',fontsize=20)
        plt.yscale('log')
        plt.ylim(bottom=1e-17)
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        plt.close()

    return X_new



def regression_metrics(true, pred):

    """
    Function to calculate regression metrics (here the Mean Squared Error and the Root Mean Squared Error).

    Args:
        true (array-like): The exact values of a target function.
        pred (array-like): The the approximate values ​​of a target function.
        
    Returns:
        mse (float): The value of the Mean Squared Error between true and pred.
        rmse (float): The value of the Root Mean Squared Error between true and pred.
    """

    mse = mean_squared_error(true, pred)
    rmse = np.sqrt(mse)

    return mse, rmse 