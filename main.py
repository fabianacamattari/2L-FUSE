# %% [markdown]
# # Application Example of the 2L-FUSE procedure
# ## Replicating example f_1

# %%
## Import the needed libraries and/or functions
import numpy as np
from Dictionary import param_definition
from auxiliary_functions import create_dataset
from auxiliary_functions import run_A_optimization
from auxiliary_functions import build_new_data
from auxiliary_functions import regression_metrics
from kernel_interp import KernelInterp
import matplotlib.pyplot as plt

# %% [markdown]
# ### Dataset preparation and hyperparameter configuration

# %%
## Build dataset
X, Y = create_dataset()
features_labels = [f'x_{i+1}' for i in range(X.shape[1])]

# %%
## Select kernel name: possible options are 'gaussian', 'mat0' and 'mat2'
kernel_str = 'gaussian'

hyperparameter = param_definition()[0]

if kernel_str=='mat0':
    epsilon = hyperparameter.shape_para_matern0
    kernel_type_str_ = str(kernel_str)+f'_k_{hyperparameter.shape_para_matern0}'
    kernel_color = 'green'
    kernel_marker = '^'
elif kernel_str=='mat2':
    epsilon = hyperparameter.shape_para_matern2
    kernel_type_str_ = str(kernel_str)+f'_k_{hyperparameter.shape_para_matern2}'
    kernel_color = 'red'
    kernel_marker = 's'
elif kernel_str == 'gaussian':
    epsilon = hyperparameter.shape_para_gaussian
    kernel_type_str_ = str(kernel_str)+f'_shape_para_{hyperparameter.shape_para_gaussian}'
    kernel_color = 'blue'
    kernel_marker = 'o'

# %% [markdown]
# ### Optimization

# %%
## Run A optimization 

# SELECT HERE whether to use diagonal optimization: possible options are flag_diag_A = True and flag_diag_A = False
flag_diag_A = False 

A_opt, X_train_ALL, Y_train, X_test_ALL, Y_test, indices_TRAIN, indices_TEST = run_A_optimization(X, Y, flag_diag_A, kernel_type=kernel_str)

# %%
if flag_diag_A:
    print('Diagonal of optimized A:\n', np.diag(A_opt))
else:
    print('Optimized A:\n', A_opt)

# %%
## Build the reduced dataset (i.e., the mapped points)

# DEFINE HERE the threshold to choose eigenvalues ​​after optimization
thresh_eigenvalues = 1e-02

# DEFINE HERE the number of linear combinations to build (non-diagonal case only)
num_lin_comb = 10

if flag_diag_A:
    X_reduced = build_new_data(X, features_labels, A_opt, flag_diag_A, plot_color=kernel_color, marker=kernel_marker, threshold_eigenvalues=thresh_eigenvalues)
else:
    X_reduced = build_new_data(X, features_labels, A_opt, flag_diag_A, plot_color=kernel_color, marker=kernel_marker, reduced_dim=num_lin_comb, threshold_eigenvalues=None)


# %% [markdown]
# ### Regression 

# %%
## Run regression tests with reduced dataset

# Some parameters for the kernel interpolant
smoothing = hyperparameter.reg_para
nNeighbors = 24

if flag_diag_A:

    X_reduced_TRAIN = X_reduced[indices_TRAIN]
    X_reduced_TEST = X_reduced[indices_TEST]
    
    pred_TEST_reduced = KernelInterp(X_reduced_TRAIN,X_reduced_TEST,epsilon,Y_train,nNeighbors,smoothing,which_kernel_=kernel_str)
    _, RMSE_reduced = regression_metrics(Y_test, pred_TEST_reduced)

    print(f'RMSE with selected (relevant) features: {RMSE_reduced}')

else:

    # Initialize list to save performance scores
    RMSE_comb = []

    for j in range(num_lin_comb):

        X_reduced_j_TRAIN = X_reduced[indices_TRAIN,0:j+1]
        X_reduced_j_TEST = X_reduced[indices_TEST,0:j+1]
        
        if j==0:
            X_reduced_j_TRAIN = X_reduced_j_TRAIN.reshape(-1,1)
            X_reduced_j_TEST = X_reduced_j_TEST.reshape(-1,1)

        pred_TEST_reduced_j = KernelInterp(X_reduced_j_TRAIN,X_reduced_j_TEST,epsilon,Y_train,nNeighbors,smoothing,which_kernel_=kernel_str)

        _, RMSE_reduced_j = regression_metrics(Y_test, pred_TEST_reduced_j)

        RMSE_comb.append(RMSE_reduced_j)

        if j==0:
            print(f'RMSE with {j+1} linear combination: {RMSE_reduced_j}')
        else:
            print(f'RMSE with {j+1} linear combinations: {RMSE_reduced_j}')


# %%
## Run regression test with all features (original dataset)

pred_TEST_ALL = KernelInterp(X_train_ALL,X_test_ALL,epsilon,Y_train,nNeighbors,smoothing,which_kernel_=kernel_str)
_, RMSE_ALL = regression_metrics(Y_test, pred_TEST_ALL)

print(f'RMSE with all features: {RMSE_ALL}')


