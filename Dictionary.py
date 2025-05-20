

"""
Classes and functions adopted with modifications from the folder freely available at 
https://gitlab.mathematik.uni-stuttgart.de/pub/ians-anm/paper-2023-data-driven-kernel-designs.

Reference:
    Tizian Wenzel, Francesco Marchetti, and Emma Perracchione. 
    "Data-driven kernel designs for optimized greedy schemes: A machine learning perspective." 
    SIAM J. Sci. Comput., 46(1):C101–C126, 2024.
    https://doi.org/10.1137/23M1551201.

"""


class example_dictionary():

    reg_para = 1e-3      
    learning_rate = 5e-4
    
    batch_size = 32
    n_folds = None

    flag_initialize_diagonal = True
    flag_symmetric_A = False
    flag_gaussian = False
    
    n_cross_val = 5
    shape_para_matern0 = 1
    shape_para_matern2 = 1
    shape_para_gaussian = 1


class example_data(example_dictionary):
    n_epochs = 15

dic_hyperparams = {'simulated_data': example_data()}

def param_definition(): 

    hyperparameter = dic_hyperparams['simulated_data'] 
    reg_para = hyperparameter.reg_para
    learning_rate = hyperparameter.learning_rate
    n_epochs = hyperparameter.n_epochs
    batch_size = hyperparameter.batch_size
    n_folds = hyperparameter.n_folds
    flag_initialize_diagonal = hyperparameter.flag_initialize_diagonal
    flag_symmetric_A = hyperparameter.flag_symmetric_A

    return hyperparameter, reg_para, learning_rate, n_epochs, batch_size, n_folds, flag_initialize_diagonal, flag_symmetric_A
