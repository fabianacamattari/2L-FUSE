
import numpy as np

def build_single_feature(data, eigenvalue, eigvenvector):

    """
    Function to build the linear combinations starting from the informations stored in eigenvalues and eigenvectors.

    Args:
        data (array-like): The dataset of input points. Each row corresponds to a data point, each column to a feature.
        eigenvalue (float): An eigenvalue of a matrix. Here A^T@A.
        eigvenvector (array-like): The eigenvector corrisponding to eigenvalue.

    Returns:
        new_feature (array-like): A single linear combination obtained (for each point in data) by the dot product between the eigenvector and the data point, then scaled by sqrt(eigenvalue).
    """

    new_feature = np.sum(np.sqrt(eigenvalue) * (data * eigvenvector),axis=1).reshape(-1,1)

    return new_feature

def build_new_features(data, eigvalues_arr, eigvectors_arr, reduced_dim=1):

    """
    Function to build the linear combinations starting from the informations stored in eigenvalues and eigenvectors.

    Args:
        data (array-like): The dataset of input points. Each row corresponds to a data point, each column to a feature.
        eigvalues_arr (array-like): An array containing the eigenvalues of a matrix. Here A^T@A.
        eigvectors_arr (array-like): The corrisponding eigenvectors (columns).
        reduced_dim (int): An integer specyfing the number of new features (linear combination) to built. Default is 1.

    Returns:
        new_features_arr (array-like): The array of the new dataset of linear combinations (i.e., the mapped point.)
    """

    idxs_sort = np.flip(np.argsort(eigvalues_arr))

    eigvalues_arr_ = eigvalues_arr[idxs_sort]
    eigvectors_arr_ = eigvectors_arr[idxs_sort]

    print(f'*** Building total number of new features: {reduced_dim} ***')

    for k in range(reduced_dim):

        eigval = eigvalues_arr_[k]
        eigvect = eigvectors_arr_[:,k]
        new_feat = build_single_feature(data, eigval, eigvect)

        if k==0:
            print(f'---------------- Building {k+1}st linear combination...')
        else:
            print(f'---------------- Building {k+1}th linear combination...')
            
        print(f'Index k for eigenvalue and eigenvector selection: {k}')
        print(f'Corresponding eigenvalue: {eigval}')
        print('Corresponding eigenvector:\n'+f'{eigvect}')

        if k==0:
            new_features_arr = new_feat
        else:
            new_features_arr = np.hstack((new_features_arr, new_feat))

    return new_features_arr


