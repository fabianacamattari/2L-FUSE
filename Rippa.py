"""
This code is based on the Matlab repository https://github.com/cesc14/RippaExtCV provided by Francesco Marchetti
"""

import torch 

def compute_cv_loss_via_rippa_ext_2(kernel_matrix, y, n_folds, reg_for_matrix_inversion):
    """
    Implementation without the need to provide a kernel and points: Simply provide the kernel matrix
    """

    # Some precomputations
    kernel_matrix_reg = kernel_matrix + reg_for_matrix_inversion * torch.eye(kernel_matrix.shape[0])
    inv_kernel_matrix = torch.inverse(kernel_matrix_reg)
    coeffs = torch.linalg.solve(kernel_matrix_reg, y) #[0]

    # Some initializations and preparations: It is required that n_folds divides y.shape[0] without remainder
    array_error = torch.zeros(y.shape[0], 1)
    n_per_fold = int(y.shape[0] / n_folds)
    indices = torch.arange(0, y.shape[0]).view(n_per_fold, n_folds)

    # Standard Rippa's scheme
    if n_folds == y.shape[0]:
        array_error = coeffs / torch.diag(inv_kernel_matrix).view(-1,1)

    # Extended Rippa's scheme
    else:
        for j in range(n_folds):
            inv_kernel_matrix_loc1 = inv_kernel_matrix[indices[:, j], :]
            inv_kernel_matrix_loc = inv_kernel_matrix_loc1[:, indices[:, j]]

            array_error[j * n_per_fold: (j+1) * n_per_fold, 0] = \
                (torch.linalg.solve(inv_kernel_matrix_loc, coeffs[indices[:, j]])).view(-1)

    cv_error_sq = torch.sum(array_error ** 2) / array_error.numel()

    return cv_error_sq, array_error
