
"""
Class adopted from the folder freely available at 
https://gitlab.mathematik.uni-stuttgart.de/pub/ians-anm/paper-2023-data-driven-kernel-designs.
"""

import torch
import numpy as np
from torch import nn
from Rippa import compute_cv_loss_via_rippa_ext_2

class OptimizedKernel(torch.nn.Module):
    '''
    Class for optimizing a two-layered kernel, i.e. to optimize the first-layer matrix A.
    '''

    def __init__(self, kernel, dim,
                 reg_para=1e-5, learning_rate=1e-3, n_epochs=100, batch_size=32, n_folds=None,
                 flag_initialize_diagonal=False, flag_symmetric_A=False, flag_diag_A=False):
        super().__init__()

        # Some settings, mostly optimization related
        self.kernel = kernel

        self.dim = dim
        self.reg_para = reg_para
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        self.flag_symmetric_A = flag_symmetric_A
        self.flag_diag_A = flag_diag_A

        # Define linear maps - hardcoded
        if torch.is_tensor(flag_initialize_diagonal):
            self.B = nn.Parameter(flag_initialize_diagonal, requires_grad=True)
        elif flag_initialize_diagonal:
            self.B = nn.Parameter(torch.eye(self.dim, self.dim), requires_grad=True)
        else:
            self.B = nn.Parameter(torch.rand(self.dim, self.dim), requires_grad=True)

        if self.flag_symmetric_A:
            self.A = (self.B + self.B.t()) / 2
        else:
            self.A = self.B


        if n_folds is None:
            self.n_folds = self.batch_size
        else:
            self.n_folds = n_folds


        # Set optimizer and scheduler
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=.7)

        # Initliaze lists from tracking
        self.list_obj = []
        self.list_parameters = []


    def optimize(self, X, y, flag_optim_verbose=True):

        assert X.shape[0] == y.shape[0], 'Data sizes do not match'
        n_batches = X.shape[0] // self.batch_size

        # Append initial parameters
        if self.flag_symmetric_A:
          self.list_parameters.append(torch.clone((self.B + self.B.t()) / 2).detach().numpy())
        elif self.flag_diag_A:
          self.list_parameters.append(torch.clone(torch.diag(torch.diag(self.A))).detach().numpy())
        else:
          self.list_parameters.append(torch.clone(self.A).detach().numpy())


        for idx_epoch in range(self.n_epochs):
            shuffle = np.random.permutation(X.shape[0])  # reshuffle the data set every epoch

            list_obj_loc = []

            for idx_batch in range(n_batches):

                # Select minibatch from the data
                ind = shuffle[idx_batch * self.batch_size : (idx_batch + 1) * self.batch_size]
                Xb, yb = X[ind, :], y[ind, :]

                # Compute kernel matrix for minibatch
                if self.flag_diag_A:
                  kernel_matrix = self.kernel.eval(Xb @ torch.diag(torch.diag(self.A)), Xb @ torch.diag(torch.diag(self.A)))
                else:
                  kernel_matrix = self.kernel.eval(Xb @ self.A, Xb @ self.A)

                # use cross validation loss via rippa to assess the error
                optimization_objective, _ = compute_cv_loss_via_rippa_ext_2(kernel_matrix, yb, self.n_folds, self.reg_para)

                # Keep track of optimization quantity within epoch
                list_obj_loc.append(optimization_objective.detach().item())
                if idx_epoch == 0 and flag_optim_verbose:
                    print('First epoch: Iteration {:5d}: Training objective: {:.3e}'.format(
                        idx_batch, optimization_objective.detach().item()))

                # Do optimization stuff
                optimization_objective.backward()
                self.optimizer.step()  # do one optimization step
                self.optimizer.zero_grad()  # set gradients to zero

                if self.flag_symmetric_A:
                    self.A = (self.B + self.B.t()) / 2
                else:
                    self.A = self.B

            # Keep track of some quantities and print something
            mean_obj = np.mean(list_obj_loc)

            if flag_optim_verbose:
                print('Epoch {:5d} finished, mean training objective: {:.3e}.'.format(
                    idx_epoch + 1, mean_obj))

            self.list_obj.append(mean_obj)

            self.list_parameters.append(torch.clone(self.A).detach().numpy())



