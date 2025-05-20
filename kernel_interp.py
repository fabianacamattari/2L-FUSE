import numpy as np
import scipy

def kernel(ep,r,kernel_type='mat0'):

    if kernel_type=='mat0':
        mykernel = np.exp(-ep*r)
    elif kernel_type=='mat2':
        mykernel = np.array(np.exp(-ep*r))*(3 + 3 * np.array(ep * r) + np.array(ep * r) ** 2)
    elif kernel_type=='gaussian':
        mykernel = np.exp(-np.array(ep*r)**2)

    return mykernel

def ComputeKernelApproximant(dsites,epoints,ep,fvalf,smoothing,which_kernel=None):

    dsites = np.matrix(dsites)
    epoints = np.matrix(epoints)
    # Compute kernel and evaluation matrices
    DM = np.zeros((dsites.shape[0], dsites.shape[0]))        
    DM_eval = np.zeros((epoints.shape[0], dsites.shape[0]))      
    for count in range(0,dsites.shape[1]):
        dr, cc = np.meshgrid(epoints[:,count],dsites[:,count])
        DM_eval = DM_eval + (np.power((dr-cc),2)).T
        dr, cc = np.meshgrid(dsites[:,count],dsites[:,count]); 
        DM = DM + (np.power((dr-cc),2)).T
    if which_kernel is not None:
        if which_kernel != 'gaussian' and which_kernel != 'mat0' and which_kernel != 'mat2':
            raise ValueError('Supported strings for param which_kernel are "gaussian", "mat0" and "mat2".')
        else:
            IM = kernel(ep,np.sqrt(DM),which_kernel)+smoothing*np.eye(DM.shape[0],DM.shape[0])
            EM = kernel(ep,np.sqrt(DM_eval),which_kernel)
    else:
        IM = kernel(ep,np.sqrt(DM))+smoothing*np.eye(DM.shape[0],DM.shape[0])
        EM = kernel(ep,np.sqrt(DM_eval))
    coef = np.linalg.solve(IM,fvalf)
    Pf = np.dot(EM,coef)
    return Pf

    
def KernelInterp(dsites,epoints,ep,fvalf,nNeighbors,smoothing,which_kernel_):
    if nNeighbors == 'all':
        Pf = ComputeKernelApproximant(dsites,epoints,ep,fvalf,smoothing,which_kernel=which_kernel_)
    else:
        _tree = scipy.spatial.KDTree(dsites)
        _, yindices = _tree.query(epoints, nNeighbors)
        # Multiple evaluation points may have the same neighborhood of
        # observation points. Make the neighborhoods unique so that we only
        # compute the interpolation coefficients once for each
        # neighborhood.
        yindices = np.sort(yindices, axis=1)
        yindices, inv = np.unique(yindices, return_inverse=True, axis=0)
        # `inv` tells us which neighborhood will be used by each evaluation
        # point. Now we find which evaluation points will be using each
        # neighborhood.
        xindices = [[] for _ in range(len(yindices))]
        for i, j in enumerate(inv):
            xindices[j].append(i)

        Pf = np.zeros((epoints.shape[0], 1))    

        for xidx, yidx in zip(xindices, yindices):
            # `yidx` are the indices of the observations in this
            # neighborhood. `xidx` are the indices of the evaluation points
            # that are using this neighborhood.
            xnbr = epoints[xidx,:]
            ynbr = dsites[yidx,:]
            dnbr = fvalf[yidx]

            Pf[xidx,:] = ComputeKernelApproximant(ynbr,xnbr,ep,dnbr,smoothing,which_kernel=which_kernel_)

        return Pf