import astra
import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse.linalg import gmres, LinearOperator, lsqr, lsmr, cg, cgs, bicg, bicgstab, minres, spsolve, lgmres, qmr
import scipy
from phantominator import shepp_logan
import tifffile as tf
from tqdm.auto import tqdm
from skimage.metrics import mean_squared_error as mse

def generate_plan(ref,angle_rng,num_angle,option):
    """
    create an acquisition plan.

    Parameters
    ----------
    ref : ndarray, shape (num_energy, num_state)
        Reference spectra for each energy/state.
    angle_rng : tuple (min_angle, max_angle)
        Range of projection angles (radians or degrees, consistent with ASTRA).
    num_angle : int
        Total number of projections to generate.
    option : str
        Sampling strategy.  Accepted values include
        `'uniform-interlaced'`, `'random-random'`, `'golden-ratio-interlaced'`,
        … see source for full list.

    Returns
    -------
    angle_list : ndarray, shape (num_angle,)
        Sorted list of projection angles.
    spectra_list : ndarray, shape (num_angle, num_state)
        Corresponding spectrum selected for each angle.
    ind : ndarray, shape (num_angle,) or []
        Indices of `ref` used (only for options that choose randomly).
    """
    num_energy, num_state = np.shape(ref)
    
    multiplicity = num_angle//num_energy
    remainder = num_angle%num_energy
    
    spectra_list = []    
    ind = []
    if option == 'uniform-interlaced': 
        angle_list = np.linspace(angle_rng[0],angle_rng[1],num_angle)
        for i in range(multiplicity):
            if len(spectra_list) == 0:
                spectra_list = ref
            else:
                spectra_list = np.concatenate((spectra_list,ref),axis = 0) 
        if len(spectra_list) == 0:
            spectra_list = spectra_list,ref[:remainder,:]
        else:
            spectra_list = np.concatenate((spectra_list,ref[:remainder,:]),axis = 0)
    elif option == 'uniform-interlaced-random': 
        np.random.shuffle(ref)
        angle_list = np.linspace(angle_rng[0],angle_rng[1],num_angle)
        for i in range(multiplicity):
            if len(spectra_list) == 0:
                spectra_list = ref
            else:
                spectra_list = np.concatenate((spectra_list,ref),axis = 0) 
        if len(spectra_list) == 0:
            spectra_list = spectra_list,ref[:remainder,:]
        else:
            spectra_list = np.concatenate((spectra_list,ref[:remainder,:]),axis = 0)
    elif option == 'uniform-segmented': 
        angle_list = np.linspace(angle_rng[0],angle_rng[1],num_angle)
        spectra_list = np.zeros((num_angle,num_state))
        if multiplicity == 0:
            multiplicity == 1
        for i in range(num_angle):
            index = i//multiplicity
            if index >= num_energy:
                index = index-num_energy
            spectra_list[i,:] = ref[index,:]
                
    elif option == 'golden-ratio-interlaced':
        rng = angle_rng[1]-angle_rng[0]
        angle_list = angle_rng[0] + np.mod(rng*np.linspace(0,num_angle,num_angle)/1.618,rng)
        angle_list = np.sort(angle_list)
        for i in range(multiplicity):
            if len(spectra_list) == 0:
                spectra_list = ref
            else:
                spectra_list = np.concatenate((spectra_list,ref),axis = 0) 
        if len(spectra_list) == 0:
            spectra_list = spectra_list,ref[:remainder,:]
        else:
            spectra_list = np.concatenate((spectra_list,ref[:remainder,:]),axis = 0) 
    elif option == 'random-random':
        angle_list = np.random.rand(num_angle)*(angle_rng[1]-angle_rng[0])+angle_rng[0]
        angle_list = np.sort(angle_list)
        ind = np.array(np.floor(np.random.rand(num_angle)*num_energy),dtype='int')
        spectra_list = ref[ind,:]
    elif option == 'uniform-random':
        angle_list = np.linspace(angle_rng[0],angle_rng[1],num_angle)
        ind = np.array(np.floor(np.random.rand(num_angle)*num_energy),dtype='int')
        spectra_list = ref[ind,:]
    elif option == 'random-interlaced':
        angle_list = np.random.rand(num_angle)*(angle_rng[1]-angle_rng[0])+angle_rng[0]
        angle_list = np.sort(angle_list)
        for i in range(multiplicity):
            if len(spectra_list) == 0:
                spectra_list = ref
            else:
                spectra_list = np.concatenate((spectra_list,ref),axis = 0) 
        if len(spectra_list) == 0:
            spectra_list = spectra_list,ref[:remainder,:]
        else:
            spectra_list = np.concatenate((spectra_list,ref[:remainder,:]),axis = 0) 
    elif option == 'golden-ratio-random':
        rng = angle_rng[1]-angle_rng[0]
        angle_list = angle_rng[0] + np.mod(rng*np.linspace(0,num_angle,num_angle)/1.618,rng)
        angle_list = np.sort(angle_list)
        ind = np.array(np.floor(np.random.rand(num_angle)*num_energy),dtype='int')
        spectra_list = ref[ind,:]
    elif option == 'uniform-random2':
        angle_list = np.linspace(angle_rng[0],angle_rng[1],num_angle)
        
        for i in range(multiplicity):
            ind = rand_list(np.linspace(0,num_energy-1,num_energy,dtype='int'))
            if len(spectra_list) == 0:
                spectra_list = ref[ind,:]
            else:
                spectra_list = np.concatenate((spectra_list,ref[ind,:]),axis = 0)
            if remainder > 0:
                ind = rand_list(np.linspace(0,remainder-1,remainder,dtype='int'))
                spectra_list = np.concatenate((spectra_list,ref[ind,:]),axis = 0)
    elif option == 'random-random2':
        angle_list = np.random.rand(num_angle)*(angle_rng[1]-angle_rng[0])+angle_rng[0]
        angle_list = np.sort(angle_list)
        for i in range(multiplicity):
            ind = rand_list(np.linspace(0,num_energy-1,num_energy,dtype='int'))
            if len(spectra_list) == 0:
                spectra_list = ref[ind,:]
            else:
                spectra_list = np.concatenate((spectra_list,ref[ind,:]),axis = 0)
    elif option == 'special':
        n = 10
        angle_list = np.random.rand(n)*(angle_rng[1]-angle_rng[0])+angle_rng[0]
        spectra_list = np.reshape(ref[num_energy//2,:],(1,-1))
        print(np.shape(spectra_list))
        for i in range(n-1):
            spectra_list = np.concatenate((spectra_list,np.reshape(ref[num_energy//2,:],(1,-1))),axis = 0)
        angle_list = np.concatenate((angle_list,np.random.rand(num_angle-n)*(angle_rng[1]-angle_rng[0])+angle_rng[0]),axis=0)
        angle_list = np.sort(angle_list)
        ind = np.array(np.floor(np.random.rand(num_angle-n)*num_energy),dtype='int')
        tmp = ref[ind,:]
        print(np.shape(tmp))
        spectra_list = np.concatenate((spectra_list,ref[ind,:]),axis=0)
    else:
        print('default uniform-random sampling is used')
        angle_list = np.linspace(angle_rng[0],angle_rng[1],num_angle)
        ind = np.array(np.floor(np.random.rand(num_angle)*num_energy),dtype='int')
        spectra_list = ref[ind,:]
         
    return angle_list, spectra_list, ind

def rand_list(input_list):
    """
    return a random permutation of the elements of `input_list`.

    A simple in-place shuffle implemented by repeatedly selecting and
    deleting a random element.

    Parameters
    ----------
    input_list : iterable
        Values to permute.

    Returns
    -------
    list
        Permuted values.
    """
    n = len(input_list)
    output_list = []
    for i in range(n):
        ind = int(np.floor(np.random.rand()*(n-i)))
        output_list.append(input_list[ind])
        input_list = np.delete(input_list,ind)
    return output_list
    

def multistate_tomo_joint_TV(sinogram, angle, rot_cen_offset, obj_size, ref, 
                             mu1, mu2,max_iter,x0=None,nonnegative=False,
                             seq_save=False, method='gmres',**kwargs):
    """
    joint total-variation reconstruction for multistate tomography.

    This routine minimises
    
        0.5 ||WAx - b||_2^2 + mu1 * TV(x),

    where `b` is the measured data, `A` is the ASTRA forward projector assembled from the projection
    geometry and `W` weights the different spectral states.  The algorithm
    is an ADMM-style iteration solving a linear system with the chosen
    `method` at each outer iteration.

    Parameters
    ----------
    sinogram : ndarray, shape (num_proj, det_size)
        Measured projection data (stacked for all angles).
    angle : ndarray
        Projection angles.
    rot_cen_offset : float
        Rotation centre offset for ASTRA geometry.
    obj_size : tuple (nx, ny)
        Reconstruction image dimensions.
    ref : ndarray, shape (num_proj, num_state)
        Spectral weights for each projection at given energy.
    mu1 : float
        Regularisation weight for TV.
    mu2 : float
        Hyperparameter for ADMM contolling the convergence rate.
    max_iter : int
        Number of outer iterations.
    x0 : list of ndarray, optional
        Initial guess for each state; if provided, pre-filled in `err`.
    nonnegative : bool, default False
        Enforce non-negativity on each update.
    seq_save : bool, default False
        If True return the entire sequence of iterates.
    method : str, default 'gmres'
        Name of the linear solver to use; one of the keys of `solvers`.

    Returns
    -------
    x : list of ndarray or list of ndarray (sequence)
        Reconstructed states; if `seq_save` is True, a list of the iterates.
    err : list of float
        Objective value (convergence) at each iteration.
    """

    num_proj, det_size = np.shape(sinogram)
    num_proj, num_state = np.shape(ref)
    
    # create ASTRA geometry and projector
    proj_geom = astra.create_proj_geom('parallel',1.0,det_size,angle)
    proj_geom_new = astra.geom_postalignment(proj_geom, rot_cen_offset)
    vol_geom = astra.create_vol_geom(obj_size[0],obj_size[1])
    proj_id = astra.create_projector('linear',proj_geom_new,vol_geom)
    A = astra.OpTomo(proj_id)
    data_size, rec_size = np.shape(A)
    print("data_size = {}, rec_size = {}".format(data_size,rec_size*num_state))
    
    # create finite-difference operators for TV regularisation
    Dx, Dy = der_mat(obj_size)
    # precompute DtD for the linear system; this is the same for each state and does not change during iterations
    DtD = mu2*(Dx.T@Dx + Dy.T@Dy) 
    # define a function for the matrix-vector product with DtD, to be used in the linear solver; this avoids forming the full DtD matrix explicitly, which can be large and sparse
    def DtDx(x):
        return DtD@x
    # create a linear operator for DtD to be used in the linear solver; this allows us to use iterative solvers without forming the full matrix
    DtD_op = LinearOperator((rec_size,rec_size), matvec = DtDx)
    # define the linear solvers to be used; these are functions that take a matrix (or linear operator) and a right-hand side, and return a solution; we include a simple wrapper for `spsolve` to match the interface of the iterative solvers
    solvers = {
        "spsolve": lambda A, b, **kw: (spsolve(A, b), 0),
        "cg": cg,
        "bicg": bicg,
        "cgs": cgs,
        "bicgstab": bicgstab,
        "gmres": gmres,
        "minres": minres,
        "lsqr": lsqr,
        "lsmr": lsmr,
        "lgmres": lgmres,
        "qmr": qmr,
    }
    # check if the specified method is valid and get the corresponding solver function
    if method not in solvers:
        raise ValueError(f"Unknown solver '{method}'. Available: {list(solvers)}")
    solver = solvers[method]
    # initialise variables for the ADMM iterations; we maintain separate variables for each state.
    W = []
    x = []
    op = []
    b0 = []
    ux = []
    uy = []
    zx = []
    zy = []
    err = []
    # define a function to create the weighted projection operator for a given state; this constructs a block-diagonal 
    # operator where each block corresponds to the projection operator for that state weighted by the corresponding 
    # spectral weight from `ref`
    def make_w(state,num_proj,det_size,ref):
        T = scipy.sparse.eye(det_size)*ref[0,state]
        for j in range(1,num_proj):
            t = scipy.sparse.eye(det_size)*ref[j,state]
            T = scipy.sparse.block_diag((T,t))
        def w_op(x):
            return T@x
        return w_op
    # create the weighted projection operators and initialise the variables for each state in the ADMM iterations; 
    # we also compute the initial right-hand side for the linear system based on the measured data and the weighted 
    # projection operators. Note: all state maps are flatterned to vectors and concatenated together for the linear 
    # system solved in each ADMM iteration; this allows us to solve for all states simultaneously while accounting 
    # for their coupling through the data fidelity term.
    for i in range(num_state):
        f = make_w(i,num_proj,det_size,ref)
        W.append(LinearOperator((data_size,data_size),matvec=f,rmatvec=f))
        x.append(np.zeros(rec_size))
        op.append(A.T*W[i].T*W[i]*A)
        b0.append(A.T*W[i].T*sinogram.ravel())
        ux.append(np.zeros(obj_size[0]*(obj_size[1]-1)))
        uy.append(np.zeros((obj_size[0]-1)*obj_size[1]))
        zx.append(np.zeros(obj_size[0]*(obj_size[1]-1)))
        zy.append(np.zeros((obj_size[0]-1)*obj_size[1]))
    # if an initial guess `x0` is provided, we compute the initial objective value and store it in `err`; 
    # this allows us to track the convergence from the initial guess.     
    if x0 is not None:
        fp = 0
        reg = 0
        for i in range(num_state):
            x[i] = x0[i].ravel() 
            fp = fp + W[i]*A*x[i]
            reg = reg + mu1*(np.linalg.norm(Dx@x[i],1) + np.linalg.norm(Dy@x[i],1))
        err.append(0.5*np.linalg.norm(sinogram.ravel()-fp)**2+reg)
    # we concatenate the right-hand sides for each state into a single vector `B`, and similarly concatenate the variables 
    # for each state into a single vector `X`; this allows us to solve the linear system for all states simultaneously 
    # in the ADMM iterations.
    B = []
    X = []
    for i in range(num_state):
        B = np.concatenate((B,b0[i]))
        X = np.concatenate((X,x[i]))

    # we define a function to create the joint linear operator for the linear system solved in each ADMM iteration; 
    # this operator combines the contributions from the data fidelity term (involving the weighted projection operators) 
    # and the regularisation term (involving DtD); it is structured as a block matrix where each block corresponds to the 
    # interactions between different states, allowing us to solve for all states simultaneously while accounting for their 
    # coupling through the data fidelity term.
    def make_joint_op(A, W, DtD_op, rec_size):
        num_state = len(W)
        def joint_op(x):
            subgroup_x = []
            output = []
            for i in range(num_state):
                subgroup_x.append(x[i*rec_size:(i+1)*rec_size])
            for i in range(num_state):
                tmp = DtD_op*subgroup_x[i]
                for j in range(num_state):
                    tmp = tmp + A.T*W[i].T*W[j]*A*subgroup_x[j]
                output = np.concatenate((output,tmp))
            return output
        return joint_op
    
    # we create the joint linear operator for the linear system solved in each ADMM iteration, 
    # and then we enter the main loop of the ADMM iterations; in each iteration, 
    # we compute the right-hand side for the linear system based on the current estimates 
    # of the variables and the dual variables, solve the linear system using the chosen solver, 
    # apply non-negativity constraints if specified, and then update the dual variables based 
    # on the new estimates; we also compute and store the objective value at each iteration to 
    # track convergence, and optionally save the sequence of iterates if `seq_save` is True.
    F = make_joint_op(A,W,DtD_op,rec_size)
    F_OP = LinearOperator((rec_size*num_state,rec_size*num_state),matvec=F,rmatvec=F)
    seq = []
    
    for itr in tqdm(range(max_iter)):# outer ADMM iterations
        C = []
        for i in range(num_state):
            C = np.concatenate((C,(Dx.T@ux[i]+Dy.T@uy[i]) - mu2*(Dx.T@zx[i]+Dy.T@zy[i])))
        X = solver(F_OP,B-C,x0=X)[0] # suboptimization for the primal variable update; this solves the linear system defined by the joint operator `F_OP` and the right-hand side `B-C`, where `C` accounts for the contributions from the dual variables and the regularisation term; the solution is stored back in `X`, which contains the updated estimates for all states concatenated together.   
        if nonnegative:
            X[X<0] = 0
        for i in range(num_state): # update the dual variables for each state based on the new estimates; this involves computing the finite differences of the current estimates, applying soft-thresholding to enforce sparsity in the gradients (which corresponds to TV regularisation), and then updating the dual variables `ux` and `uy` based on the difference between the finite differences and the thresholded values; this step is crucial for the ADMM iterations as it enforces the TV regularisation while allowing for efficient updates of the primal variable in the next iteration.   
            vx = Dx@X[i*rec_size:(i+1)*rec_size]+(1/mu2)*ux[i]
            zx[i] = np.fmax(np.abs(vx)-mu1/mu2,0)*np.sign(vx)
            vy = Dy@X[i*rec_size:(i+1)*rec_size]+(1/mu2)*uy[i]
            zy[i] = np.fmax(np.abs(vy)-mu1/mu2,0)*np.sign(vy)
            # dual variable update; this step updates the dual variables `ux` and `uy` based on the 
            # difference between the finite differences of the current estimates and the thresholded 
            # values `zx` and `zy`; this is a standard step in ADMM iterations for TV regularisation, 
            # where the dual variables are updated to enforce consistency between the primal variable 
            # (the image estimates) and the auxiliary variables (the thresholded gradients), 
            # allowing for efficient convergence to a solution that balances data fidelity and regularisation.
            ux[i] = ux[i] + mu2*(Dx@X[i*rec_size:(i+1)*rec_size]-zx[i])
            uy[i] = uy[i] + mu2*(Dy@X[i*rec_size:(i+1)*rec_size]-zy[i])    
            
        fp = 0
        reg = 0
        for i in range(num_state):
            x[i] = X[i*rec_size:(i+1)*rec_size]
            fp = fp + W[i]*A*x[i]
            reg = reg + mu1*(np.linalg.norm(Dx@x[i],1) + np.linalg.norm(Dy@x[i],1))
        err.append(0.5*np.linalg.norm(sinogram.ravel()-fp)**2+reg)
        if seq_save is True:
            seq.append(np.reshape(x,[num_state,obj_size[0],obj_size[1]]))
    for i in range(num_state):
        x[i] = np.reshape(x[i],obj_size)
    if seq_save is True:
        return seq, err
    else:
        return x, err

# the following functions are utility functions for computing finite differences, 
# convolution, deconvolution, and visualization; they are used in the main reconstruction 
# routine for computing the TV regularisation and for visualizing the results.

# `der_im` computes the derivative of an image `f` in the x and y directions.  
def der_im(f):
    dx_f = np.diff(f,1,1)
    dy_f = np.diff(f,1,0)
    # peridoic boundary conditions to ensure the returned derivatives have the same shape as the input image; 
    # this is important for consistency in the ADMM iterations, where we need to apply the finite difference 
    # operators and their adjoints without changing the dimensions of the variables.
    dx_f = np.concatenate((dx_f,np.reshape(f[:,0]-f[:,-1],[-1,1])),1)
    dy_f = np.concatenate((dy_f,np.reshape(f[0,:]-f[-1,:],[1,-1])),0)    
    return dx_f, dy_f
# `der_t` computes the adjoint of the finite difference operators, which is used in the ADMM iterations for updating the dual variables; it also uses periodic boundary conditions to ensure consistency with `der_im`.
def der_t(ux,uy):
    dxt_ux = np.concatenate((np.reshape(ux[:,-1]-ux[:,0],[-1,1]),-np.diff(ux,1,1)),1)
    dyt_uy = np.concatenate((np.reshape(uy[-1,:]-uy[0,:],[1,-1]),-np.diff(uy,1,0)),0)
    return dxt_ux,dyt_uy
# `conv_2d` computes the convolution of an image `f` with a kernel `h` using the Fourier transform; it pads the kernel to the size of the image and applies a shift to ensure that the convolution is computed correctly with periodic boundary conditions; this is used in the implementation of the TV regularisation in the ADMM iterations, where we need to apply the finite difference operators and their adjoints efficiently.
def conv_2d(f,h):
    sz = np.shape(f)
    h_sz = np.shape(h)
    h = np.pad(h,((0,sz[0]-h_sz[0]),(0,sz[1]-h_sz[1])))
    h = np.roll(h,[-(h_sz[0]//2),-(h_sz[1]//2)],[0,1])
    H = np.fft.fft2(h)
    F = np.fft.fft2(f)
    g = np.fft.ifft2(F*H)
    return np.real(g)
# deconvolution with periodic boundary conditions; this is used in the implementation of the TV regularisation in the ADMM iterations.
def deconv_2d(g,h):
    sz = np.shape(g)
    h_sz = np.shape(h)
    h = np.pad(h,((0,sz[0]-h_sz[0]),(0,sz[1]-h_sz[1])))
    h = np.roll(h,[-(h_sz[0]//2),-(h_sz[1]//2)],[0,1])
    H = np.fft.fft2(h)
    HtH = np.abs(H)**2
    G = np.fft.fft2(g)
    f = np.fft.ifft2(np.conj(H)*G/HtH)
    return np.real(f)
# derivative matrices in x and y directions of a vectorized image with a size of `sz`; this is used to compute the TV regularisation term in the reconstruction routine.
def der_mat(sz):
    A = np.ones([1,sz[1]])
    B = scipy.sparse.spdiags(np.vstack((-A,A)),[0,1],sz[1]-1,sz[1])
    Dx = B
    for i in range(1,sz[0]):
        Dx = scipy.sparse.block_diag((Dx,B))
    
    A = np.ones((1,sz[1]*sz[0]))
    Dy = scipy.sparse.spdiags(np.vstack((-A,A)),[0,sz[1]],sz[1]*(sz[0]-1),sz[0]*sz[1])
    return Dx,Dy
# plot images in a grid.
def grid_view(frames):
    sz = np.shape(frames)
    col = 4
    row = int(np.ceil(sz[0]/col))
    #plt.figure()
    fig, ax = plt.subplots(row, col, figsize=(col*3,row*3))
    fig.tight_layout()
    ind = 0
    for i in range(row):
        for j in range(col):
            #plt.subplot(row, col, ind, figsize=(4,3))
            if ind < sz[0]:
                ax[i,j].imshow(frames[ind,:,:])
            ind = ind + 1
            ax[i,j].axis('off')
    #plt.tight_layout()
# normalize the sinogram by scaling each projection to have the same total mass; this is a common pre-processing step in tomography to mitigate variations in intensity across projections, which can arise from factors such as varying exposure or detector sensitivity; the function computes the total mass for each projection and scales the sinogram accordingly, returning the normalized sinogram and the scaling ratios used for each projection.
def normalize_sinogram(im):
    sz = np.shape(im)
    out_im = np.zeros(sz)
    ratio = np.zeros((sz[1],sz[0]))
    for i in range(sz[1]):
        mass = np.squeeze(np.sum(im[0,i,:]))
        for j in range(sz[0]):
            new_mass = np.squeeze(np.sum(im[j,i,:]))
            ratio[i,j] = mass/(new_mass+1e-6)
            out_im[j,i,:] = im[j,i,:]*ratio[i,j]
            #mass = new_mass
    return out_im, ratio