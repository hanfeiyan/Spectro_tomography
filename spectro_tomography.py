import astra
import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse.linalg import gmres, LinearOperator
import scipy
from phantominator import shepp_logan
import tifffile as tf
from tqdm.auto import tqdm


def generate_plan(ref,angle_rng,num_angle,option):
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
    n = len(input_list)
    output_list = []
    for i in range(n):
        ind = int(np.floor(np.random.rand()*(n-i)))
        output_list.append(input_list[ind])
        input_list = np.delete(input_list,ind)
    return output_list
    

def multistate_tomo_joint_TV(sinogram, angle, rot_cen_offset, obj_size, ref, mu1, mu2,max_iter,x0=None,nonnegative=False,seq_save=False):
#     regularization with TV: L1 norm of gradients
    num_proj, det_size = np.shape(sinogram)
    num_proj, num_state = np.shape(ref)
    
    proj_geom = astra.create_proj_geom('parallel',1.0,det_size,angle)
    proj_geom_new = astra.geom_postalignment(proj_geom, rot_cen_offset)
    vol_geom = astra.create_vol_geom(obj_size[0],obj_size[1])
    proj_id = astra.create_projector('linear',proj_geom_new,vol_geom)
    
    A = astra.OpTomo(proj_id)
    data_size, rec_size = np.shape(A)
    print("data_size = {}, rec_size = {}".format(data_size,rec_size))
    
    Dx, Dy = der_mat(obj_size)
    DtD = mu2*(Dx.T@Dx + Dy.T@Dy) 
    
    def DtDx(x):
        return DtD@x
    DtD_op = LinearOperator((rec_size,rec_size), matvec = DtDx)
    
    # create linear operators from ref
    W = []
    x = []
    op = []
    b0 = []
    ux = []
    uy = []
    zx = []
    zy = []
    err = []
    #test =[]
    
    def make_w(state,num_proj,det_size,ref):
        T = scipy.sparse.eye(det_size)*ref[0,state]
        for j in range(1,num_proj):
            t = scipy.sparse.eye(det_size)*ref[j,state]
            T = scipy.sparse.block_diag((T,t))
        def w_op(x):
            return T@x
        return w_op
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
         
    if x0 is not None:
        fp = 0
        reg = 0
        for i in range(num_state):
            x[i] = x0[i].ravel() 
            fp = fp + W[i]*A*x[i]
            reg = reg + mu1*(np.linalg.norm(Dx@x[i],1) + np.linalg.norm(Dy@x[i],1))
        err.append(0.5*np.linalg.norm(sinogram.ravel()-fp)**2+reg)
    B = []
    X = []
    for i in range(num_state):
        B = np.concatenate((B,b0[i]))
        X = np.concatenate((X,x[i]))
    
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
    F = make_joint_op(A,W,DtD_op,rec_size)
    F_OP = LinearOperator((rec_size*num_state,rec_size*num_state),matvec=F)
    seq = []
    for itr in tqdm(range(max_iter)):
        C = []
        for i in range(num_state):
            C = np.concatenate((C,(Dx.T@ux[i]+Dy.T@uy[i]) - mu2*(Dx.T@zx[i]+Dy.T@zy[i])))
        X = gmres(F_OP,B-C,X)[0]    
        if nonnegative:
            X[X<0] = 0
        for i in range(num_state):    
            vx = Dx@X[i*rec_size:(i+1)*rec_size]+(1/mu2)*ux[i]
            zx[i] = np.fmax(np.abs(vx)-mu1/mu2,0)*np.sign(vx)
            vy = Dy@X[i*rec_size:(i+1)*rec_size]+(1/mu2)*uy[i]
            zy[i] = np.fmax(np.abs(vy)-mu1/mu2,0)*np.sign(vy)
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


def der_im(f):
    dx_f = np.diff(f,1,1)
    dy_f = np.diff(f,1,0)
    dx_f = np.concatenate((dx_f,np.reshape(f[:,0]-f[:,-1],[-1,1])),1)
    dy_f = np.concatenate((dy_f,np.reshape(f[0,:]-f[-1,:],[1,-1])),0)    
    return dx_f, dy_f
def der_t(ux,uy):
    dxt_ux = np.concatenate((np.reshape(ux[:,-1]-ux[:,0],[-1,1]),-np.diff(ux,1,1)),1)
    dyt_uy = np.concatenate((np.reshape(uy[-1,:]-uy[0,:],[1,-1]),-np.diff(uy,1,0)),0)
    return dxt_ux,dyt_uy
def conv_2d(f,h):
    sz = np.shape(f)
    h_sz = np.shape(h)
    h = np.pad(h,((0,sz[0]-h_sz[0]),(0,sz[1]-h_sz[1])))
    h = np.roll(h,[-(h_sz[0]//2),-(h_sz[1]//2)],[0,1])
    H = np.fft.fft2(h)
    F = np.fft.fft2(f)
    g = np.fft.ifft2(F*H)
    return np.real(g)
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
def der_mat(sz):
    A = np.ones([1,sz[1]])
    B = scipy.sparse.spdiags(np.vstack((-A,A)),[0,1],sz[1]-1,sz[1])
    Dx = B
    for i in range(1,sz[0]):
        Dx = scipy.sparse.block_diag((Dx,B))
    
    A = np.ones((1,sz[1]*sz[0]))
    Dy = scipy.sparse.spdiags(np.vstack((-A,A)),[0,sz[1]],sz[1]*(sz[0]-1),sz[0]*sz[1])
    return Dx,Dy

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