import numpy as np

def prop_iter(Cp,U,x,f,E,dt):
    '''
    Single propagation step using the split-operator technique
    U: list of U matrices [x,y,z], which are eigenvectors of transition dipole matrix
    x: list of epsilon vectors [x,y,z], which are eigenvalues of transition dipole matrix
    f: field strength list of field strengths [x,y,z]
    E: state energy
    '''
    Cn = np.einsum('i,i->i',np.exp(-1.0j*E*dt),Cp)
    for a in range(3):
        Cn = np.einsum('kj,j->k',U[a].T,Cn)
        Cn = np.einsum('k,k->k',np.exp(1.0j*x[a]*f[a]*dt),Cn)
        Cn = np.einsum('ki,k->i',U[a].T,Cn)
    return Cn

def build_U(trans_list):
    '''
    diagonalize transition dipole matrix
    returns [x,y,z] lists of sorted eigenvectors and eigenvalues
    '''
    eps_list = []
    U_list = []
    for a in range(3):
        eps_a,U_a = np.linalg.eigh(trans_list[a])
        eps_list.append(eps_a)
        U_list.append(U_a)
    return eps_list,U_list 

def unwrapped_prop_iter(Cp,U,x,f,E,dt):
    '''
    Split-operator propagation using unwrapped for-loops for timing purposes
    U: list of U matrices [x,y,z], which are eigenvectors of transition dipole matrix
    x: list of epsilon vectors [x,y,z], which are eigenvalues of transition dipole matrix
    f: field strength list of field strengths [x,y,z]
    E: state energy
    '''
    # shamelessly copying the TDCI-v1.7 algorithm...
    nstates = Cp.shape[0]
    Cn = np.zeros_like(Cp).astype('complex128')
    for i in range(nstates):
        Cn[i] = np.exp(-1.0j * E[i] * dt) * Cp[i]

    aux1 = np.zeros_like(Cp).astype('complex128')
    for tau in range(nstates):
        for j in range(nstates):
            aux1[tau] += U[0][tau,j] * Cn[j]

    aux2 = np.zeros_like(Cp).astype('complex128')
    for sig in range(nstates):
        aux2[sig] += np.exp(1.0j * x[0][sig] * f[0] * dt) * aux1[sig]

    aux1 = np.zeros_like(Cp).astype('complex128')
    for k in range(nstates):
        for sig in range(nstates):
            aux1[k] += U[0][sig,k] * aux2[sig]

    aux2 = np.zeros_like(Cp).astype('complex128')
    for tau in range(nstates):
        for j in range(nstates):
                aux2[tau] += U[1][tau,j] * aux1[j]

    for sig in range(nstates):
        aux1[sig] = np.exp(1.0j * x[1][sig] * f[1] * dt) * aux2[sig]

    aux2 = np.zeros_like(Cp).astype('complex128')
    for k in range(nstates):
        for sig in range(nstates):
            aux2[k] += U[1][sig,k] * aux1[sig]

    aux1 = np.zeros_like(Cp).astype('complex128')
    for tau in range(nstates):
        for j in range(nstates):
            aux1[tau] += U[2][tau,j] * aux2[j]

    for sig in range(nstates):
        aux2[sig] = np.exp(1.0j * x[2][sig] * f[2] * dt) * aux1[sig]

    Cn = np.zeros_like(Cp).astype('complex128')
    for k in range(nstates):
        for sig in range(nstates):
            Cn[k] += U[2][sig,k] * aux2[sig]

    return Cn
