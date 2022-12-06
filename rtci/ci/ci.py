import psi4
import numpy as np

class ciwfn(object):
    def __init__(self,ref_e,ref_wfn,method="CIS"):
        allowed_methods = ["CIS"]
        if method.upper() not in allowed_methods:
            raise Exception('Method type {} not accepted.\nPlease select from: {}'
                            .format(method,allowed_methods))

        self.ref_e = ref_e
        self.ref_wfn = ref_wfn
        self.nmo = self.ref_wfn.nmo()
        self.no = self.ref_wfn.doccpi()[0]
        self.nv = self.nmo - self.no
        self.o = slice(0,self.no)
        self.v = slice(self.no,self.nmo)

        # setup everything needed for CIS and TDCIS
        Ca = self.ref_wfn.Ca_subset("AO","ACTIVE") # ACTIVE and ALL should be the same
        eps = self.ref_wfn.epsilon_a().to_array()
        mints = psi4.core.MintsHelper(self.ref_wfn.basisset())
        ERI = mints.mo_eri(Ca,Ca,Ca,Ca).to_array() # (pr|qs)
        ERI = ERI.swapaxes(1,2)                    # <pq|rs>
        ERI = 2*ERI - ERI.swapaxes(2,3)              # 2<pq|rs> - <pq|sr>
        
        excitations = [] # build all possible i,a excitations
        for i in range(self.no):
            for a in range(self.no, self.nmo):
                excitations.append((i, a))

        # dipole integrals
        # transform dipole integrals
        ao_mu = mints.ao_dipole()
        mo_mu = []
        for ax in range(3):
            mo_mu.append(Ca.to_array().T @ ao_mu[ax].to_array() @ Ca.to_array())

        self.mo_mu = np.asarray(mo_mu)
        self.eps = eps
        self.ERI = ERI
        self.excitations = excitations
        self.nuc_dip = np.array([self.ref_wfn.molecule().nuclear_dipole()[i]
            for i in range(3)])
        
    def do_cis(self):
        '''
        run a CIS calculation 
        '''
        H = self._build_H()
        
        ECIS, CCIS = np.linalg.eigh(H)
        CCIS = 2.**0.5 * CCIS # normalize to 2 ala Psi4
        
        return ECIS,CCIS

    def _build_H(self):
        '''
        Build the (CIS) Hamiltonian
        '''
        no = self.no
        nv = self.nv
        e = self.ref_e
        H = np.zeros((no*nv,no*nv)) # build up H[p,q]
        for p,bra in enumerate(self.excitations):
            i,a = bra
            for q,ket in enumerate(self.excitations):
                j,b = ket
                H[p,q] = (self.eps[a]-self.eps[i])*(i==j)*(a==b) + self.ERI[i,b,a,j]
        return H

    def dipole(self,d):
        '''return x,y,z dot products of mu[x,y,z] and d'''
        mu = np.zeros(3)
        for a in range(3):
            mu[a] = self.mo_mu[a].flatten().dot(d.flatten())
        return mu

    def density(self,C1,state="0n",C2=None):
        '''
        return a general 1RDM in the MO basis
        state: "0n", "n0", "00", "nn", or "nm"
        NOTE: THIS IS TESTED FOR CIS ONLY
        '''
        d = np.zeros((self.nmo,self.nmo))
        state = state.lower()
        if state == "00":
            d[self.o,self.o] += 2.0*np.eye(self.no)
        elif state == "n0":
            X = C1.reshape((self.no,self.nv))
            d[self.v,self.o] += X.T
        elif state == "0n":
            X = C1.reshape((self.no,self.nv))
            d[self.o,self.v] += X
        elif state == "nm":
            X1 = C1.reshape((self.no,self.nv))
            X2 = C2.reshape((self.no,self.nv)) 
            d[self.o,self.o] += -1 * X2 @ X1.T
            d[self.v,self.v] += X1.T @ X2
            d /= 2
        elif state == "nn":
            X = C1.reshape((self.no,self.nv))
            d[self.o,self.o] += -1 * X @ X.T
            d[self.v,self.v] += X.T @ X
            d /= 2
            d[self.o,self.o] += 2.0*np.eye(self.no)
        else:
            e = "state type {} not recognized.\nplease use state='00','0n','n0','nm', or 'nn'"
            raise(Exception(e))
        return d

    def transition_dipole(self,A):
        '''
        build transition dipole matrix from CI stationary states A
        A should NOT include the ground state
        '''
        trans_list = []
        eps_list = []
        U_list = []
        nstates = A.shape[1]
        for a in range(3):
            trans_list.append(np.zeros((nstates+1,nstates+1)))
    
        # reference dipole
        d = self.density(None,state="00")
        for a in range(0,3):
            trans_list[a][0,0] = self.mo_mu[a].flatten().dot(d.flatten()) + self.nuc_dip[a]
    
        # 0n and n0
        for x in range(nstates):
            d = self.density(A[:,x],state="0n")
            for a in range(3):
                trans_list[a][0,x+1] = self.mo_mu[a].flatten().dot(d.flatten())
            d = self.density(A[:,x],state="n0")
            for a in range(3):
                trans_list[a][x+1,0] = self.mo_mu[a].flatten().dot(d.flatten())
    
        # nn
        for x1 in range(nstates):
            for x2 in range(nstates):
                if x1 == x2:
                    d = self.density(C1=A[:,x1],C2=A[:,x2],state="nn")
                    for a in range(3):
                        trans_list[a][x1+1,x2+1] = self.mo_mu[a].flatten().dot(d.flatten()) + self.nuc_dip[a]
                else:
                    d = self.density(C1=A[:,x1],C2=A[:,x2],state="nm")
                    for a in range(3):
                        trans_list[a][x1+1,x2+1] = self.mo_mu[a].flatten().dot(d.flatten())

        return trans_list

