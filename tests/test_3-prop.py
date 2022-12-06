import psi4
import numpy as np
from rtci import ci,prop
fs2au = 41.34144728138643
geom = """
       O
       H 1 1.1
       H 1 1.1 2 104.5
       symmetry c1
       noreorient
       nocom"""
def test_cis_prop():
    # run reference (SCF) with psi4
    psi4.core.be_quiet()
    psi4.set_options({"basis":"sto-3g",
                      "scf_type":"pk",
                      "e_convergence":1e-8,
                      "d_convergence":1e-8,
                      })
    mol = psi4.geometry(geom)
    e,wfn = psi4.energy('scf',return_wfn=True)
    
    # run CIS
    cis = ci.ciwfn(e,wfn)
    E_cis,A_cis = cis.do_cis()

    # compute transition dipole matrix and diagonalize
    nstates = 9 # 9 excited states (plus ground)
    trans_mu = cis.transition_dipole(A_cis[:,:nstates])
    eps,U = prop.build_U(trans_mu)

    # TDCI setup
    E = np.hstack(([0],E_cis[:nstates])) # relative energies
    C = np.array([1]+[0]*nstates) # coefficients
    t = 0 # start time
    dt = 0.0005*fs2au # timestep
    tf = 0.002*fs2au # end time
    f = 0.01 # field strength
    V = prop.lasers.delta(f,center=0.001*fs2au) # callable field object

    while t<tf:
        t += dt
        field = [V(t)]*3 # isotropic field
        C = prop.prop_iter(C,U,eps,field,E,dt)

    ref_C = np.array([ 0.9999999050732+0.0001969607773j, 0.0000003155373+0.0000532203885j,
                       0.0000000023186-0.0000000000159j,-0.0000010600886-0.0001379109834j,
                      -0.0000007867659-0.0000896111596j, 0.0000024613741+0.0002628347006j,
                       0.0000019654855+0.0001628777759j, 0.0000000022919-0.0000000000297j,
                       0.0000015918847+0.0001136321431j,-0.0000016752741-0.0001124166170j])

    np.allclose(ref_C,C)
