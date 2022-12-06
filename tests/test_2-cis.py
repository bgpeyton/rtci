import psi4
import numpy as np
from rtci import ci

geom = """
       O
       H 1 1.1
       H 1 1.1 2 104.5
       symmetry c1
       noreorient
       nocom"""

def test_cis_e():
    # run reference (SCF) with psi4
    psi4.core.be_quiet()
    psi4.set_options({"basis":"6-31g",
                      "scf_type":"pk",
                      "e_convergence":1e-8,
                      "d_convergence":1e-8,
                      })
    mol = psi4.geometry(geom)
    e,wfn = psi4.energy('scf',return_wfn=True)
    
    # run CIS
    cis = ci.ciwfn(e,wfn)
    E_cis,A_cis = cis.do_cis()

    # check 10 roots against psi4
    psi4.set_options({"FCI": False,
        "num_roots":11,
        "ex_level":1})
    p4e,p4wfn = psi4.energy('detci',return_wfn=True)
    
    for r in range(0,10):
        assert abs(p4wfn.variable("CI ROOT {} CORRELATION ENERGY".format(r+1)) - E_cis[r]) < 1e-7

def test_cis_mu():
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

    # compute excited state dipoles
    mu = []
    for r in range(10):
        d = cis.density(A_cis[:,r],state="nn")
        mu.append(cis.dipole(d)+cis.nuc_dip)

    # check 10 roots against psi4
    psi4.set_options({"FCI": False,
        "num_roots":11,
        "ex_level":1})
    p4e,p4wfn = psi4.properties('detci',return_wfn=True,properties=["dipole"])
    
    for r in range(0,10):
        for a in range(3):
            assert abs(p4wfn.variable("CI ROOT {} DIPOLE".format(r+1))[a] - mu[r][a]) < 1e-7

def test_cis_transition_mu():
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

    # compute transition dipole matrix
    trans_mu = cis.transition_dipole(A_cis[:,:10])

    # check 10 roots against psi4
    psi4.set_options({"FCI": False,
        "num_roots":11,
        "ex_level":1})
    p4e,p4wfn = psi4.properties('detci',return_wfn=True,properties=["dipole"])
    p4trans = [np.zeros((11,11))] * 3
    for a in range(3):
        for x in range(11):
            for y in range(x,11):
                d = p4wfn.get_opdm(x,y,"SUM", False).to_array()
                mu = []
                for a in range(3):
                    mu_a = cis.mo_mu[a].flatten().dot(d.flatten())
                    p4trans[a][x,y] = mu_a
                    p4trans[a][y,x] = mu_a
    
    for a in range(3):
        np.allclose(trans_mu[a],p4trans[a]) 
