import numpy as np

from parameters import Parameters
from hamiltonian import Hamiltonian
from states import States
from minimizer import Minimization
from penalty import Penalty
from write_functions import WriteStatic
from scf import scf_method


def run(parameters, write_static: WriteStatic):

    print('===========================================')
    print('                                           ')
    print('     Pauli Dressed Hartree Fock Solver     ')
    print('                                           ')
    print('===========================================')
    print('')

    # theory level
    hartree = False
    exchange = True

    # load restart information?
    from_scratch = False

    # calc unoccupied orbitals?
    unocc = False

    # parameters SCF
    scf_flag = True
    scf_iter_max = 80

    # parameters minimization
    method = 1  # 0 - cg, 1 - diagonalization (only possible for g_condition_flag = False)
    tolerance = 1.0e-4

    # add electronic N-representability as constraint
    enforce_pauli_constraints = False
    # also apply to lowest eigenstate? Default False, much more stable!!
    penalize_first_state = False

    # parameter constrain iteration
    penalty_iter_max = 40

    # debug mode: prints several information during the minimization
    debug = False

    minimization = Minimization(tolerance=tolerance, scf_flag=scf_flag, scf_iter_max=scf_iter_max, method=method,
                                enforce_pauli_constraints=enforce_pauli_constraints, unocc=unocc, debug=debug,
                                penalty_iter_max=penalty_iter_max)

    # initialize Hamiltonian
    hamiltonian = Hamiltonian(parameters, hartree=hartree, exchange=exchange)

    # initialize states
    states = States(parameters, write_static)
    write_static.set_states(states)

    # initialize penalty method for pauli constraints
    if minimization.enforce_pauli_constraints:
        penalty_method = 1
        states.update_reduced_quantities()
    else:
        penalty_method = 0

    penalty = Penalty(write_static, states, penalty_method, penalize_first_state=penalize_first_state)
    write_static.set_penalty(penalty)

    # start self-consistent field iteration to find the ground state
    scf_method(write_static, hamiltonian, states, minimization, penalty, from_scratch)


def main():
    # Parameters system
    spacing = 1.0
    box_length = 30
    hop = 1.0  # 0.5 / spacing ** 2  # 2nd order FD: t= 0.5 / sp**2
    boundary = 0
    n_electrons = 2
    n_photons = 1
    lamb = 0.0
    omega = 0.1
    # parameters of double-well potential
    v_charge = n_electrons / 2
    v_epsilon = 2.0
    v_dist = 0.0

    print('===========================================')
    print('                                           ')
    print('     Pauli Dressed Hartree Fock Solver     ')
    print('                                           ')
    print('===========================================')
    print('')

    # for n_photons in range(1, 12):
    write_static = None
    # for lamb in [0.0, 0.1]:
    # for lamb in np.arange(0.00, 0.55, 0.05):  # [lambda_0]: [0.0, 0.1]:
    for v_epsilon in [0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]:
    # for v_epsilon in [0.1, 1.0, 5.0, 10.0, 15.0]:
        parameters = Parameters(spacing, box_length, hop, v_charge, v_epsilon, v_dist, boundary,
                                n_electrons, n_photons, lamb, omega)
        if write_static is None:
            write_static = WriteStatic(parameters)
        else:
            write_static.new_data_set(parameters)
        run(parameters, write_static)
    write_static.write_data_series('series_data', ['converged', 'v_epsilon', 'E_total', 'dipole_me'])
    # lambda
    # parameters = Parameters(n_sites, hop, boundary, n_electrons, n_photons, lamb, omega)
    # write_static = WriteStatic(parameters)
    # run(parameters, write_static)


if __name__ == '__main__':
    main()
