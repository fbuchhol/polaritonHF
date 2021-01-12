import numpy as np
from copy import deepcopy

from hamiltonian import Hamiltonian
from states import States
from minimizer import Minimization, minimizer_method
from penalty import Penalty
from write_functions import WriteStatic, print_results


def scf_method(write_static: WriteStatic, hamiltonian: Hamiltonian, states: States, minimization: Minimization,
               penalty: Penalty, from_scratch=False, random_init=True):

    # initialization
    # default: randomized initial states
    if not from_scratch:
        states_load_flag = states.load_states_file()
        if states_load_flag:
            random_init = False
    if random_init:
        states.calc_eigenvalues_all(hamiltonian)
        states.initialize_random(angle=1.0e-1)

    # prepare states according to symmetry
    # print('initial orbital symmetrization')
    # for kk in range(states.dim_coupled):
    #     state = states.get_state(kk)
    #     orbital = state.get_orbital()
    #     # print('index', kk)
    #     # print('orbital before symmetrization:', orbital)
    #     if kk % 2 == 0:
    #         print(states.symmetry_even)
    #         orbital = orbital * states.symmetry_even
    #     else:
    #         orbital = orbital * states.symmetry_odd
    #     # print('orbital after symmetrization:', orbital)
    #     state.set_orbital(orbital)
    #     state.normalize()
    #     states.set_state(state)

    if minimization.debug:
        print('Initial orbitals')
        for ii in range(states.dim_coupled):
            print(states.get_state(ii).get_orbital().round(3))

    states.update_reduced_quantities()
    hamiltonian.build_hamiltonian_mat(states.get_gamma_hamiltonian())
    hamiltonian.build_energy_mat(states.get_gamma_hamiltonian())
    states.calc_eigenvalues_all(hamiltonian)
    if minimization.debug:
        print('Initial eigenvalues')
        print(states.get_eigenvalues())
        print('')
    # st.save_states_file('initial_state')

    # read penalty restart information
    if not from_scratch:
        penalty.read_data()
        print('')

    #
    #
    # start minimization procedure
    #
    #

    if penalty.method == 1:
        energy_old = states.energy_total(hamiltonian, penalty_class=penalty)
        while True:
            print(
                '#####################################################################################################')
            print('Penalty Iteration: %d' % penalty.get_iter())
            print("Current penalty parameter mu =%1.4f" % penalty.get_mu())
            print('')

            gamma_old = deepcopy(states.get_gamma_hamiltonian())
            scf_iter = 0
            while True:
                print('######################################################################')
                print('SCF Iteration: %d' % scf_iter)
                if minimization.debug:
                    print('Current One-body Hamiltonian:')
                    for ii in range(states.dim_coupled):
                        print(hamiltonian.hamiltonian_mat[ii, :].round(2))
                    print('')

                # print('pen_apply_test')
                # psi = states.get_state[1]
                # print(penalty.apply_penalty_state(psi, 1).get_orbital())
                # penalty_matrix = penalty.build_penalty_matrix(-1)
                # penalty_matrix_apply = penalty.penalty_derivative[-1] * np.dot(penalty_matrix, psi.get_orbital())
                # print(penalty_matrix_apply)
                # print('')
                # print('Penalty matrix')
                # print(penalty_matrix)
                # print('')
                no_iter = 0
                gamma_electronic_old = deepcopy(states.gamma_electronic_NO)
                while True:
                    print('######################################################################')
                    print('NO Iteration: %d' % no_iter)

                    minimizer_method(minimization, write_static, hamiltonian, states, penalty)

                    # convergence NO
                    # update gamma_electronic
                    states.update_gamma_electronic_NO(mixing_parameter=0.3)

                    gamma_electronic_error = np.linalg.norm(states.get_gamma_electronic_NO() - gamma_electronic_old)
                    print('NO error in gamma_electronic:', gamma_electronic_error)
                    print('')

                    if gamma_electronic_error <= minimization.get_no_tolerance() or \
                            no_iter >= minimization.no_iter_max:
                        if no_iter < minimization.no_iter_max:
                            print("SCF Iter =", scf_iter, "NO iter. = ", no_iter, "NO Error = ", gamma_electronic_error,
                                  "NO loop is Converged!!")
                        else:
                            print("SCF Iter =", scf_iter, "NO iter. = ", no_iter, "NO Error = ", gamma_electronic_error,
                                  "NO loop is NOT Converged!!")
                        print('')
                        break

                    gamma_electronic_old = deepcopy(states.get_gamma_electronic_NO())
                    no_iter += 1

                # convergence SCF
                # update gamma
                if penalty.get_iter() == 0:
                    states.update_gamma_hamiltonian()  # no mixing for the first scf
                else:
                    states.update_gamma_hamiltonian(mixing_parameter=0.4)

                hamiltonian.build_hamiltonian_mat(states.get_gamma_hamiltonian())
                hamiltonian.build_energy_mat(states.get_gamma_hamiltonian())

                # matrix norm -> ||gamma - gamma_old||
                gamma_error = np.linalg.norm(states.get_gamma_hamiltonian() - gamma_old)
                print('SCF error in gamma:', gamma_error)
                print('')

                if gamma_error <= minimization.get_scf_tolerance() or scf_iter >= minimization.scf_iter_max:
                    if scf_iter < minimization.scf_iter_max:
                        print("mu =", penalty.get_mu(), "iter. = ", scf_iter, "SCF Error = ", gamma_error,
                              "SCF is Converged!!")
                    else:
                        print("mu =", penalty.get_mu(), "iter. = ", scf_iter, "SCF Error = ", gamma_error,
                              "SCF is NOT Converged!!")
                    print('')
                    break

                gamma_old = deepcopy(states.get_gamma_hamiltonian())
                scf_iter += 1

            # convergence penalty
            states.update_gamma_hamiltonian()  # no mixing if scf is converged
            hamiltonian.build_hamiltonian_mat(states.get_gamma_hamiltonian())
            hamiltonian.build_energy_mat(states.get_gamma_hamiltonian())

            print('penalty iteration convergence')
            # residue_sum: for a converged result, we want an eigenstate of the one-body Hamiltonian
            energy_error = abs(states.energy_total(hamiltonian, penalty_class=penalty) - energy_old)
            residue_sum = sum(states.get_residues()[:states.occupied_orbitals])
            max_error = np.amax([energy_error, residue_sum])
            print('residue_error', residue_sum)
            print('energy_error', energy_error)
            print('total_energy', states.energy_total(hamiltonian, penalty))

            penalty_function_norm = penalty.get_constraint_function_norm(update_constraint_function=True)

            if (max_error <= minimization.tolerance or penalty.get_iter() >= minimization.penalty_iter_max or
                    penalty.get_mu() > penalty.mu_max):  # or penalty_function_norm < 10e-9):
                if (penalty.get_iter() < minimization.penalty_iter_max and penalty.get_mu() < penalty.mu_max and
                        max_error <= minimization.tolerance):
                    print("Iter. = ", penalty.get_iter(), "penalty error = ", max_error, "< tol = ",
                          minimization.tolerance, "penalty iteration is Converged!!")
                    minimization.set_converged(True)
                    # calculate unoccupied orbitals
                    minimizer_method(minimization, write_static, hamiltonian, states, penalty)
                else:
                    print("Iter. = ", penalty.get_iter(), "penalty error = ", max_error, "penalty function norm = ",
                          penalty_function_norm, "penalty iteration is NOT Converged!!")
                break

            # update penalty parameters
            penalty.update_parameters(minimization)

            # save quantities for convergence test
            energy_old = states.energy_total(hamiltonian, penalty)

            # update iteration counter
            penalty.iter_increment()
    else:  # no penalty
        gamma_old = deepcopy(states.get_gamma_hamiltonian())
        scf_iter = 0
        while True:
            hamiltonian.build_hamiltonian_mat(states.get_gamma_hamiltonian())
            hamiltonian.build_energy_mat(states.get_gamma_hamiltonian())

            print('######################################################################')
            print('SCF Iteration: %d' % scf_iter)
            print('')
            print('Current One-body Hamiltonian:')
            hamiltonian_non_zero = np.where(np.absolute(hamiltonian.hamiltonian_mat) > 1e-5)
            print(hamiltonian.hamiltonian_mat[hamiltonian_non_zero].round(4))
            print('')
            if minimization.method == 1:
                print('Current set of states')
                if minimization.debug:
                    range_states = states.dim_coupled
                else:
                    range_states = states.occupied_orbitals
                for ii in range(range_states):
                    print(ii, states.get_state(ii).get_orbital().round(3))
                print('')
                print('NON', states.NON.round(3))
                print('')
                states.calc_eigenvalues_all(hamiltonian)
                total_energy = states.energy_total(hamiltonian, penalty)  # without adding the penalty part!
                print('total energy:')
                print(total_energy)

            minimizer_method(minimization, write_static, hamiltonian, states, penalty)

            # convergence SCF
            # max(||gamma - gamma_old||, residue_sum)
            if scf_iter == 0:
                states.update_gamma_hamiltonian()  # no mixing for the first scf
            else:
                states.update_gamma_hamiltonian(mixing_parameter=0.5)
            gamma_error = np.linalg.norm(states.get_gamma_hamiltonian() - gamma_old)
            print('SCF errors:')
            print('gamma error:', gamma_error)

            if gamma_error <= minimization.get_scf_tolerance() or scf_iter >= minimization.scf_iter_max:
                if scf_iter < minimization.scf_iter_max:
                    print("iter. = ", scf_iter, "SCF Error = ", gamma_error, "SCF is Converged!!")
                    minimization.set_converged(True)
                else:
                    print("iter. = ", scf_iter, "SCF Error = ", gamma_error, "SCF is NOT Converged!!")
                break

            gamma_old = deepcopy(states.get_gamma_hamiltonian())
            scf_iter += 1

    # calculate solution data
    states.update_reduced_quantities()
    hamiltonian.build_hamiltonian_mat(states.get_gamma_hamiltonian())
    hamiltonian.build_energy_mat(states.get_gamma_hamiltonian())
    states.calc_eigenvalues_all(hamiltonian)
    if penalty.method == 1:
        for ii in range(states.dim_coupled):
            penalty.update_gradient(ii)
            penalty.update_eigenvalue(ii)

    # print results of calculation
    print_results(minimization, states, hamiltonian, penalty)
    write_static.calc_gs_quantities(hamiltonian, print_flag=True)

    # update static information
    write_static.read_status(minimization)
    write_static.read_states()
    write_static.read_penalty()

    # write static information
    write_static.write_data()

    print('write current set of states')
    states.save_states_file()
    if penalty.method == 1:
        penalty.write_data()

    # parameters for linear response
    # eta = 0.01
    # omega_range = np.arange(0.0, 4.5, 0.01)
    # omega_cav_range = np.arange(0.0,2.51,0.25)
    # omega_cav_range = np.arange(0.5,0.50001,0.01)
