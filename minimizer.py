import numpy as np
from copy import deepcopy

from states import Orbital, States
from line_minimization import line_minimization  # , line_minimization_fourier
from hamiltonian import Hamiltonian
from penalty import Penalty
# from write_functions import WriteStatic


class Minimization:
    # class used as data class: only stores minimization parameters and provides according read and write functions
    def __init__(self, tolerance=10e-4, scf_flag=True, scf_iter_max=40, method=0, cg_iter_max=25,
                 enforce_pauli_constraints=True, penalty_iter_max=100, no_iter_max=40, unocc=False, debug=True):

        # print debug information?
        self.debug = debug

        # SCF
        self.scf_flag = scf_flag
        if scf_flag:
            self.scf_iter_max = scf_iter_max
        else:
            self.scf_iter_max = 0
        self.method = method
        if self.method == 0:
            if enforce_pauli_constraints:
                self.cg_iter_max = cg_iter_max
            else:
                self.cg_iter_max = cg_iter_max + cg_iter_max
        elif self.method == 1:
            self.cg_iter_max = 0

        # Pauli constraints
        self.enforce_pauli_constraints = enforce_pauli_constraints
        if self.enforce_pauli_constraints:
            if not method == 0:
                print('FATAL: trying to use Pauli constraints with not valid solver. Reset to cg')
                self.method = 0
            self.penalty_iter_max = penalty_iter_max
            self.no_iter_max = no_iter_max
        else:
            self.penalty_iter_max = 0

        # tolerances
        if not self.enforce_pauli_constraints:
            self.__scf_tolerance = tolerance  # without Puali constraints, the overall tolerance is for the scf
            self.__cg_tolerance = self.__scf_tolerance * 1.0e-1
            self.__line_min_tolerance = self.__scf_tolerance * 1.0e-2
        else:
            self.__scf_tolerance = 1.0e-3  # start with bigger tolerance, will be decreased later
            # TODO: increase of scf_tolerance does not happen, if system doesn't need the multiplier: no conv!
            self.__no_tolerance = self.__scf_tolerance * 1.0e-1
            self.__cg_tolerance = self.__scf_tolerance * 1.0e-2
            self.__line_min_tolerance = self.__scf_tolerance * 1.0e-3
            self.tolerance = tolerance  # overall tolerance now on penalty

        self.__converged = False

        # calculate unoccupied orbitals?
        self.unocc = unocc

    def update_scf_tolerance(self, scf_tolerance_factor):
        self.__scf_tolerance *= scf_tolerance_factor
        if not self.enforce_pauli_constraints:
            self.__cg_tolerance = self.__scf_tolerance * 1.0e-1
            self.__line_min_tolerance = self.__scf_tolerance * 1.0e-2
        else:
            self.__no_tolerance = self.__scf_tolerance * 1.0e-1
            self.__cg_tolerance = self.__scf_tolerance * 1.0e-2
            self.__line_min_tolerance = self.__scf_tolerance * 1.0e-3

    def set_converged(self, converged):
        self.__converged = converged

    def check_converged(self):
        return self.__converged

    def get_cg_tolerance(self):
        return self.__cg_tolerance

    def get_scf_tolerance(self):
        return self.__scf_tolerance

    def get_lin_min_tolerance(self):
        return self.__line_min_tolerance

    def get_no_tolerance(self):
        return self.__no_tolerance


def conjugate_gradient(minimization: Minimization, write_static, hamiltonian: Hamiltonian, states: States,
                       index_orb: int, penalty=None):
    # optimize orbital index_orb by conjugate gradient method
    # following Payne paper (equation numbers refer to paper)

    psi = states.get_state(index_orb)
    psi_old = deepcopy(psi)

    if minimization.debug:
        print("###################################################")
        print("Optimization of state # %d" % index_orb)
        print(psi.get_orbital().round(2))
        print('')

    states.orthogonalize(psi, index_orb, norm=True)
    # print('psi after ort', psi.get_orbital().round(4))

    # gradient part of hamiltonian
    hamiltonian_gradient = Orbital(psi.dimension, psi.index)
    psi.hamiltonian_apply(hamiltonian, hamiltonian_gradient)
    # print("ham_grad", hamiltonian_gradient.get_orbital())

    # eigenvalue (5.11)
    psi.update_eigenvalue(hamiltonian)
    energy_eval = psi.get_eigenvalue()
    eig_val = energy_eval
    # print('eig_val', eig_val)

    # penalty gradient
    states.set_state(psi)  # update psi in states to properly calculate derived quantities
    states.update_gamma_electronic()
    if penalty.method == 1:
        if not (not penalty.penalize_first_state and index_orb == 0):
            penalty.update_gradient(index_orb)
            penalty.update_eigenvalue(index_orb)
            eig_val += penalty.eigenvalues[index_orb]  # add part from penalty term to eigenvalue
            # print('penalty gradient', penalty.gradients[index_orb].get_orbital().round(3))
            # print('eig_val pen', penalty.eigenvalues[index_orb])
            # print('penalty derivative', penalty.derivative)
    psi.calc_residue(hamiltonian, penalty, h_psi=hamiltonian_gradient)
    # print('residue', psi.get_residue().round(7))

    # extra convergence criterion discussed in Sec. V.B.6
    first_delta = 0.0
    energy_old = psi.energy(hamiltonian, penalty)

    # extra states penalty class for line_minimization (very expensive method)
    states_lm = deepcopy(states)
    penalty_lm = Penalty(write_static, states_lm, penalty.method)

    cg_iter = 0
    cg = Orbital(psi.dimension, psi.index)  # initialize cg vector
    gradient = Orbital(psi.dimension, psi.index)  # initialize full gradient vector
    while True:
        # steepest descend (5.10)
        grad_orbital = hamiltonian_gradient.get_orbital() - eig_val * psi.get_orbital()

        # print('grad_orbital before penalty add', grad_orbital.round(4))
        if penalty.method == 1:
            if not (not penalty.penalize_first_state and index_orb == 0):
                grad_orbital += penalty.gradients[index_orb].get_orbital()
        grad_orbital *= -1
        gradient.set_orbital(grad_orbital)

        # print('gradient before orthog', gradient.get_orbital().round(4))
        # orthogonalize (5.18)
        states.orthogonalize(gradient, gradient.index + 1, norm=False)
        # print('gradient after orthog', gradient.get_orbital().round(4))

        # conjugation (Fletcher-Reeves)
        gg = np.dot(gradient.get_orbital(), gradient.get_orbital())
        if cg_iter == 0:
            gg0 = gg
            cg.set_orbital(gradient.get_orbital())
            first_delta = abs(energy_old - psi.energy(hamiltonian, penalty))
        else:
            cg_factor = gg / gg0

            # save for next iteration
            gg0 = gg

            # conjugate gradient (5.19)
            cg_orbital = gradient.get_orbital() + cg_factor * cg.get_orbital()
            cg.set_orbital(cg_orbital)

            # orthogonalize to psi
            cg.project(psi)

        # symmetrize cg
        # cg_orbital_sym = cg.get_orbital()
        # if index_orb % 2 == 0:
        #     cg_orbital_sym = cg_orbital_sym * states.symmetry_even
        # else:
        #     cg_orbital_sym = cg_orbital_sym * states.symmetry_odd
        # cg.set_orbital(cg_orbital_sym)

        # define also normalized version of cg vector for later convenience
        cg_normalized = deepcopy(cg)
        cg_normalized.normalize()

        # print('')
        # print('before state update')
        # print('hamiltonian_gradient', hamiltonian_gradient.get_orbital().round(4))
        # print('ortho gradient', (- eig_val * psi.get_orbital()).round(4))
        # if index_orb > 0:
        #     print('penalty gradient', penalty.gradients[index_orb].get_orbital().round(4))
        # print('gradient', gradient.get_orbital().round(4))
        # print('cg', cg.get_orbital().round(4))
        # print('cg_normalized', cg_normalized.get_orbital().round(4))
        # print('NO 1st occ', states.NON[-1].round(3), states.NO[:, -1].round(3))
        # print('NO 2nd occ', states.NON[-2].round(3), states.NO[:, -2].round(3))
        # print('')

        # line minimization
        if cg.get_norm() > 1.0e-8:
            # TODO: line_minimization with exact functional: infeasible for large system!
            if penalty.method == 1:
                penalty_lm.update_mu(penalty.get_mu())
                penalty_lm.update_multiplier(penalty.get_multiplier())

            alpha = line_minimization(minimization, psi, cg_normalized, states_lm, hamiltonian, penalty_lm)

            # updates
            # psi
            psi.mix_state(cg_normalized, psi, alpha)

            # hamiltonian gradient
            psi.hamiltonian_apply(hamiltonian, hamiltonian_gradient)

            # hamiltonian eigenvalue
            psi.update_eigenvalue(hamiltonian)
            energy_eval = psi.get_eigenvalue()
            eig_val = energy_eval
            # print('new hamiltonian_gradient.norm', hamiltonian_gradient.get_norm()

            # penalty terms
            states.set_state(psi)
            states.update_gamma_electronic()
            if penalty.method == 1:
                if not (not penalty.penalize_first_state and index_orb == 0):
                    penalty.update_gradient(index_orb)
                    penalty.update_eigenvalue(index_orb)
                    eig_val += penalty.eigenvalues[index_orb]

            # residue
            psi.calc_residue(hamiltonian, penalty, h_psi=hamiltonian_gradient)
            states.set_state(psi)  # TODO: better handling of this, such that we do not have to call st_state again

        else:
            break

        # print('after state update')
        # print('hamiltonian_gradient', hamiltonian_gradient.get_orbital().round(4))
        # print('ortho gradient', (- eig_val * psi.get_orbital()).round(4))
        # if index_orb > 0:
        #     print('penalty gradient', penalty.gradients[index_orb].get_orbital().round(4))
        # # print('gradient', gradient.get_orbital().round(4))
        # print('')
        # print(penalty.derivative)
        # print(states.get_g_condition())
        # print(psi.get_orbital().round(2))
        # print(states.gamma_electronic)
        # print(states.NO[-1])
        # print(states.NO[-2])

        # hamiltonian.build_energy_mat(states.gamma)
        functional_value = states.energy_total(hamiltonian, penalty)

        # if minimization.debug:
        print("State", index_orb, "Iteration:", cg_iter, "NON>1",
              states.get_NON()[np.where(states.get_NON() > 1)].round(4),
              "alpha", round(alpha, 5), "eig_val", eig_val.round(5), "eig_val_pen",
              penalty.eigenvalues[index_orb].round(5), "residue", psi.get_residue().round(7),
              "func_value", functional_value.round(6), "cg_norm", cg.get_norm().round(5))

        # #### convergence

        # cg loop
        state_error = np.linalg.norm(psi.get_orbital() - psi_old.get_orbital())

        # extra criterion discussed in Sec. V.B.6
        delta = abs(functional_value - energy_old)
        if delta < first_delta * 0.1:  # or state_error > 0.03:
            print('Exit loop: energy or state change too strong, delta=%1.5f, state_error=%1.5f' %
                  (delta, state_error))
            # print('optimized state:')
            # print(psi.get_orbital().round(3))
            # print('')
            break

        if state_error < minimization.get_cg_tolerance() or cg_iter >= minimization.cg_iter_max:
            print('optimized state:')
            print(psi.get_orbital().round(2))
            print('')
            if cg_iter < minimization.cg_iter_max:
                print("Iter. = ", cg_iter, "Error = ", state_error, "State #", index_orb, "is converged!!")
            else:
                print("Iter. = ", cg_iter, "Error = ", state_error, "State #", index_orb, "is NOT converged!!")
            break

        psi_old = deepcopy(psi)
        cg_iter += 1


def matrix_diagonalization(hamiltonian: Hamiltonian, states: States, penalty):

    if not penalty.method == 0:
        print('Fatal Error: diagonalization method chosen with penalty method!!')
    else:
        # diagonalize one-body Hamiltonian (only valid for HF and DFT)
        eigenvalues_new, states_new = np.linalg.eigh(hamiltonian.hamiltonian_mat)

        states.matrix_to_states(states_new)


def minimizer_method(minimization: Minimization, write_static, hamiltonian: Hamiltonian, states: States, penalty):
    if minimization.method == 0:
        if not minimization.check_converged():
            orb_range = range(hamiltonian.occupied_orbitals)
        else:
            if minimization.unocc:
                orb_range = range(states.occupied_orbitals, states.dim_coupled - 1)
                penalty.method = False
            else:
                orb_range = []

        # states.update_reduced_quantities()
        for index_orb in orb_range:
            conjugate_gradient(minimization, write_static, hamiltonian, states, index_orb, penalty)

        # add last state by orthogonalization
        if minimization.check_converged() and minimization.unocc:
            psi_last = states.get_state(states.dim_coupled - 1)
            states.orthogonalize(psi_last, states.dim_coupled)
            states.set_state(psi_last)

    elif minimization.method == 1:
        matrix_diagonalization(hamiltonian, states, penalty)
