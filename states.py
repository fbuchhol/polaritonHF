import numpy as np
import math
import os

from functools import reduce
from copy import deepcopy

from parameters import Parameters
from hamiltonian import Hamiltonian
# from write_functions import WriteStatic
# from penalty import Penalty


class Orbital:
    # TODO: introduce derived classes for state-orbital, gradient-orbital, penalty-orbital
    def __init__(self, dimension, index=-1):
        self.dimension = dimension
        self.index = index
        self.__orbital = np.zeros(dimension)
        if not index == -1:
            self.__orbital[index] = 1.0
        self.__norm = 1.0
        self.__eigenvalue = 0.0
        self.__residue = 1.0

    def normalize(self):
        norm = np.linalg.norm(self.__orbital)
        self.__orbital /= norm
        self.__norm = 1.0

    def update_norm(self):
        self.__norm = np.linalg.norm(self.__orbital)

    def get_norm(self):
        return self.__norm

    def get_orbital(self):
        return deepcopy(self.__orbital)

    def set_orbital(self, orbital_new):
        self.__orbital = orbital_new
        self.update_norm()

    def inner_product(self, psi):
        inner_product_psi = np.dot(self.__orbital, psi.get_orbital())
        return inner_product_psi

    def project(self, psi_project):
        inner_prod = self.inner_product(psi_project)
        self.__orbital -= inner_prod * psi_project.get_orbital()
        self.update_norm()

    def mix_state(self, psi_2, psi_new, angle):
        psi_new_orbital = math.cos(angle) * self.__orbital + math.sin(angle) * psi_2.get_orbital()
        psi_new.set_orbital(psi_new_orbital)

    def calc_expectation_value(self, operator):
        expectation_value = reduce(np.dot, (self.__orbital, operator, self.__orbital))
        return expectation_value

    def update_eigenvalue(self, hamiltonian):
        self.__eigenvalue = self.calc_expectation_value(hamiltonian.hamiltonian_mat)

    def get_eigenvalue(self):
        return self.__eigenvalue

    def hamiltonian_apply(self, hamiltonian, psi_apply):
        psi_apply_orbital = np.dot(hamiltonian.hamiltonian_mat, self.__orbital)
        psi_apply.set_orbital(psi_apply_orbital)

    def energy(self, hamiltonian: Hamiltonian, penalty_class, update_eigenvalue=True):
        # calculates energy change in linear approximation of the Hamiltonian
        # penalty is calculated exactly (though penalty function needs to be updated before)

        if update_eigenvalue:
            self.update_eigenvalue(hamiltonian)
        energy = self.__eigenvalue

        if penalty_class.method == 1:
            if not (not penalty_class.penalize_first_state and self.index == 0):
                energy += penalty_class.barrier_function()

        return energy

    def calc_residue(self, hamiltonian, penalty_class, h_psi=None):
        eigenvalue = self.__eigenvalue

        if h_psi is None:
            h_psi = Orbital(self.dimension, self.index)
            self.hamiltonian_apply(hamiltonian, h_psi)
        orbital = h_psi.get_orbital()
        # print('h_psi in res', orbital.round(4))
        # print('eval in res', eigenvalue.round(4))

        if penalty_class.method == 1:
            if not (not penalty_class.penalize_first_state and self.index == 0):
                orbital += penalty_class.gradients[self.index].get_orbital()
                eigenvalue += penalty_class.eigenvalues[self.index]
        # print('g_psi in res', penalty_class.gradients[self.index].get_orbital().round(4))
        # print('eval pen in res', penalty_class.eigenvalues[self.index].round(4))
        # print('eig_val complete', eigenvalue.round(4))
        # print('')
        self.__residue = np.linalg.norm(orbital - eigenvalue * self.__orbital)

    def get_residue(self):
        return self.__residue


class States(Parameters):

    # TODO: states objects shouldn't be changed.
    # TODO: instead of being a normal class, States and Hamiltonian are actually singletons
    def __init__(self, parameters, write_static):
        super().__init__(parameters.spacing, parameters.box_length, parameters.hop,
                         parameters.v_charge, parameters.v_epsilon, parameters.v_dist,
                         parameters.boundary, parameters.n_electrons, parameters.n_photons,
                         parameters.lamb, parameters.omega)

        # orbital occupations for restricted formalism
        self.occupations = np.zeros(self.dim_coupled)
        self.occupations[:self.occupied_orbitals] = np.ones(self.occupied_orbitals) * 2.0

        # state converged? # TODO: use this parameter to skip unnecessary state iterations
        self.converged = np.full(self.dim_coupled, False)

        # restart information directory
        self.restart_file = ''
        self.update_restart_file(write_static)

        # states initialized to eigenstates of uncoupled problem
        self.__states = np.empty(self.dim_coupled, dtype=Orbital)
        for II in range(self.dim_coupled):
            self.__states[II] = Orbital(self.dim_coupled, II)

        # 1RDM of coupled space: 2 "versions"
        # a) gamma: can be updated within the minimization to e.g. calculate the exact penalty function
        self.__gamma = np.zeros((self.dim_coupled, self.dim_coupled))
        self.build_gamma()
        # b) gamma_hamiltonian: is only updated after a full scf cycle to build a new Hamiltonian
        self.__gamma_hamiltonian = deepcopy(self.__gamma)
        # 1RDM of electronic space in
        # a) energy basis:
        # # 1) gamma_electronic: to calculate approximate NON_i = <NO_i | gamma_electronic | NO_i >
        self.gamma_electronic = np.zeros((self.dim_electrons, self.dim_electrons))
        self.build_gamma_electronic()
        # # 2) gamma_electronic_NO: only for diagonalization and mixing, remains constant during state minimization
        self.gamma_electronic_NO = deepcopy(self.gamma_electronic)
        # b) eigen basis: NO, NON
        self.NO = np.zeros((self.dim_electrons, self.dim_electrons))
        self.NON = np.zeros(self.dim_electrons)
        self.calc_natural_orbitals()

        # symmetry operators
        self.symmetry_even = np.ones(self.dim_coupled)
        self.symmetry_even[1::2] = 0
        self.symmetry_odd = np.ones(self.dim_coupled)
        self.symmetry_odd[::2] = 0

    def get_state(self, ist: int):
        return deepcopy(self.__states[ist])

    def set_state(self, psi: Orbital):
        self.__states[psi.index] = deepcopy(psi)

    def states_to_matrix(self):
        states_matrix = np.zeros((self.dim_coupled, self.dim_coupled))
        for ii in range(self.dim_coupled):
            states_matrix[:, ii] = self.__states[ii].get_orbital()
        return states_matrix

    def matrix_to_states(self, states_matrix):
        for ii in range(self.dim_coupled):
            self.__states[ii].set_orbital(states_matrix[:, ii])

    def update_restart_file(self, write_static):
        self.restart_file = write_static.output_dir + '/state_restart'

    def save_states_file(self):
        states_matrix = self.states_to_matrix()
        print("Save states in file:", self.restart_file)
        np.savetxt(self.restart_file, states_matrix)

    def load_states_file(self):
        if os.path.exists(self.restart_file):
            print("Load initial states file:", self.restart_file)
            states_matrix = np.loadtxt(self.restart_file)
            self.matrix_to_states(states_matrix)
            states_load_flag = True
        else:
            print('Cannot load states file from restart folder. File %s not found.' % self.restart_file)
            states_load_flag = False
        return states_load_flag

    def set_states_full(self, states_new):
        # saves matrix into a states file
        for ii in range(self.dim_coupled):
            orbital_new = states_new[:, ii]
            self.__states[ii].set_orbital(orbital_new)

    def initialize_random(self, angle=1.0):
        print("Generation of initial state")
        rand_mat = np.random.rand(self.dim_coupled, self.dim_coupled)
        sym_mat = np.triu(rand_mat)
        sym_mat = sym_mat + np.conj(sym_mat.T)
        sym_mat = angle * sym_mat + (1 - angle) * np.diag(np.sort(self.get_eigenvalues()))

        eigenvalues_test, states_test = np.linalg.eigh(sym_mat)
        # uncomment for starting point close to unphysical solution
        # states_exchange_1 = deepcopy(states_test[:, 1])
        # states_exchange_4 = deepcopy(states_test[:, 4])
        # states_test[:, 1] = deepcopy(states_exchange_4)
        # states_test[:, 4] = deepcopy(states_exchange_1)

        self.set_states_full(states_test)
        # self.eigenvalues = eigenvalues_test

    def orthogonalize(self, psi: Orbital, ist, norm=True):
        if norm:
            psi.normalize()
        for jst in range(ist):
            psi_jst = self.get_state(jst)
            psi.project(psi_jst)
        if norm:
            psi.normalize()

    def get_eigenvalues(self):
        eigenvalues = np.zeros(self.dim_coupled)
        for ii in range(self.dim_coupled):
            eigenvalues[ii] = self.__states[ii].get_eigenvalue()
        return eigenvalues

    def calc_eigenvalues_all(self, hamiltonian: Hamiltonian):
        for ii in range(self.dim_coupled):
            self.__states[ii].update_eigenvalue(hamiltonian)

    def get_residues(self):
        residues = np.zeros(self.dim_coupled)
        for ii in range(self.dim_coupled):
            residues[ii] = self.__states[ii].get_residue()
        return residues

    def energy_total(self, hamiltonian: Hamiltonian, penalty_class, add_penalty=True):
        # calculates the sum of orbital energies
        energy = 0.0
        for ii in range(self.occupied_orbitals):
            energy += self.__states[ii].calc_expectation_value(hamiltonian.energy_mat)

        if penalty_class.method == 1 and add_penalty:
            energy += penalty_class.barrier_function()

        return energy

    def energy_electronic(self, hamiltonian: Hamiltonian):
        # calculates electronic energy
        energy = np.trace(np.dot(np.diag(hamiltonian.electron_energy), self.gamma_electronic))
        return energy

    def energy_coupling(self, hamiltonian: Hamiltonian):
        # calculates coupling energy
        coupling_mat = hamiltonian.build_coupling_energy_mat(self.__gamma_hamiltonian)
        energy = 0.0
        for ii in range(self.occupied_orbitals):
            energy += self.__states[ii].calc_expectation_value(coupling_mat)
        return energy

    def energy_self(self, hamiltonian: Hamiltonian):
        # calculates dipole self-energy
        self_mat = hamiltonian.build_self_energy_mat(self.__gamma_hamiltonian)
        energy = 0.0
        for ii in range(self.occupied_orbitals):
            energy += self.__states[ii].calc_expectation_value(self_mat)
        return energy

    def energy_photon(self, hamiltonian: Hamiltonian):
        # calculates photon energy
        energy = 0.0
        for ii in range(self.occupied_orbitals):
            energy += self.__states[ii].calc_expectation_value(hamiltonian.photon_energy_matrix)

        return energy

    def build_gamma(self, hamiltonian=False, norm_one=False):
        # update polaritonic 1RDM
        gamma = np.zeros((self.dim_coupled, self.dim_coupled))

        if norm_one:
            occupations = np.ones(self.occupied_orbitals)
        else:
            occupations = self.occupations

        for ii in range(self.occupied_orbitals):
            psi = self.get_state(ii).get_orbital()
            gamma += occupations[ii] * np.outer(psi, np.conj(psi))

        if hamiltonian:
            self.__gamma_hamiltonian = gamma
        else:
            self.__gamma = gamma

    def get_gamma_hamiltonian(self):
        return deepcopy(self.__gamma_hamiltonian)

    def update_gamma_hamiltonian(self, mixing_parameter=1.0):
        print('update gamma_hamiltonian, mixing:', mixing_parameter)
        if mixing_parameter < 1.0:
            gamma_old = deepcopy(self.__gamma_hamiltonian)
            self.build_gamma(hamiltonian=True)
            self.__gamma_hamiltonian = deepcopy((1 - mixing_parameter) * gamma_old +
                                                mixing_parameter * self.__gamma_hamiltonian)
        else:
            self.build_gamma(hamiltonian=True)
        # when we update gamma for the Hamiltonian, we want also all other gamma and derived gamma be consistent
        self.__gamma = deepcopy(self.__gamma_hamiltonian)
        self.update_gamma_electronic_NO(update_gamma=False)

    def build_gamma_electronic(self, NO=False, sparse=True):
        # this is only necessary for self.gamma not for self.__gamma_hamiltonian
        reshape_gamma = deepcopy(self.__gamma.reshape(
                                (self.dim_photons, self.dim_electrons, self.dim_photons, self.dim_electrons)
                                ))
        gamma_electronic = np.zeros((self.dim_electrons, self.dim_electrons))
        for n in range(self.dim_photons):
            gamma_electronic += reshape_gamma[n, :, n, :]

        if sparse:
            set_zero = abs(gamma_electronic) < 10e-9
            gamma_electronic[set_zero] = 0

        if NO:
            self.gamma_electronic_NO = gamma_electronic
        else:
            self.gamma_electronic = gamma_electronic

    def update_gamma_electronic_NO(self, mixing_parameter=1.0, update_gamma=True):
        print('update gamma_electronic_NO, mixing:', mixing_parameter, 'update gamma:', update_gamma)
        if update_gamma:
            self.build_gamma()
        if mixing_parameter < 1.0:
            gamma_electronic_NO_old = deepcopy(self.gamma_electronic_NO)
            self.build_gamma_electronic(NO=True)
            self.gamma_electronic_NO = deepcopy((1 - mixing_parameter) * gamma_electronic_NO_old +
                                                mixing_parameter * self.gamma_electronic_NO)
        else:
            self.build_gamma_electronic(NO=True)
        self.calc_natural_orbitals()

    def update_gamma_electronic(self):
        # used to calculate approximation to NON, no mixing necessary
        self.build_gamma()
        self.build_gamma_electronic(NO=False)

    def get_gamma_electronic_NO(self):
        return deepcopy(self.gamma_electronic_NO)

    def calc_natural_orbitals(self):
        self.NON, self.NO = np.linalg.eigh(self.gamma_electronic_NO)

    def update_reduced_quantities(self):
        # updates all derived properties for the electronic subsystem of the states object
        # no mixing option here, since this function is only to be used to make all the following quantities consistent:
        # self.__gamma
        # self.__gamma_hamiltonian
        # self.gamma_electronic
        # self.gamma_electronic_NO
        # self.NO, self.NON

        self.update_gamma_hamiltonian()
        self.update_gamma_electronic()

    def get_NON(self):
        non = np.zeros(self.dim_electrons)
        for ii in range(self.dim_electrons):
            non[ii] = reduce(np.dot, (self.NO[:, ii], self.gamma_electronic, self.NO[:, ii]))
        return non

    def get_g_condition(self):
        # non_square = np.square(self.NON)
        # g_condition = self.NON - non_square
        # g_condition = 2 - self.NON
        g_condition = 2 - self.get_NON()
        return g_condition

    def get_g_condition_derivative(self):
        # g_condition_derivative = 1 - 2 * self.NON
        g_condition_derivative = - 2.0 * np.ones(self.NON.shape[0])
        return g_condition_derivative
