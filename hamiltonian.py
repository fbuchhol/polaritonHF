import numpy as np

from functools import reduce

from parameters import Parameters


class Hamiltonian(Parameters):

    def __init__(self, parameters, hartree=True, exchange=True):
        super().__init__(parameters.spacing, parameters.box_length, parameters.hop,
                         parameters.v_charge, parameters.v_epsilon, parameters.v_dist,
                         parameters.boundary, parameters.n_electrons, parameters.n_photons,
                         parameters.lamb, parameters.omega)

        print('Initialize Hamiltonian')

        # Include Hartree/exchange term?
        self.hartree = hartree
        self.exchange = exchange

        # Orbital occupation (so far only restricted formalism implemented)
        self.occupation = 2.0

        # Prefactors
        self.prefactor_coupling = - self.lamb * np.sqrt(self.omega / (2 * self.n_electrons))
        self.prefactor_self = (self.lamb**2 / 2)

        # build electron kinetic energy operator in site basis
        t_mat = np.zeros((self.n_sites, self.n_sites))
        for i in range(self.n_sites - 1):
            t_mat[i, i + 1] = t_mat[i + 1, i] = - self.hop

        if self.n_sites == 2 or self.boundary == 1:
            t_mat[0, self.n_sites - 1] = t_mat[self.n_sites - 1, 0] = - self.hop

        electron_mat = t_mat
        v_mat = np.diag(self.v_func(self.v_charge, self.v_epsilon, self.v_dist))
        electron_mat += v_mat

        # get electronic eigen basis (EEB)
        self.electron_energy, self.electron_trafo = np.linalg.eigh(electron_mat)
        print('electronic eigenenergies')
        print(self.electron_energy)
        # electron_trafo: connects EEB with real space basis (RSB),
        # usage:
        # a) mat_RSB = electron_trafo * mat_EEB * electron_trafo.T
        # b) mat_EEB = electron_trafo.T * mat_RSB * electron_trafo
        # test for correct usage of trafo (both directions):
        # print('t_mat', reduce(np.dot, (self.electron_trafo, np.diag(self.electron_energy), self.electron_trafo.T)))
        # print('electron_energy', reduce(np.dot, (self.electron_trafo.T, t_mat, self.electron_trafo)).round(4))

        # electronic dipole operator in site basis
        self.dip_op = np.zeros((self.n_sites, self.n_sites))

        for i in range(int(self.n_sites / 2)):
            self.dip_op[i, i] = -((self.n_sites / 2 - 0.5) - i) * self.spacing
            self.dip_op[self.n_sites - i - 1, self.n_sites - i - 1] = - self.dip_op[i, i]

        # transform dipole operator in EEB
        self.dipole_mat = reduce(np.dot, (self.electron_trafo.T, self.dip_op, self.electron_trafo))

        # initialize Hamiltonian matrix with only one-body part
        self.one_body_energies = self.build_one_body_energies()
        self.hamiltonian_mat = np.zeros((self.dim_coupled, self.dim_coupled))
        self.hamiltonian_mat = np.diag(self.one_body_energies)

        # initialize matrix for total energy calculation (differs from hamiltonian_mat only by Hartree and Fock terms)
        self.energy_mat = np.zeros((self.dim_coupled, self.dim_coupled))
        self.energy_mat = np.diag(self.one_body_energies)

        # photon energy matrix
        self.photon_energy_matrix = self.omega * np.kron(np.diag(np.arange(self.dim_photons)),
                                                         np.eye(self.dim_electrons))

        print('')

    def calc_dipole_mean(self, gamma):
        # calculates the dipole expectation value for a given density matrix gamma
        reshape_gamma = gamma.reshape((self.dim_photons, self.dim_electrons, self.dim_photons, self.dim_electrons))

        dipole_mean = 0.0
        for n in range(self.dim_photons):
            for i in range(self.dim_electrons):
                for j in range(self.dim_electrons):
                    dipole_mean += reshape_gamma[n, i, n, j] * self.dipole_mat[i, j]
        return dipole_mean

    def calc_polarization_mean(self, gamma):
        # calculates the polarization (a + a^+) expectation value for a given density matrix gamma
        reshape_gamma = gamma.reshape((self.dim_photons, self.dim_electrons, self.dim_photons, self.dim_electrons))

        polarization_mean = 0.0
        for i in range(self.dim_electrons):
            for n in range(self.n_photons):
                polarization_mean += reshape_gamma[n, i, n + 1, i] * np.sqrt(n + 1)
        return polarization_mean

    def build_coupling_mat(self, gamma, total_energy_flag):
        coupling_mat = np.zeros((self.dim_coupled, self.dim_coupled))

        # One-body: delta(n,n+1) sqrt(n+1) dip_me
        for n in range(self.dim_photons - 1):
            coupling_mat[n * self.dim_electrons:n * self.dim_electrons + self.dim_electrons,
                         (n + 1) * self.dim_electrons:(n + 1) * self.dim_electrons + self.dim_electrons] \
                = self.dipole_mat * np.sqrt(n + 1)

        # add dressed orbital occupation by hand for one-body part
        # coupling_mat *= 2

        # weight one-body part doubly for total energy calculation
        if total_energy_flag:
            coupling_mat *= 2

        # Hartree
        if self.hartree:
            # part one: sum over dipole of occ. states
            dipole_mean = self.calc_dipole_mean(gamma)
            for l in range(self.dim_photons - 1):
                coupling_mat[l * self.dim_electrons:l * self.dim_electrons + self.dim_electrons,
                             (l + 1) * self.dim_electrons:(l + 1) * self.dim_electrons + self.dim_electrons] \
                    += np.eye(self.dim_electrons) * np.sqrt(l + 1) * dipole_mean

            # part two: sum over photon elongation of occ. states
            polarization_mean = self.calc_polarization_mean(gamma)
            for l in range(self.dim_photons):
                coupling_mat[l * self.dim_electrons:l * self.dim_electrons + self.dim_electrons,
                             l * self.dim_electrons:l * self.dim_electrons + self.dim_electrons] \
                    += self.dipole_mat * polarization_mean

        # Exchange
        if self.exchange:
            # part one
            reshape_gamma = gamma.reshape((self.dim_photons, self.dim_electrons, self.dim_photons, self.dim_electrons))
            for l in range(self.dim_photons):
                for m in range(self.dim_photons - 1):
                    contracted_mat = np.dot(self.dipole_mat, reshape_gamma[l, :, m, :])
                    # print('contracted_mat', contracted_mat)
                    coupling_mat[l * self.dim_electrons:l * self.dim_electrons + self.dim_electrons,
                                 (m + 1) * self.dim_electrons:(m + 1) * self.dim_electrons + self.dim_electrons] \
                        -= 0.5 * contracted_mat * np.sqrt(m + 1)

            # part two
            for l in range(self.dim_photons - 1):
                for m in range(self.dim_photons):
                    contracted_mat = np.dot(reshape_gamma[l + 1, :, m, :], self.dipole_mat)
                    # print('contracted_mat', contracted_mat)
                    coupling_mat[l * self.dim_electrons:l * self.dim_electrons + self.dim_electrons,
                                 m * self.dim_electrons:m * self.dim_electrons + self.dim_electrons] \
                        -= 0.5 * contracted_mat * np.sqrt(l + 1)

        # restore correct prefactor for total energy calculation
        # if total_energy_flag:
        #     coupling_mat *= 0.5

        diagonal_coupling_mat = np.diag(coupling_mat)
        coupling_mat = np.triu(coupling_mat, k=1)
        coupling_mat = coupling_mat + np.conj(coupling_mat.T) + np.diag(diagonal_coupling_mat)

        return coupling_mat

    def build_self_mat(self, gamma, total_energy_flag):
        self_mat = np.zeros((self.dim_coupled, self.dim_coupled))

        # One-body
        dipole_mat_square = np.dot(self.dipole_mat, self.dipole_mat)
        for l in range(self.dim_photons):
            self_mat[l * self.dim_electrons:l * self.dim_electrons + self.dim_electrons,
                     l * self.dim_electrons:l * self.dim_electrons + self.dim_electrons] = dipole_mat_square

        # add dressed orbital occupation by hand for one-body part
        # self_mat *= 2

        # weight one-body part doubly for total energy calculation
        if total_energy_flag:
            self_mat *= 2  

        # Hartree
        if self.hartree:
            dipole_mean = self.calc_dipole_mean(gamma)

            for l in range(self.dim_photons):
                self_mat[l * self.dim_electrons:l * self.dim_electrons + self.dim_electrons,
                         l * self.dim_electrons:l * self.dim_electrons + self.dim_electrons] \
                    += self.dipole_mat * dipole_mean

        # Exchange
        if self.exchange:
            reshape_gamma = gamma.reshape((self.dim_photons, self.dim_electrons, self.dim_photons, self.dim_electrons))
            for l in range(self.dim_photons):
                for m in range(self.dim_photons):
                    contracted_mat = reduce(np.dot, (self.dipole_mat, reshape_gamma[l, :, m, :], self.dipole_mat))
                    self_mat[l * self.dim_electrons:l * self.dim_electrons + self.dim_electrons,
                             m * self.dim_electrons:m * self.dim_electrons + self.dim_electrons] \
                        -= 0.5 * contracted_mat

        # restore correct pre-factor for total energy calculation
        # if total_energy_flag:
        #     self_mat *= 0.5

        diagonal_self_mat = np.diag(self_mat)
        self_mat = np.triu(self_mat, k=1)
        self_mat = self_mat + np.conj(self_mat.T) + np.diag(diagonal_self_mat)

        return self_mat

    def build_one_body_energies(self):
        # print('electron energies', self.electron_energy)
        one_body_energies = np.kron((np.arange(self.dim_photons) + 0.5) * self.omega,
                                    np.ones(self.dim_electrons))
        one_body_energies += np.kron(np.ones(self.dim_photons), self.electron_energy)

        return one_body_energies

    def build_hamiltonian_mat(self, gamma):
        coupling_mat = self.build_coupling_mat(gamma, total_energy_flag=False)
        self_mat = self.build_self_mat(gamma, total_energy_flag=False)

        # uncoupled
        self.hamiltonian_mat = np.diag(self.one_body_energies)

        # coupling
        self.hamiltonian_mat += self.prefactor_coupling * coupling_mat

        # self-interaction
        self.hamiltonian_mat += self.prefactor_self * self_mat

        # multiply by occupation
        self.hamiltonian_mat *= self.occupation

    def build_energy_mat(self, gamma):
        coupling_mat = self.build_coupling_mat(gamma, total_energy_flag=True)
        self_mat = self.build_self_mat(gamma, total_energy_flag=True)

        # one-body (factor 2 from occupations)
        self.energy_mat = self.occupation * np.diag(self.one_body_energies)

        # coupling
        self.energy_mat += self.prefactor_coupling * coupling_mat

        # self-interaction
        self.energy_mat += self.prefactor_self * self_mat

    def build_coupling_energy_mat(self, gamma):
        coupling_mat = self.prefactor_coupling * self.build_coupling_mat(gamma, total_energy_flag=True)
        return coupling_mat

    def build_self_energy_mat(self, gamma):
        self_mat = self.prefactor_self * self.build_self_mat(gamma, total_energy_flag=True)
        return self_mat
