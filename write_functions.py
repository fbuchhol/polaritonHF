import numpy as np
import os
from copy import deepcopy
from functools import reduce
from numpy.lib import recfunctions

from parameters import Parameters
from hamiltonian import Hamiltonian
from states import States, Orbital
from penalty import Penalty
from minimizer import Minimization


class WriteStatic:
    def __init__(self, parameters: Parameters):
        self.parameters = parameters
        self.states = None
        self.penalty = None

        # labels the entries if a data series is calculated. Initialized always with index 0.
        self.data_series_index = 0

        # create output directory
        self.output_dir = None
        self.create_output_directory()

        # definition of output data
        self.static_dt = np.dtype([
            ('converged', np.bool),
            ('n_sites', np.int32), ('n_electrons', np.int32), ('n_photons', np.int32),
            ('hopping', np.float64), ('omega', np.float64), ('lambda', np.float64),
            ('v_charge', np.float64), ('v_epsilon', np.float64), ('v_dist', np.float64),
            ('potential', np.float64, (self.parameters.dim_electrons,)),
            ('energy_eigenvalues', np.float64, (self.parameters.dim_coupled,)),
            ('NON', np.float64, (self.parameters.dim_electrons,)),
            ('NO', np.float64, (self.parameters.dim_electrons, self.parameters.dim_electrons)),
            ('gamma_elec_rs', np.float64, (self.parameters.dim_electrons,
                                           self.parameters.dim_electrons)),
            ('density', np.float64, (self.parameters.dim_electrons,)),
            ('residue', np.float64, (self.parameters.dim_coupled,)),
            ('mu', np.float64),
            ('lagrange_multiplier', np.float64, (self.parameters.dim_electrons,)),
            ('pauli_eigenvalues', np.float64, (self.parameters.dim_coupled,)),
            ('E_total', np.float64),
            ('E_phys', np.float64),
            ('E_electronic', np.float64),
            ('E_photonic', np.float64),
            ('E_coupling', np.float64),
            ('E_self', np.float64),
            ('photon_number', np.float64),
            ('mode_occupation', np.float64),
            ('dipole_me', np.float64, (self.parameters.dim_coupled, self.parameters.dim_coupled))
        ])
        self.data_static = np.empty(1, dtype=self.static_dt)
        self.read_parameters()

    def new_data_set(self, parameters_new: Parameters):
        self.data_series_index += 1
        self.data_static = np.append(self.data_static, np.empty(1, dtype=self.static_dt))
        self.parameters = parameters_new
        self.read_parameters()
        self.create_output_directory()
        if self.states is not None:
            self.states.update_restart_file(self)
        if self.penalty is not None:
            self.penalty.update_restart_file(self)

    def create_output_directory(self):
        directory = 'output_t_%1.2f_v_%1.2f_%2.2f_%1.2f_om_%1.2f_nsites_%s_n_phot_%s_nelec_%s_lam_%1.2f' % \
                    (self.parameters.hop, self.parameters.v_charge, self.parameters.v_epsilon, self.parameters.v_dist,
                     self.parameters.omega, self.parameters.n_sites, self.parameters.n_photons,
                     self.parameters.n_electrons, self.parameters.lamb)

        if os.path.isdir(directory):
            print('Warning: output directory %s exists. Static output will be overwritten.' % directory)
        else:
            os.mkdir(directory)
            print('creating output directory:', directory)
        self.output_dir = directory

    def set_states(self, states: States):
        self.states = states

    def set_penalty(self, penalty: Penalty):
        self.penalty = penalty

    def read_parameters(self):
        self.data_static[self.data_series_index]['n_sites'] = self.parameters.n_sites
        self.data_static[self.data_series_index]['n_electrons'] = self.parameters.n_electrons
        self.data_static[self.data_series_index]['n_photons'] = self.parameters.n_photons
        self.data_static[self.data_series_index]['hopping'] = self.parameters.hop
        self.data_static[self.data_series_index]['omega'] = self.parameters.omega
        self.data_static[self.data_series_index]['lambda'] = self.parameters.lamb
        self.data_static[self.data_series_index]['v_charge'] = self.parameters.v_charge
        self.data_static[self.data_series_index]['v_epsilon'] = self.parameters.v_epsilon
        self.data_static[self.data_series_index]['v_dist'] = self.parameters.v_dist
        self.data_static[self.data_series_index]['potential'] = self.parameters.v_func(self.parameters.v_charge,
                                                                                       self.parameters.v_epsilon,
                                                                                       self.parameters.v_dist)

    def read_states(self):
        self.data_static[self.data_series_index]['energy_eigenvalues'] = self.states.get_eigenvalues()
        self.data_static[self.data_series_index]['NON'] = self.states.NON
        self.data_static[self.data_series_index]['NO'] = self.states.NO
        self.data_static[self.data_series_index]['residue'] = self.states.get_residues()

    def read_penalty(self):
        self.data_static[self.data_series_index]['mu'] = self.penalty.get_mu()
        self.data_static[self.data_series_index]['lagrange_multiplier'] = self.penalty.get_multiplier()
        self.data_static[self.data_series_index]['pauli_eigenvalues'] = self.penalty.eigenvalues

    def read_status(self, minimization):
        self.data_static[self.data_series_index]['converged'] = minimization.check_converged()

    def calc_gs_quantities(self, hamiltonian, print_flag=False):
        total_energy = self.states.energy_total(hamiltonian, self.penalty, add_penalty=False)
        self.data_static[self.data_series_index]['E_total'] = total_energy

        physical_energy = total_energy - (self.states.n_electrons - 1) / 2 * self.states.omega
        self.data_static[self.data_series_index]['E_phys'] = physical_energy

        electronic_energy = self.states.energy_electronic(hamiltonian)
        self.data_static[self.data_series_index]['E_electronic'] = electronic_energy

        photon_energy = self.states.energy_photon(hamiltonian)
        self.data_static[self.data_series_index]['E_photonic'] = photon_energy

        mode_occupation = photon_energy / self.states.omega
        self.data_static[self.data_series_index]['mode_occupation'] = mode_occupation

        coupling_energy = self.states.energy_coupling(hamiltonian)
        self.data_static[self.data_series_index]['E_coupling'] = coupling_energy

        self_energy = self.states.energy_self(hamiltonian)
        self.data_static[self.data_series_index]['E_self'] = self_energy

        photon_number = (photon_energy + coupling_energy + self_energy) / self.states.omega
        self.data_static[self.data_series_index]['photon_number'] = photon_number

        one_rdm_real_space = reduce(np.dot, (hamiltonian.electron_trafo, self.states.gamma_electronic,
                                             hamiltonian.electron_trafo.T))
        self.data_static[self.data_series_index]['gamma_elec_rs'] = one_rdm_real_space
        self.data_static[self.data_series_index]['density'] = np.diag(one_rdm_real_space)

        # dipole matrix elements
        ones_mat = np.eye(self.states.dim_photons)
        elec_momentum = np.kron(ones_mat, hamiltonian.dipole_mat)

        eig_vec = self.states.states_to_matrix()
        print(eig_vec)
        dipole_me = reduce(np.dot, (eig_vec.T, elec_momentum, eig_vec))
        self.data_static[self.data_series_index]['dipole_me'] = dipole_me

        if print_flag:
            print('groundstate properties:')
            print('total energy: \t\t', total_energy.round(4))
            print('physical energy: \t', physical_energy.round(4))
            print('electronic energy: \t', electronic_energy.round(4))
            print("photon energy: \t\t", photon_energy.round(4))
            print("photon number: \t\t", photon_number.round(4))
            print("coupling energy: \t", coupling_energy.round(4))
            print("self energy: \t\t", self_energy.round(4))
            print('')
            print('electronic density')
            print(np.diag(one_rdm_real_space.round(3)))
            print('')
            print('electronic 1RDM in real space:')
            print(one_rdm_real_space.round(3))
            print('')
            print('natural occupation numbers:')
            print(self.states.NON.round(4))
            print('highest occupied NOs:')
            for ii in range(self.states.dim_electrons):
                orbital = self.states.NO[:, ii]
                orbital_rs = reduce(np.dot, (hamiltonian.electron_trafo, orbital))
                print(self.states.NON[ii].round(4), orbital.round(2), orbital_rs.round(2))
            # print(self.states.NON[-1], self.states.NO[:, -1].round(2))
            # print(self.states.NON[-2], self.states.NO[:, -2].round(2))
            print('')
            print('electronic 1RDM in standard basis:')
            print(self.states.gamma_electronic.round(3))
            print('')

    def write_data(self, text_file=True):
        data_file_binary = self.output_dir + '/static.npy'
        # self.data_static.tofile(data_file_binary)
        np.save(data_file_binary, self.data_static)

        if text_file:
            data_file = self.output_dir + '/static_information'
            print('Write static information to ', data_file)
            with open(data_file, 'w') as output_file:
                for data_field_names in self.static_dt.names:
                    data = self.data_static[self.data_series_index][data_field_names]
                    if not type(data) is np.ndarray:
                        output_file.write('%s: \t %s \n' % (data_field_names, np.array2string(data)))
                    else:
                        output_file.write('%s: \n' % data_field_names)
                        output_file.write('%s \n' % np.array2string(data))

    def write_data_series(self, data_file, quantities, text_file=True):
        data = deepcopy(self.data_static[quantities])
        data = recfunctions.repack_fields(data, align=True)
        print('\t'.join(data.dtype.names))
        print(data)

        data_file_binary = data_file + '.npy'
        np.save(data_file_binary, data)

        if text_file:
            data_file_text = data_file + '.log'
            print('Write series output to ', data_file_text)
            with open(data_file_text, 'w') as output_file:
                output_file.write('\t'.join(data.dtype.names) + '\n')
                for data_line in range(data.shape[0]):
                    output_file.write('\t'.join(map(str, data[data_line])) + '\n')


def print_results(minimization, states, hamiltonian: Hamiltonian, penalty):
    print('')
    print('###################################################')
    print('Solution')
    print('###################################################')
    print('')

    print('Parameters')
    print('hopping', states.hop)
    print('lambda', states.lamb)
    print('omega', states.omega)
    if hamiltonian.v_charge > 0:
        print('soft Coulomb potential applied: e=%1.2f, eps=%1.2f' % (hamiltonian.v_charge, hamiltonian.v_epsilon))
    print('')

    if not minimization.unocc:
        print_orbitals = states.occupied_orbitals
    else:
        print_orbitals = states.dim_coupled

    # check x,p transformation invariance
    trafo_elec = np.eye(states.dim_electrons)[::-1]
    print('elec trafo in real space')
    print(trafo_elec.round(2))
    # trafo_elec = reduce(np.dot, (hamiltonian.electron_trafo.T, trafo_elec, hamiltonian.electron_trafo))
    # print('elec trafo in energy space')
    # print(trafo_elec.round(2))

    phot_array = np.ones(states.dim_photons)
    # phot_array[1::2] = -1
    trafo_phot = np.diag(phot_array)
    # print('photon trafo in number state space')
    # print(trafo_phot)
    trafo_pol = np.kron(trafo_phot, trafo_elec)
    # # print(trafo_pol.round(2))
    #
    # check_invariance = np.dot(hamiltonian.hamiltonian_mat, trafo_pol) - np.dot(trafo_pol, hamiltonian.hamiltonian_mat)
    # print(check_invariance[np.where(check_invariance > 1e-4)])

    print('Occupied space')
    print('Orbitals')
    for ii in range(print_orbitals):
        print('orbital', ii)
        print(states.get_state(ii).get_orbital().round(3))
        orbital = states.get_state(ii).get_orbital().round(3)
        orbital = orbital.reshape((states.dim_photons, states.dim_electrons))
        orbital_rs = deepcopy(orbital)
        for nn in range(states.dim_photons):
            orbital_rs[nn, :] = reduce(np.dot, (hamiltonian.electron_trafo, orbital[nn, :]))
        print(orbital_rs.round(2))
        # orbital_rs_trafo = np.dot(trafo_pol, orbital_rs.reshape(states.dim_coupled))
        # print(orbital_rs_trafo.round(2))
    print('')

    print('Eigenvalues')
    print(states.get_eigenvalues()[:print_orbitals])
    print('')

    if penalty.method == 1:
        print('Eigenvalues + penalty')
        eigenvalues_penalty = states.get_eigenvalues() + penalty.eigenvalues
        print(eigenvalues_penalty[:print_orbitals])
        print('')

        print('Non-zero Lagrange multiplier')
        pen_multiplier = penalty.get_multiplier()
        print(pen_multiplier[np.where(pen_multiplier > 0.0)])
        print('')

    print('H * psi - e * psi')
    for ii in range(print_orbitals):
        psi = states.get_state(ii)
        psi_test = Orbital(psi.dimension, psi.index)
        psi.hamiltonian_apply(hamiltonian, psi_test)
        orb_test = psi_test.get_orbital() - psi.get_eigenvalue() * psi.get_orbital()
        print(ii, orb_test.round(4))
    print('')

    if penalty.method == 1:
        print('H * psi - (e + lm) * psi')
        if penalty.penalize_first_state:
            start_print = 0
        else:
            start_print = 1
        for ii in range(start_print, print_orbitals):
            psi = states.get_state(ii)
            h_psi = Orbital(psi.dimension, psi.index)
            psi.hamiltonian_apply(hamiltonian, h_psi)
            grad_orbital = h_psi.get_orbital() + penalty.gradients[ii].get_orbital()
            eig_val = psi.get_eigenvalue() + penalty.eigenvalues[ii]
            orb_test = grad_orbital - eig_val * psi.get_orbital()
            psi.calc_residue(hamiltonian, penalty)
            print(ii, orb_test.round(4), psi.get_residue())
        print('')

    # print('reordering')
    # for ii in range(1, states.dim_coupled):
    #     states.get_state(1).set_orbital(states.get_state(ii).get_orbital())
    #     states.update_reduced_quantities()
    #     hamiltonian.build_energy_mat(states.gamma)
    #     hamiltonian.build_hamiltonian_mat(states.gamma)
    #     states.calc_eigenvalues_all(hamiltonian)
    #     print(ii, states.get_state(ii).get_orbital().round(3), 'new NON', states.NON.round(3),
    #           'energy', states.energy_total(hamiltonian))
