import numpy as np
from copy import deepcopy
import os

from states import States, Orbital
# from write_functions import WriteStatic


class Penalty:

    def __init__(self, write_static, states: States, method: int, penalize_first_state=False):
        # class delivers all necessary object related to a penalty method to penalyze some inequality constraint
        # implemented only augmented Lagrangian method following
        # Gonn Gould Toint, Trust Region Methods, SIAM/MPS, 2000 
        # Algorithm 14.4.1, p.595
        self.method = method  # 0 - no penalty, 1 - augmented Lagrangian method
        self.penalize_first_state = penalize_first_state

        # parameters with predefined initial values
        self.__mu = 1.5e1  # penalty parameter: pre-factor for penalty term
        self.mu_max = 1.0e9
        self.__multiplier = np.zeros(states.dim_electrons)
        multiplier_initial = 0.0
        self.__multiplier[-1] = multiplier_initial  # lam=0 calculation, 4 electrons, 4 sites, om=0.6
        if states.n_electrons == 6:
            self.__multiplier[-2] = multiplier_initial
        self.__multiplier_tolerance = (1 / self.__mu) ** 0.1
        self.states = states

        self.__constraint_function = np.zeros(states.dim_electrons)
        self.update_constraint_function()

        self.derivative = np.zeros(states.dim_electrons)
        self.update_derivative()

        # self.derivative_second = np.zeros(states.dim_electrons)
        # self.update_derivative_second()

        # penalty gradient objects for every state in states
        self.gradients = np.empty(states.dim_coupled, dtype=object)
        for ii in range(states.dim_coupled):
            self.gradients[ii] = Orbital(states.dim_coupled, ii)

        self.eigenvalues = np.zeros(states.dim_coupled)
        
        self.__iter = 0

        # restart information
        self.restart_file = ''
        self.update_restart_file(write_static)

    #
    # read/write functions
    def update_restart_file(self, write_static):
        self.restart_file = write_static.output_dir + '/penalty_restart_information'

    def write_data(self):
        data = np.array([self.__mu])
        data = np.append(data, self.__multiplier)
        print('write penalty data to: ', self.restart_file)
        np.savetxt(self.restart_file, data)

    def read_data(self):
        if os.path.exists(self.restart_file):
            print('Read penalty data from file:', self.restart_file)
            data = np.loadtxt(self.restart_file)
            self.__mu = data[0]
            self.__multiplier = data[1:]
        else:
            print('No penalty restart information found.')

    #
    # access/control of parameters and tolerances
    def get_mu(self):
        return self.__mu

    def update_mu(self, mu_update):
        self.__mu = mu_update

    def get_multiplier(self):
        return self.__multiplier

    def update_multiplier(self, multiplier_update):
        self.__multiplier = multiplier_update

    def update_multiplier_tolerance(self, multiplier_tolerance_update):
        self.__multiplier_tolerance = multiplier_tolerance_update

    def get_iter(self):
        return self.__iter

    def iter_increment(self):
        self.__iter += 1

    def update_parameters(self, minimization):
        # update lagrange multipliers or mu depending on penalty_function_norm
        self.update_constraint_function()
        constraint_function_norm = self.get_constraint_function_norm()
        print('update penalty parameters according to penalty function norm:', constraint_function_norm)

        if 0.0 < constraint_function_norm < self.__multiplier_tolerance:
            print('update penalty Lagrange multipliers:', self.__multiplier[np.where(self.__multiplier > 0)])
            # update rule, Eq. 14.4.3, Gonn Gould Toint
            self.__multiplier -= 2 * self.__mu * self.__constraint_function
            #  update rule,algorithm 14.4.2
            self.__multiplier_tolerance *= ((0.5 / self.__mu) ** 0.9)
            mu_factor = 1.5
            self.__mu *= mu_factor
            minimization.update_scf_tolerance(1 / mu_factor)
        else:
            print('update penalty parameter mu:', self.__mu)
            # update recommendation
            mu_factor = 10.0
            self.__mu *= mu_factor
            # update rule, algorithm 14.4.2
            self.__multiplier_tolerance = min(0.1, (0.5 / self.__mu)) ** 0.1
            # if minimization.get_scf_tolerance() > minimization.tolerance:
            minimization.update_scf_tolerance(1 / mu_factor)

        print('new mu:', self.__mu)
        print('new Lagrange multiplier:', self.__multiplier[np.where(self.__multiplier > 0)])
        print('new constraint tolerance:', self.__multiplier_tolerance)
        print('new scf tolerance', minimization.get_scf_tolerance())
        print('')

    #
    # objects, provided by penalty class for minimization
    def update_constraint_function(self):
        # inequality constraint function
        g_condition = self.states.get_g_condition()
        self.__constraint_function = np.zeros(self.states.dim_electrons)
        for ii in range(self.states.dim_electrons):
            if g_condition[ii] < 0:
                self.__constraint_function[ii] = g_condition[ii]

    def get_constraint_function_norm(self, update_constraint_function=True):
        if update_constraint_function:
            self.update_constraint_function()
        constraint_function_norm = np.linalg.norm(self.__constraint_function)
        return constraint_function_norm

    def barrier_function(self):
        # requires to update NON before call: states.update_reduced_quantities()
        self.update_constraint_function()
        barrier = self.__mu * sum(np.square(self.__constraint_function))

        # TODO: clarify if function or g_condition should be put here
        barrier -= np.dot(self.__multiplier, self.__constraint_function)
        # barrier -= np.dot(self.__multiplier, self.states.get_g_condition())
        return barrier

    def derivative_mu(self):
        self.update_constraint_function()
        derivative_mu = self.__mu * 2 * self.states.get_g_condition_derivative() * self.__constraint_function
        return derivative_mu

    # def derivative_second_mu(self):
    #     self.update_constraint_function()
    #     derivative_mu_second = self.__mu * 2 * ((1 - 2 * self.states.NON) ** 2 - 2 * self.__constraint_function
    #                                                   + (1 - 2 * self.states.NON) * self.__constraint_function)
    #     return derivative_mu_second

    def derivative_multiplier(self):
        derivative_multiplier = - self.__multiplier * self.states.get_g_condition_derivative()
        return derivative_multiplier

    # def derivative_multiplier_second(self):
    #     derivative_multiplier_second = - self.__multiplier * (-2 + (1 - 2 * self.states.NON))
    #     return derivative_multiplier_second

    def update_derivative(self):
        self.derivative = self.derivative_mu()
        self.derivative += self.derivative_multiplier()

    # def update_derivative_second(self):
    #     self.derivative_second = self.derivative_second_mu
    #     self.derivative_second += self.derivative_multiplier_second()

    def apply_state(self, psi: Orbital, derivative: int):
        grad_state_pen = Orbital(psi.dimension, psi.index)
        grad_state_pen.set_orbital(np.zeros(psi.dimension))
        grad_state_pen_reshape = grad_state_pen.get_orbital().reshape((self.states.dim_photons,
                                                                       self.states.dim_electrons))
        reshape_orbital = psi.get_orbital().reshape((self.states.dim_photons, self.states.dim_electrons))

        if derivative == 1:
            self.update_derivative()
            derivative_function = self.derivative
        # elif derivative == 2:
        #     self.update_derivative_second()
        #     derivative_function = self.derivative_second
        else:
            print('FATAL ERROR: only first or second derivative implemented.')
            return False

        # print('penalty_gradient psi', psi.get_orbital().round(3))
        # print('penalty_gradient derivative', derivative_function)
        # print('penalty_gradient NO[-1]', self.states.NO[:, -1])
        for k in range(self.states.dim_electrons):
            for n in range(self.states.dim_photons):
                grad_state_pen_reshape[n, :] += self.states.NO[:, k] * derivative_function[k] * \
                                                          np.dot(self.states.NO[:, k], reshape_orbital[n, :])

        grad_state_pen.set_orbital(grad_state_pen_reshape.reshape(self.states.dim_coupled))
        return grad_state_pen

    def build_matrix(self, index: int):
        matrix = np.zeros((self.states.dim_coupled, self.states.dim_coupled))
        matrix_reshape = matrix.reshape((self.states.dim_photons, self.states.dim_electrons,
                                         self.states.dim_photons, self.states.dim_electrons))
        NO_index = self.states.NO[:, index]

        for n in range(self.states.dim_photons):
            matrix_reshape[n, :, n, :] = np.outer(NO_index, NO_index)

        matrix = matrix_reshape.reshape((self.states.dim_coupled, self.states.dim_coupled))

        return matrix

    def update_gradient(self, index):
        psi = self.states.get_state(index)
        gradient = self.apply_state(psi, derivative=1)
        self.gradients[index] = gradient

    def update_eigenvalue(self, index, update_grad=False):
        if update_grad:
            self.update_gradient(index)

        gradient = self.gradients[index]
        psi = self.states.get_state(index)

        eigenvalue = psi.inner_product(gradient)
        self.eigenvalues[index] = eigenvalue

    # def alpha(self, psi, cg):
    #     psi_apply_penalty = self.apply_state(psi, derivative=2)
    #     cg_apply_penalty = self.apply_state(cg, derivative=1)
    #     alpha_penalty = psi_apply_penalty.inner_product(psi) ** 2 - cg_apply_penalty.inner_product(cg)
    #     return alpha_penalty
