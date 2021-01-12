import numpy as np
import math

from states import Orbital, States
from hamiltonian import Hamiltonian


def minimize_functional(alpha_min, alpha_max, psi: Orbital, cg_normalized: Orbital, states: States,
                        hamiltonian: Hamiltonian, penalty, print_flag=False):
    psi_new = Orbital(psi.dimension, psi.index)
    line_min = []
    internal_precision = 10
    for ii, angle in enumerate(np.arange(alpha_min, alpha_max, (alpha_max - alpha_min) / internal_precision)):
        psi.mix_state(cg_normalized, psi_new, angle)
        states.set_state(psi_new)
        if penalty.method == 1:
            states.update_gamma_electronic()
        energy_new = psi_new.energy(hamiltonian, penalty)
        # penalty.method = False
        # energy_new_fock = psi_new.energy(hamiltonian, penalty)
        # penalty.method = True
        line_min.append((angle, energy_new))  # , energy_new_fock, energy_new - energy_new_fock

    line_min = np.array(line_min)
    if print_flag:
        print(line_min)

    error = 0
    if len(line_min) > 0:
        min_ind = np.where(line_min == np.amin(line_min[:, 1]))
        minimum = line_min[min_ind[0], :]
        angle = minimum[0][0]
    else:
        print("Fatal error in line minimization! Check input")
        error = 1
        angle = float(alpha_min)
        print("alpha_min=%1.4f (should be zero)" % alpha_min)
        print("g_condition should not be violated: %s" % np.array2string(states.get_g_condition()))

    if angle < (alpha_max - alpha_min) / internal_precision:
        # print("CG descend cannot lower functional at current precision!")
        error = 2

    return angle, error


def line_minimization(minimization, psi, cg, states: States, hamiltonian: Hamiltonian, penalty):
    alpha_min = 0
    # alpha_max = math.pi / 128
    alpha_max = math.pi / 2
    counter = 0
    while True:
        if counter == 0 and psi.index > 0:
            print_flag = False  # set True to print energy values of line_minimization (debug)
        else:
            print_flag = False
        alpha, error = minimize_functional(alpha_min, alpha_max, psi, cg, states, hamiltonian, penalty, print_flag)
        if (alpha_max - alpha_min) / 2 < minimization.get_lin_min_tolerance():
            # print("lin_min converged!")
            break

        if error == 1 or error == 2:
            alpha_max = alpha_max - (alpha_max - alpha_min) / 2
        else:
            delta = abs(alpha_max - alpha_min) / 4
            alpha_min = max(alpha - delta, 0)
            alpha_max = min(alpha + delta, math.pi / 2)
        counter += 1
    # print("alpha", alpha, "error", error)

    return alpha


# def line_minimization_fourier(psi: Orbital, gradient: Orbital, cg: Orbital, states: States, hamiltonian: Hamiltonian,
#                               penalty_class=None):
#     # calculates Lagrangian minimum by first order Fourier expansion
#     # following Payne ...
#
#     beta = 2 * gradient.inner_product(cg)  # gradient already includes penalty term if used
#
#     states.calc_eigenvalue(hamiltonian, psi.index)
#     cg_eval = cg.calc_expectation_value(hamiltonian)
#     alpha = 2 * (states.eigenvalues[psi.index] - cg_eval)
#     if penalty_class is not None:
#         alpha += penalty_class.penalty_alpha(psi, cg)
#
#     theta = 0.5 * math.atan(- beta / alpha)
#
#     if theta > 0:
#         return theta
#     else:
#         return theta + math.pi * 0.5
