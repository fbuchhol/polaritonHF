import numpy as np


class Parameters:
    def __init__(self, spacing, box_length, hop, v_charge, v_epsilon, v_dist, boundary, n_electrons, n_photons,
                 lamb, omega):
        self.spacing = spacing
        self.box_length = box_length
        self.n_sites = int(box_length / spacing)
        if not self.n_sites * self.spacing == self.box_length:
            print('FATAL error: spacing and box length are chosen such that n_sites is non-integer ')
        self.hop = hop
        self.v_charge = v_charge
        self.v_epsilon = v_epsilon
        self.v_dist = v_dist
        self.boundary = int(boundary)
        self.n_electrons = int(n_electrons)
        self.n_photons = int(n_photons)
        self.lamb = lamb
        self.omega = omega

        self.dim_photons = self.n_photons + 1
        self.dim_electrons = self.n_sites
        self.dim_coupled = int(self.dim_photons * self.dim_electrons)

        self.occupied_orbitals = int(self.n_electrons / 2)
        self.unoccupied_orbitals = self.dim_coupled - self.occupied_orbitals

        if not self.n_electrons % 2 == 0:
            print('FATAL: Odd number of electrons is not supported.')

    def v_func(self, charge, epsilon, dist):
        func = np.zeros(self.n_sites)
        center = (self.n_sites / 2 - 0.5)
        for ii in range(self.n_sites):
            func[ii] = - charge / np.sqrt(((ii - (center - dist / 2)) * self.spacing) ** 2 + epsilon ** 2)
            func[ii] -= charge / np.sqrt(((ii - (center + dist / 2)) * self.spacing) ** 2 + epsilon ** 2)
        print('local potential:', func)
        return func
