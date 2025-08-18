# Copyright 2024-2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
from chex import ArrayTree
from jax.numpy import cos, sin, tan

from deephall.config import InteractionType, System
from deephall.types import AngularMomenta, LocalEnergy, LogPsiNetwork, OtherObservables
from deephall.vvmc import velocity_utils as v_utils
from deephall.vvmc.velocity_utils import calculate_d_metric_xy

######################################################################################
def calculateVectPotential(_2Q: float, electron_xy: jnp.ndarray):
    x = electron_xy[..., 0]
    y = electron_xy[..., 1]
    tmp = 1 + x**2 + y**2
    Ax = _2Q / tmp * y
    Ay = -_2Q / tmp * x
    return jnp.stack([Ax, Ay], axis = -1)

# def createNablaPhi(self, coord):
#     batch_size = coord.size()[0]
#     coord.requires_grad_(True)
#     self.Phi = self.createPhi(coord)
#     self.D_Phi = torch.zeros(batch_size, self.Ne, DIM)
#     for iw in range(batch_size):
#         grad = torch.autograd.grad(self.Phi[iw], (coord,), retain_graph=True)
#         self.D_Phi[iw] = grad[0][iw]
#     self.D_Phi = self.D_Phi.to(DEVICE)
#     return self.D_Phi
######################################################################################

def coulomb_potential(cos12: jnp.ndarray, Q: float, r: jnp.ndarray) -> jnp.ndarray:
    """Returns the electron-electron Coulomb potential.

    Args:
        cos12: The cosine of the angle between two electrons.
            Shape (..., nelec, nelec).
        Q: Monopole strength. Unused.
        r: Sphere radius.

    Returns:
        potential energy
    """
    del Q
    r_ee = jnp.sqrt(2 - 2 * cos12)
    return jnp.sum(jnp.triu(1 / r_ee, k=1)) / r


def harmonic_potential(cos12: jnp.ndarray, Q: float) -> jnp.ndarray:
    """Returns the simple harmonic potential.

    The word "harmonic" describes the form of the Haldane pseudopotential on LLL:
        V(L) = L(L+1) / 2Q(Q+1) / sqrt(Q)
    and the corresponding real space form is:
        V(theta_12) = 1 + (Q+1) / Q * cos theta_12

    Args:
        cos12: The cosine of the angle between two electrons.
            Shape (..., nelec, nelec).
        Q: Monopole strength.

    Returns:
        potential energy
    """
    return jnp.sum(jnp.triu(1 + (Q + 1) / Q * cos12, k=1))


def make_potential(
    interaction_type: InteractionType, Q: float, r: jnp.ndarray
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Create potential energy function with a given type and geometry."""
    if interaction_type == InteractionType.coulomb:
        potential_function = partial(coulomb_potential, Q=Q, r=r)
    if interaction_type == InteractionType.harmonic:
        potential_function = partial(harmonic_potential, Q=Q)

    def potential(data: jnp.ndarray) -> jnp.ndarray:
        theta, phi = data[..., 0], data[..., 1]
        xyz_data = jnp.stack(
            [sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta)], axis=-1
        )
        assert len(xyz_data.shape) == 2  # (n_electrons, 3)
        cos12 = jnp.einsum("ia,ja->ij", xyz_data, xyz_data)
        return potential_function(cos12)

    return potential

def make_potential_xy(
    interaction_type: InteractionType, Q: float, r: jnp.ndarray
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Create potential energy function with a given type and geometry."""
    if interaction_type == InteractionType.coulomb:
        potential_function = partial(coulomb_potential, Q=Q, r=r)
    if interaction_type == InteractionType.harmonic:
        potential_function = partial(harmonic_potential, Q=Q)

    def potential(data: jnp.ndarray) -> jnp.ndarray:
        x, y = data[..., 0, None], data[..., 1, None]
        r = jnp.sqrt(x**2 + y**2)
        
        xi = x[:, None, :]
        xj = x[None, :, :]
        xij = xi-xj

        yi = y[:, None, :]
        yj = y[None, :, :]
        yij = yi-yj

        zij = jnp.concatenate([xij, yij], axis = -1)
        x_zij = jnp.concatenate([-yij, xij], axis = -1)
        rij2 = (xij**2 + yij**2) + 1e-10

        dij = jnp.sqrt(4 * Q * rij2) / jnp.sqrt( (1 + xi**2 + yi**2) * (1 + xj**2 + yj**2) )
        dij = jnp.squeeze(dij) + 1e-10
        
        mask = ~jnp.eye(dij.shape[0], dtype=bool)  # shape: (N, N)
        # print('in make_potential_xy', mask.shape, dij.shape)
        Vij = 1.0 / dij * mask
        return jnp.sum(Vij) / 2

    return potential

def make_local_kinetic_energy(f: LogPsiNetwork, Q: float, r: jnp.ndarray):
    r"""Creates a function to for the local kinetic energy, -1/2 \nabla^2 ln|f|.

    Args:
        f: Callable which evaluates the log of the magnitude of the wavefunction.
        Q: Monopole strength
        r: Sphere radius

    Returns:
        Callable which evaluates the local kinetic energy,
        -1/2f \nabla^2 f = -1/2 (\nabla^2 log|f| + (\nabla log|f|)^2).
    """

    def _lapl_over_f(
        params: ArrayTree, data: jnp.ndarray
    ) -> tuple[jnp.ndarray, AngularMomenta]:
        theta, phi = data[..., 0], data[..., 1]

        #        +----------------------------------------------------------+
        #        |           Prepare first and second detivatives           |
        #        +----------------------------------------------------------+

        grad_real = jax.grad(lambda p, x: f(p, x).real, argnums=1)(params, data)
        grad_imag = jax.grad(lambda p, x: f(p, x).imag, argnums=1)(params, data)
        grad_theta = grad_real[..., 0] + 1j * grad_imag[..., 0]
        grad_phi = grad_real[..., 1] + 1j * grad_imag[..., 1]
        # $(\nabla \log \psi) \cdot (\nabla \log \psi)$ on a sphere
        square_grad_logpsi = jnp.sum(grad_theta**2 + grad_phi**2 / sin(theta) ** 2)

        hess_real = jax.hessian(lambda p, x: f(p, x).real, argnums=1)(params, data)
        hess_imag = jax.hessian(lambda p, x: f(p, x).imag, argnums=1)(params, data)
        hess_logpsi = hess_real + 1j * hess_imag

        #        +----------------------------------------------------------+
        #        |                Calculating kinetic energy                |
        #        +----------------------------------------------------------+

        # $\nabla^2 \log \psi$ on a sphere
        grad_grad_logpsi = jnp.sum(
            grad_theta / tan(theta)
            + jnp.diagonal(hess_logpsi[:, 0, :, 0])
            + jnp.diagonal(hess_logpsi[:, 1, :, 1]) / sin(theta) ** 2
        )
        # See section 3.10.3 of "Composite Fermions"
        magnetic_contribution = jnp.sum(
            (Q / tan(theta)) ** 2 + 2j * Q * cos(theta) / sin(theta) ** 2 * grad_phi
        )
        sum_kinetic_momentum_square = (
            -grad_grad_logpsi - square_grad_logpsi + magnetic_contribution
        )
        kinetic_energy = sum_kinetic_momentum_square / 2 / r**2

        #        +----------------------------------------------------------+
        #        |        Calculating angular momentum square (L^2)         |
        #        +----------------------------------------------------------+

        i = (Ellipsis, slice(None), jnp.newaxis)  # same as [..., :, None]
        j = (Ellipsis, jnp.newaxis, slice(None))  # same as [..., None, :]
        r_hat = jnp.stack([sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta)])
        phi_hat = jnp.stack([-sin(phi), cos(phi), jnp.zeros_like(phi)])
        theta_hat_prime = jnp.stack(  # Rescaled theta_hat with 1/sin(theta)
            [cos(phi) / tan(theta), sin(phi) / tan(theta), -jnp.ones_like(theta)]
        )
        hess_theta_theta = hess_logpsi[:, 0, :, 0] + grad_theta[*i] * grad_theta[*j]
        hess_theta_phi = hess_logpsi[:, 0, :, 1] + grad_theta[*i] * grad_phi[*j]
        hess_phi_phi = hess_logpsi[:, 1, :, 1] + grad_phi[*i] * grad_phi[*j]
        # Note that theta_hat_prime alrealdy has a 1/sin factor
        magnetic_term = Q * (theta_hat_prime * cos(theta) + r_hat)
        # We first assume everything commutes, and add back extra terms at the end
        angular_momentum_square = jnp.sum(
            2 * phi_hat[*i] * theta_hat_prime[*j] * hess_theta_phi
            - phi_hat[*i] * phi_hat[*j] * hess_theta_theta
            - (theta_hat_prime[*i] * theta_hat_prime[*j] * hess_phi_phi)
            - (2j * magnetic_term[*j])
            * (phi_hat[*i] * grad_theta[*i] - theta_hat_prime[*i] * grad_phi[*i])
            + magnetic_term[*i] * magnetic_term[*j],
        ) - jnp.sum(grad_theta / tan(theta))  # Diagonal extra terms

        #        +----------------------------------------------------------+
        #        |                     Assemble outputs                     |
        #        +----------------------------------------------------------+

        other_observables = AngularMomenta(
            angular_momentum_z=jnp.sum(grad_phi).imag,  # same as (-1j * d_phi).real
            angular_momentum_z_square=-jnp.sum(hess_phi_phi).real,
            angular_momentum_square=angular_momentum_square.real,
        )
        return kinetic_energy, other_observables

    return _lapl_over_f

def make_local_kinetic_v_energy(f: LogPsiNetwork, Q: float, r: float):

    # def thetaphi_xy(electron_thetaphi: jnp.ndarray):
    #     theta = electron_thetaphi[..., 0]
    #     phi = electron_thetaphi[..., 1]
    #     x = jnp.cos(phi) / jnp.tan(theta / 2)
    #     y = jnp.sin(phi) / jnp.tan(theta / 2)
    #     electron_xy = jnp.stack([x, y], axis=-1)
    #     return electron_xy
    
    def divergence(F):
        def per_point_div(x_single):
            # x_single: [2] (a single point)
            def F_single(x):
                # wrap F to work on a single point [1, 2]
                return F(x[None, :])[0]  # get [2] vector

            jac = jax.jacfwd(F_single)(x_single)  # shape [2, 2]
            return jnp.trace(jac)  # scalar

        # Vectorize over N points
        return jax.vmap(per_point_div)
    
    def kinetic_E(params: ArrayTree, electron_xy: jnp.ndarray):
        """Compute divergence using autodiff (F_func is a JAX function)."""
        dmat = jnp.squeeze(calculate_d_metric_xy(electron_xy, 2 * Q))
        
        A = calculateVectPotential( 2 * Q, electron_xy=electron_xy)
        A2 = jnp.sum(A * A, axis = -1)
        F = f(params, electron_xy)
        FF = jnp.sum(F * F, axis = -1)
        AF = jnp.sum(A * F, axis = -1)

        
        paramed_model = lambda x: f(params, x)
        grad = divergence(paramed_model)(electron_xy)
        div_F = grad
        pdt = 0.5 * dmat * (- div_F - FF - 2 * 1j * AF + A2)
        
        ke = jnp.sum(pdt, axis = -1)

        return  ke, None
    
    
    return lambda p, ele_xy: kinetic_E(p, ele_xy)

def local_energy(f: LogPsiNetwork, system: System) -> LocalEnergy:
    """Creates the function to evaluate the local energy.

    Args:
        f: Callable which returns the sign and log of the magnitude of the
            wavefunction given the network parameters and configurations data.
        system: Config for system.

    Returns:
        Callable with signature e_l(params, key, data) which evaluates the local
        energy of the wavefunction given the parameters params, RNG state key,
        and a single MCMC configuration in data.
    """
    Q = system.flux / 2
    radius = jnp.array(system.radius or jnp.sqrt(Q))
    ke = make_local_kinetic_energy(f, Q, radius)
    pe = make_potential(system.interaction_type, Q, radius)

    def _e_l(
        params: ArrayTree, data: jnp.ndarray
    ) -> tuple[jnp.ndarray, OtherObservables]:
        """Returns the total energy.

        Args:
            params: network parameters.
            data: MCMC configuration.

        Returns:
            Local energy and other observables.
        """
        potential = pe(data) * system.interaction_strength
        kinetic, angular_momenta = ke(params, data)
        return kinetic + potential, angular_momenta | {
            "potential": potential,
            "kinetic": kinetic,
        }

    return _e_l

def local_v_energy(v_model: LogPsiNetwork, system: System) -> LocalEnergy:
    """Creates the function to evaluate the local energy.

    Args:
        f: Callable which returns the sign and log of the magnitude of the
            wavefunction given the network parameters and configurations data.
        system: Config for system.

    Returns:
        Callable with signature e_l(params, key, data) which evaluates the local
        energy of the wavefunction given the parameters params, RNG state key,
        and a single MCMC configuration in data.
    """
    Q = system.flux / 2
    radius = jnp.array(system.radius or jnp.sqrt(Q))
    ke = make_local_kinetic_v_energy(v_model, Q, radius)
    pe = make_potential_xy(system.interaction_type, Q, radius)

    def _e_l(
        params: ArrayTree, electrons_xy: jnp.ndarray
    ) -> tuple[jnp.ndarray, OtherObservables]:
        """Returns the total energy.

        Args:
            params: network parameters.
            data: MCMC configuration.

        Returns:
            Local energy and other observables.
        """
        potential = pe(electrons_xy) * system.interaction_strength
        kinetic, _ = ke(params, electrons_xy)
        return kinetic + potential, {
            "potential": potential,
            "kinetic": kinetic,
        }

    return _e_l

def weighted_local_energy(f: LogPsiNetwork, system: System) -> LocalEnergy:
    """Creates the function to evaluate the local energy.

    Args:
        f: Callable which returns the sign and log of the magnitude of the
            wavefunction given the network parameters and configurations data.
        system: Config for system.

    Returns:
        Callable with signature e_l(params, key, data) which evaluates the local
        energy of the wavefunction given the parameters params, RNG state key,
        and a single MCMC configuration in data.
    """
    Q = system.flux / 2
    radius = jnp.array(system.radius or jnp.sqrt(Q))
    ke = make_local_kinetic_energy(f, Q, radius)
    pe = make_potential(system.interaction_type, Q, radius)

    def _e_l(
        params: ArrayTree, data_and_weights: tuple[jnp.ndarray, jnp.ndarray]
    ) -> tuple[jnp.ndarray, OtherObservables]:
        """Returns the total energy.

        Args:
            params: network parameters.
            data: MCMC configuration.

        Returns:
            Local energy and other observables.
        """
        data, weights = data_and_weights
        potential = pe(data) * system.interaction_strength
        kinetic, angular_momenta = ke(params, data)
        # kinetic = kinetic * weights
        # potential = potential * weights

        # TODO: check if this is correct
        # angular_momenta = angular_momenta * weights / jnp.sum(weights)
        return kinetic + potential, angular_momenta | {
            "potential": potential,
            "kinetic": kinetic,
        }

    return _e_l
