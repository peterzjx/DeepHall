
from flax import linen as nn
import numpy as np
import jax
from jax import numpy as jnp
from flax.training import train_state
import optax

from deephall.config import OrbitalType

from .blocks import Jastrow, Orbitals




class PfafformerLayers(nn.Module):
    num_heads: int
    heads_dim: int
    num_layers: int

    @nn.compact
    def __call__(self, electrons: jnp.ndarray):
        theta, phi = electrons[..., 0], electrons[..., 1]  # [num, 2], [num, 2]
        h_one = self.input_feature(theta, phi)
        attention_dim = self.num_heads * self.heads_dim
        h_one = nn.Dense(attention_dim, use_bias=False)(h_one)
        for _ in range(self.num_layers):
            attn_out = nn.MultiHeadAttention(num_heads=self.num_heads)(h_one)
            h_one += nn.Dense(attention_dim, use_bias=False)(attn_out)
            h_one = nn.LayerNorm(epsilon=1e-5)(h_one)
            h_one += nn.tanh(nn.Dense(attention_dim)(h_one))
            h_one = nn.LayerNorm(epsilon=1e-5)(h_one)
        return h_one

    def input_feature(self, theta: jnp.ndarray, phi: jnp.ndarray):
        return jnp.stack(
            [
                jnp.cos(theta),
                jnp.sin(theta) * jnp.cos(phi),
                jnp.sin(theta) * jnp.sin(phi),
            ],
            axis=-1,
        )


def extract_pairs(electron):
    """
    Generate unordered (ri, rj) pairs from electron coordinates.
    Returns: [Ne, Ne, D]
    D = length(features: ui, vi, ...)
    """
    theta, phi = electron[..., 0], electron[..., 1]
    Ne, _ = electron.shape
    idx_i, idx_j = jnp.meshgrid(jnp.arange(Ne), jnp.arange(Ne), indexing='ij')
    

    # Gather the corresponding (ri, rj) pairs
    # thetai = jnp.reshape(theta[idx_i], [-1])  # Shape: [Ne * Ne]
    # phii = jnp.reshape(phi[idx_i], [-1])  # Shape: [Ne * Ne]
    # thetaj = jnp.reshape(theta[idx_j], [-1])  # Shape: [Ne * Ne]
    # phij = jnp.reshape(phi[idx_j], [-1])  # Shape: [Ne * Ne]
    
    thetai = theta[idx_i]  # Shape: [Ne * Ne]
    phii = phi[idx_i] # Shape: [Ne * Ne]
    thetaj = theta[idx_j] # Shape: [Ne * Ne]
    phij = phi[idx_j] # Shape: [Ne * Ne]
    
    
    upper_i, upper_j = jnp.triu_indices(thetai.shape[0], k=1)
    upper_theta_i = thetai[upper_i, upper_j]
    upper_theta_j = thetaj[upper_i, upper_j]
    upper_phi_i = phii[upper_i, upper_j]
    upper_phi_j = phij[upper_i, upper_j]

    theta_pair = jnp.stack([upper_theta_i, upper_theta_j], axis=-1)  # [Ne*(Ne-1)/2, 2]
    phi_pair = jnp.stack([upper_phi_i, upper_phi_j], axis=-1)  # [Ne*(Ne-1)/2, 2]
    pair_feature = jnp.stack([theta_pair, phi_pair], axis=-1)  # [Ne*(Ne-1)/2, 2, 2]

    # [B, (i, j), (theta, phi)]
    
    return pair_feature, upper_i, upper_j

def pairwise_trunc(R, theta, phi, mask_len=0.1):
    theta1 = theta[0]
    theta2 = theta[1]
    phi1 = phi[0]
    phi2 = phi[1]
    u1 = jnp.cos(theta1 / 2) * jnp.exp(1j * phi1 / 2)
    u2 = jnp.cos(theta2 / 2) * jnp.exp(1j * phi2 / 2)
    v1 = jnp.sin(theta1 / 2) * jnp.exp( - 1j * phi1 / 2)
    v2 = jnp.sin(theta2 / 2) * jnp.exp( - 1j * phi2 / 2)
    
    chord = jnp.abs(u1 * v2 - u2 * v1)
    
    return 1-jnp.exp(-(chord/mask_len)**2)

@nn.compact
def original_pfaf(electron, mask_len = 0.1, truncate = False):
    """Generate unordered (ri, rj) pairs from electron coordinates."""
    theta, phi = electron[..., 0], electron[..., 1]
    u = (jnp.cos(theta / 2) * jnp.exp(0.5j * phi))[..., None]  # [..., N, 1]
    v = (jnp.sin(theta / 2) * jnp.exp(-0.5j * phi))[..., None]  # [..., N, 1]
    Ne, _ = electron.shape

    pair_ij = (u * v[:, 0] - u[:, 0] * v)
    rho_ij = jnp.abs(pair_ij)
    #original pfaffian
    pf_ij = (1 - jnp.eye(Ne)) / (u * v[:, 0] - u[:, 0] * v + jnp.eye(u.shape[0]) + 1e-8)  # [..., N, N]
    if truncate == False:
        return pf_ij
    #masked pfaffian
    else:
        mask_pf_ij = pf_ij * (1-jnp.exp(-(rho_ij/mask_len)**2))
        return mask_pf_ij

class PairOrbitals(nn.Module):
    Q: float
    ndets: int

    def setup(self):
        m = np.arange(-self.Q, self.Q + 1)
        self.norm_factor = jnp.array(np.sqrt(ss.comb(2 * self.Q, self.Q - m)))
        
        self.featured_orbitals = FeaturedOrbitals(
            nspins=self.nspins,
            features=(int(self.Q * 2) + 1, sum(self.nspins), self.ndets),
        )

    def __call__(self, h_one, theta, phi):
        orbitals = self.featured_orbitals(h_one)

        m = jnp.arange(-self.Q, self.Q + 1)
        u = (jnp.cos(theta / 2) * jnp.exp(0.5j * phi))[..., None]
        v = (jnp.sin(theta / 2) * jnp.exp(-0.5j * phi))[..., None]
        envelope = self.norm_factor * u ** (self.Q + m) * v ** (self.Q - m)
        orbitals = jnp.sum(orbitals * envelope[..., None, None], axis=1)

        return jnp.moveaxis(orbitals, -1, 0)  # Move ndets dim to the front
class Pfaffian(nn.Module):
    nspins: tuple[int, int]
    Q: float
    ndets: int
    num_heads: int
    heads_dim: int
    num_layers: int
    orbital_type: OrbitalType
    mask_len: float = 0.2
    benchmark_original: bool = False

    def setup(self):
        """Define submodules within the same Flax scope."""
        self.h_one_function = PfafformerLayers(
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            heads_dim=self.heads_dim
        )

        self.pair_orbitals = PairOrbitals(
            type=self.orbital_type, 
            Q=self.Q, 
            ndets=1
        )

    def __call__(self, electrons: jnp.ndarray):
        # Using NN for wfn
        Ne = electrons.shape[0]
        electron_pair, upper_i, upper_j = extract_pairs(electron=electrons)
        theta, phi = electron_pair[..., 0], electron_pair[..., 1]
        pair_num = Ne * (Ne-1) // 2 

        
        

        if self.benchmark_original:
            # Using original Moore-Read Pfaffian for benchmarking
            pfaf_ij = original_pfaf(electron=electrons)
            pfaffian = jnp.sqrt(jnp.linalg.det(pfaf_ij))
            
        else:
            #########################################################################
            pair_orbs = jnp.zeros(pair_num, dtype=jnp.complex64)
        
            #     h_one = PfafformerLayers(
            #         num_heads=self.num_heads,
            #         num_layers=self.num_layers,
            #         heads_dim=self.heads_dim,
            #     )(electrons)
            #     orbitals = Orbitals(
            #         type=self.orbital_type, Q=self.Q, nspins=(2, 0), ndets=self.ndets
            #     )(h_one, theta, phi)

            # h_one_function = PfafformerLayers(
            #     num_heads=self.num_heads,
            #     num_layers=self.num_layers,
            #     heads_dim=self.heads_dim)
            # h_one_param = h_one_function.init(jax.random.PRNGKey(0), jnp.zeros_like(electron_pair[0]))
            # h_one_value = h_one_function.apply({'params': h_one_param['params']},electron_pair[0])
            # pair_orbitals = Orbitals(
            #     type=self.orbital_type, 
            #     Q=self.Q, nspins=(2, 0), 
            #     ndets=1)
            # pair_orb_param = pair_orbitals.init(jax.random.PRNGKey(1), jnp.zeros_like(h_one_value),jnp.zeros_like(theta[0]),jnp.zeros_like(phi[0]))  # Dummy input to init parameters
            h_one_param = self.h_one_function.init(jax.random.PRNGKey(0), jnp.zeros_like(electron_pair[0]))
            h_one_value = self.h_one_function.apply({'params': h_one_param['params']},electron_pair[0])
            pair_orb_param = self.pair_orbitals.init(jax.random.PRNGKey(1), jnp.zeros_like(h_one_value),jnp.zeros_like(theta[0]),jnp.zeros_like(phi[0]))  # Dummy input to init parameters
#################################################################################
            h_one_value = self.h_one_function(electron_pair)
            print('h1 shape',h_one_value.shape)
            # h_one_value = self.h_one_function.apply({'params': h_one_param['params']},electron_pair[i])
            pair_value = self.pair_orbitals(h_one_value, theta, phi)
            # pair_value = self.pair_orbitals.apply({'params': pair_orb_param['params']},h_one_value, theta[i], phi[i])
            # pair_value = pair_value * pairwise_trunc(jnp.sqrt(self.Q), theta=theta, phi=phi,mask_len=self.mask_len)
            print('ele pair', electron_pair.shape)
            print('pair_value',pair_value.shape)

            dets = jnp.linalg.det(pair_value)
            print(dets.shape)
            pair_orbs = pair_orbs.at[i].set(jnp.sum(dets))
#################################################################################
            # for i in jnp.arange(0, pair_num):
            #     # h_one_value = self.h_one_function(electron_pair[i])
            #     h_one_value = self.h_one_function.apply({'params': h_one_param['params']},electron_pair[i])
            #     # pair_value = self.pair_orbitals(h_one_value, theta[i], phi[i])
            #     pair_value = self.pair_orbitals.apply({'params': pair_orb_param['params']},h_one_value, theta[i], phi[i])
            #     pair_value = pair_value * pairwise_trunc(jnp.sqrt(self.Q), theta=theta[i], phi=phi[i],mask_len=self.mask_len)
            #     dets = jnp.linalg.det(pair_value)
            #     pair_orbs = pair_orbs.at[i].set(jnp.sum(dets))
            #     print(h_one_value.shape)
            #     print(pair_value.shape)
            pfaf_ij = jnp.zeros([Ne, Ne], dtype=jnp.complex64)
            pfaf_ij = pfaf_ij.at[upper_i, upper_j].set(pair_orbs)
            pfaf_ij = pfaf_ij - pfaf_ij.T
            #########################################################################
            # orig_pfaf_ij = original_pfaf(electron=electrons)
            cusp_matrix = self.cusp_matrix(electrons, mask_len=self.mask_len)
            pfaf0 = original_pfaf(electrons, mask_len=self.mask_len, truncate=True)
            # print(pfaf_ij+cusp_matrix)
            # print(cusp_matrix)
            # print(pfaf0)
            pfaffian = jnp.sqrt(jnp.linalg.det((pfaf_ij+cusp_matrix)))
            # pfaffian = jnp.sqrt(jnp.linalg.det(cusp_matrix))
        
        
        cf_flux = self.flux_attachment(electrons, mask_len=self.mask_len, truncate=True)
        return jnp.log(pfaffian * cf_flux)
        # return jnp.log(pfaffian)
    @nn.compact
    # def pair_orbitals(self, electrons):
    #     theta, phi = electrons[..., 0], electrons[..., 1]
    #     h_one = PfafformerLayers(
    #         num_heads=self.num_heads,
    #         num_layers=self.num_layers,
    #         heads_dim=self.heads_dim,
    #     )(electrons)
    #     orbitals = Orbitals(
    #         type=self.orbital_type, Q=self.Q, nspins=(2, 0), ndets=self.ndets
    #     )(h_one, theta, phi)
        
    @nn.compact
    def get_rhoij(self, electrons):
        theta, phi = electrons[..., 0], electrons[..., 1]

        u = (jnp.cos(theta / 2) * jnp.exp(0.5j * phi))[..., None]  # [..., N, 1]
        v = (jnp.sin(theta / 2) * jnp.exp(-0.5j * phi))[..., None]  # [..., N, 1]
        
        element = u * v[:, 0] - u[:, 0] * v + jnp.eye(u.shape[0])  # [..., N, N]
        rho = jnp.abs(element)
        return rho

    @nn.compact
    def cusp_matrix(self, electrons, mask_len=0.1):
        theta, phi = electrons[..., 0], electrons[..., 1]

        u = (jnp.cos(theta / 2) * jnp.exp(0.5j * phi))[..., None]  # [..., N, 1]
        v = (jnp.sin(theta / 2) * jnp.exp(-0.5j * phi))[..., None]  # [..., N, 1]
        
        element = u * v[:, 0] - u[:, 0] * v + jnp.eye(u.shape[0])  # [..., N, N]
        rho = jnp.abs(element)

        # cusp_element = (1 - jnp.eye(Ne)) / (u * v[:, 0] - u[:, 0] * v + jnp.eye(u.shape[0]) + 1e-8)

        cusp_element =  element * jnp.exp(-(rho / mask_len)**2)
        return cusp_element

    @nn.compact
    def flux_attachment(self, electrons, mask_len=0.1, truncate=False):
        """
            electrons: [..., N, 2]
        """
        theta, phi = electrons[..., 0], electrons[..., 1]

        u = (jnp.cos(theta / 2) * jnp.exp(0.5j * phi)
             )[..., None]  # [..., N, 1]
        v = (jnp.sin(theta / 2) * jnp.exp(-0.5j * phi)
             )[..., None]  # [..., N, 1]

        # element = u * v[:, 0] - u[:, 0] * v + jnp.eye(u.shape[0])  # [..., N, N]
        element = u * v[:, 0] - u[:, 0] * v  # [..., N, N]
        
        if truncate == True:
            rho = jnp.abs(element)
            masked_element =  jnp.sqrt(mask_len**2 + rho**2) * element / (rho + jnp.eye(u.shape[0])) + jnp.eye(u.shape[0])
            windings = jnp.prod(masked_element, keepdims=False)  # [..., N, 1]
        else:
            windings = jnp.prod(element + jnp.eye(u.shape[0]), keepdims=False)  # [..., N, 1]
        
        return windings
