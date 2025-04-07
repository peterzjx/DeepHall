
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
        for i in range(self.num_layers):
            attn_out = nn.MultiHeadAttention(num_heads=self.num_heads, name=f'mha_{i}')(h_one)
            h_one += nn.Dense(attention_dim, use_bias=False, name=f'dense1_{i}')(attn_out)
            h_one = nn.LayerNorm(epsilon=1e-5, name=f'ln1_{i}')(h_one)
            h_one += nn.tanh(nn.Dense(attention_dim, name=f'dense2_{i}')(h_one))
            h_one = nn.LayerNorm(epsilon=1e-5, name=f'ln2_{i}')(h_one)
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
    mask_len: float = 0.1
    benchmark_original: bool = False

    def setup(self):
        """Define submodules within the same Flax scope."""
        self.h_one_function = PfafformerLayers(
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            heads_dim=self.heads_dim
        )

        self.pair_orbitals = Orbitals(
            type=self.orbital_type, 
            Q=3/2, 
            nspins=(2, 0),
            ndets=1
        )

    def __call__(self, electrons: jnp.ndarray):
        Ne = electrons.shape[0]
        electron_pair, upper_i, upper_j = extract_pairs(electron=electrons)
        theta, phi = electron_pair[..., 0], electron_pair[..., 1]
        pair_num = Ne * (Ne-1) // 2 
        
        # Initialize parameters outside the loop
        h_one_value = jax.vmap(self.h_one_function)(electron_pair)
        
        pair_values = jax.vmap(self.pair_orbitals)(h_one_value, theta, phi)
        
        # Apply truncation to all pairs at once
        trunc_factors = jax.vmap(pairwise_trunc, in_axes=(None, 0, 0, None))(
            jnp.sqrt(self.Q), theta, phi, self.mask_len
        )
        pair_values = pair_values * trunc_factors[..., None, None]
        
        # Calculate determinants for all pairs
        dets = jax.vmap(jnp.linalg.det)(pair_values)
        pair_orbs = jnp.sum(dets, axis=-1)
        
        # Construct the antisymmetric matrix
        pfaf_ij = jnp.zeros([Ne, Ne], dtype=jnp.complex64)
        pfaf_ij = pfaf_ij.at[upper_i, upper_j].set(pair_orbs)
        pfaf_ij = pfaf_ij - pfaf_ij.T
        #########################################################################
        # orig_pfaf_ij = original_pfaf(electron=electrons)
        cusp_matrix = self.cusp_matrix(electrons, mask_len=self.mask_len)
        pfaffian = jnp.sqrt(jnp.linalg.det((pfaf_ij+cusp_matrix)))
        # pfaffian = jnp.sqrt(jnp.linalg.det(cusp_matrix))
        
        
        cf_flux = self.flux_attachment(electrons, mask_len=self.mask_len, truncate=True)
        return jnp.log(pfaffian * cf_flux)

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
