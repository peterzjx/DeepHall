class PfafformerLayers(nn.Module):
    num_heads: int
    heads_dim: int
    num_layers: int

    @nn.compact
    def __call__(self, electrons: jnp.ndarray):
        theta, phi = electrons[..., 0], electrons[..., 1]
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


class Pfafformer(nn.Module):
    Q: float
    ndets: int
    num_heads: int
    heads_dim: int
    num_layers: int
    orbital_type: OrbitalType

    def __call__(self, electrons):
        orbitals = self.orbitals(electrons)
        signs, logdets = jnp.linalg.slogdet(orbitals)
        logmax = jnp.max(logdets)  # logsumexp trick
        return jnp.log(jnp.sum(signs * jnp.exp(logdets - logmax))) + logmax

    @nn.compact
    def orbitals(self, electrons):
        theta, phi = electrons[..., 0], electrons[..., 1]

        h_one = PfafformerLayers(
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            heads_dim=self.heads_dim,
        )(electrons)
        orbitals = Orbitals(
            type=self.orbital_type, Q=self.Q, nspins= (2, 0), ndets=self.ndets
        )(h_one, theta, phi)
        jastrow = Jastrow((2, 0))(electrons)
        return jnp.exp(jastrow / 2) * orbitals
