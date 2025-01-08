<p align="center">
  <img src=".github/img/banner.png">
</p>

Simulating the fractional quantum Hall effect (FQHE) with neural network variational Monte Carlo.

This repository contains the codebase for the paper [Taming Landau level mixing in fractional quantum Hall states with deep learning](https://arxiv.org/abs/2412.14795). If you use this code in your work, please [cite our paper](CITATIONS.bib).

Currently, DeepHall supports running simulations with spin-polarized electrons on a sphere and has been tested with 1/3 and 2/5 fillings.

## Installation

DeepHall requires Python 3.11 or higher. It is highly recommended to install DeepHall in a separate virtual environment.

```bash
# Remember to activate your virtual environment
git clone https://github.com/bytedance/DeepHall
cd DeepHall
pip install -e .                  # Install CPU version
pip install -e ".[cuda12]"        # Download CUDA libraries from PyPI
pip install -e ".[cuda12_local]"  # Or, use local CUDA libraries
```

To customize JAX installation, please refer to the [JAX documentation](https://jax.readthedocs.io/en/latest/installation.html).

## Performing Simulations

### Command Line Invocation

You can use the `deephall` command to run FQHE simulations. The configurations can be passed to DeepHall using the `key=value` syntax (see [OmegaConf](https://omegaconf.readthedocs.io/en/2.3_branch/usage.html#from-a-dot-list)). A simple example would be:

```bash
deephall 'system.nspins=[6,0]' system.flux=15 optim.iterations=100
```

In this example, we place 6 electrons on a sphere with a total flux $2Q=15$ through the spherical surface. The radius of the sphere is implicitly set as $\sqrt{Q}=\sqrt{15/2}$. This configuration corresponds to 1/3 filling. (Remember that the particle–flux relation on the sphere geometry is $2Q = N / \nu - \mathcal{S}$, where $\mathcal{S}=3$ for 1/3 filling.) The energy output includes only the kinetic part and the electron–electron interactions.

If you just want to test the installation, an even simpler example is the non-interacting case with a smaller network and batch size:

```bash
deephall 'system.nspins=[3,0]' system.flux=2 system.interaction_strength=0 optim.iterations=100 network.psiformer.num_layers=2 batch_size=100
```

Details of available settings are available at [config.py](deephall/config.py).

### Python API

You can also use DeepHall from your Python script. For example:

```python
from deephall import Config, train

config = Config()
config.system.nspins = (3, 0)
config.system.flux = 2
config.system.interaction_strength = 0.0
config.optim.iterations = 100
config.network.psiformer.num_layers = 2
config.batch_size = 100

train(config)
```

## Output

By default, the results directory is named like `DeepHall_n3l2_xxxxxx_xx:xx:xx`. You can configure the output location with the `log.save_path` config, which can be any writable path on the local machine or a remote path supported by [universal_pathlib](https://github.com/fsspec/universal_pathlib).

In the results directory, the file you will need most of the time is `train_stats.csv`, which contains the energy, angular momentum, and other useful quantities per step. The checkpoint files like `ckpt_000099.npz` store Monte Carlo walkers and neural network parameters so that the wavefunction can be analyzed, and the training can be resumed.

## Wavefunction Analysis with NetObs

DeepHall contains a `netobs_bridge` module to calculate the pair correlation function, overlap with the Laughlin wavefunction, and the one-body reduced density matrix. With [NetObs](https://github.com/bytedance/netobs) installed:

```bash
netobs deephall unused deephall@overlap --with steps=50 --net-restore save_path/ckpt_000099.npz --ckpt save_path/overlap
```

## Citing Our Paper

If you use this code in your work, please cite the following paper:

```bib
@misc{qian_taming_2024,
  title = {Taming {{Landau}} Level Mixing in Fractional Quantum {{Hall}} States with Deep Learning},
  author = {Qian, Yubing and Zhao, Tongzhou and Zhang, Jianxiao and Xiang, Tao and Li, Xiang and Chen, Ji},
  year = {2024},
  month = dec,
  number = {arXiv:2412.14795},
  eprint = {2412.14795},
  primaryclass = {cond-mat},
  publisher = {arXiv},
  doi = {10.48550/arXiv.2412.14795},
  urldate = {2024-12-23},
  archiveprefix = {arXiv}
}
```
