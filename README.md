# julax

## 2025-09-26

- Rethink [Why Flax NNX?](https://flax.readthedocs.io/en/latest/why.html)
    - **Inspection**: doesn't work for us.
    - **Running computation**: worth rethinking whether we should support it or not.
    - **State handling**: doesn't work for us. The state is explicitly considered. A dedicated new class is created before `02_mnist/v5.py`. But it seems a dict like object is already enough.
    - **Model surgery**: unclear about the real benefit here. The param & state sync might be an issue. But the operation should be easy since model/param/state are mirrored trees.
    - **Transforms**: need revisit in the future.
- It seems [Lux.jl](https://lux.csail.mit.edu/stable/) is mainly inspired by the [linen](https://flax-linen.readthedocs.io/en/latest) style in flax. While the NNX style is more close to pytorch. And existing implementation in this repo is more close to Lux and axlearn.

## 2025-09-15

- [How to jointly tune learning rate and weight decay for AdamW](https://fabian-sp.github.io/posts/2024/02/decoupling/)

## 2025-08-21

- [DiLoCo: Distributed Low-Communication Training of Language Models](https://arxiv.org/abs/2311.08105)
- [Photon: Federated LLM Pre-Training](https://arxiv.org/abs/2411.02908)
- [Convergence of Distributed Adaptive Optimization with Local Updates](https://arxiv.org/abs/2409.13155)
- [DES-LOC: Desynced Low Communication Adaptive Optimizers for Training Foundation Models](https://arxiv.org/abs/2505.22549)

## 2025-08-20

- [kvax](https://github.com/nebius/kvax)

## 2025-08-07

- [chex](https://github.com/google-deepmind/chex)

## 2025-08-06

- [ArcticTraining](https://arctictraining.readthedocs.io/en/latest/index.html)
    - Looks like our config system is similar to this.

## 2025-08-03

- [penzai v2 background](https://penzai.readthedocs.io/en/stable/guides/v2_differences.html)
    - > Parameters and state variables becoming mutable, shareable variable objects
    - This seems to be aligned with current design. Currently a general `dict` is used. Maybe I should also introduce a dedicated class for params and states.
    - > all variable objects must have a unique label, which can either be specified manually or generated automatically.
    - Hmm, I find it difficult to search for a specific Param/State. MAYBE a unique label will do some help here?
    - > Eager parameter initialization
    - In current design, params & states are separated from models. So more close to lazy initialization?
    - > The built-in Transformer implementation also supports loading Llama, Mistral, and GPT-NeoX / Pythia models.
    - TODO: this is a good feature to have.

## 2025-07-28

- [optax](https://optax.readthedocs.io/en/latest/getting_started.html)

## 2025-07-26

- [Penzai+ Treescope: A Toolkit for Interpreting, Visualizing, and Editing Models As Data](https://github.com/google-deepmind/penzai)
- [ml_dtypes](https://github.com/jax-ml/ml_dtypes)
- [Controllable Video Generation: A Survey](https://arxiv.org/pdf/2507.16869)

## 2025-07-25

- Understand Jax Array and shard_map
- [HighPerfLLMs2024](https://github.com/rwitten/HighPerfLLMs2024)

## 2025-07-24

- [Introduction to Pathways on Cloud](https://cloud.google.com/ai-hypercomputer/docs/workloads/pathways-on-cloud/pathways-intro)

## 2025-07-20

### Investigations

- [Plum: Multiple Dispatch in Python](https://github.com/beartype/plum)
- [jsonargparse](https://github.com/omni-us/jsonargparse)
- [jaxtyping](https://github.com/patrick-kidger/jaxtyping)

- [maxtext](https://github.com/AI-Hypercomputer/maxtext)
- [flax](https://github.com/google/flax)
- [Equinox](https://docs.kidger.site/equinox/)
- [levanter](https://github.com/stanford-crfm/levanter)