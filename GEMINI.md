# JULAX: Just Layers over JAX

`julax` is a lightweight, modular deep learning library built on top of JAX. It emphasizes a clear separation between model configuration (immutable), parameters (learnable weights), and state (mutable buffers). It leverages `pydantic` for configuration and `plum` for multiple dispatch.

## Project Overview

### Core Architecture

*   **Layers (`LayerBase`):**
    *   Layers are defined as **immutable Pydantic models**. This ensures configuration is static and validatable.
    *   **Parameters (`Param`):** Learnable weights (e.g., matrix weights, biases). Initialized via `param()`.
    *   **State (`State`):** Mutable non-gradient state (e.g., batch norm statistics, optimizer state). Initialized via `state()`.
    *   **Forward Pass:** `forward(x, p, s) -> (y, s_new)`. Pure functions taking explicit parameters and state.

*   **Experiment Framework:**
    *   **`Experiment`:** Orchestrates the training loop, managing the `Trainer`, `Dataset`, and `CheckpointManager`.
    *   **`Trainer`:** Wraps a `Learner` and an `Optimizer` (Optax), handling the update step (`forward_and_backward`).
    *   **`Learner`:** Wraps the model and loss function, computing the loss.

### Key Technologies

*   **JAX:** Core numerical computing and transformations (JIT, Grad).
*   **Pydantic:** Model configuration and validation.
*   **Plum:** Multiple dispatch for flexible function overloading.
*   **Optax:** Optimization library.
*   **Grain:** Data loading.
*   **Orbax:** Checkpointing.
*   **Einops:** Tensor operations.

## Building and Running

The project uses `uv` for dependency management and packaging.

### Prerequisites

*   Python >= 3.12
*   `uv` (Universal Python Package Installer)

### Installation

```bash
# Install dependencies
uv sync
```

### Running Experiments

Experiments are located in the `experiments/` directory.

```bash
# Run the MNIST example
# Ensure you are in the project root
uv run experiments/01_mnist.py
```

### Running Tests

Tests are located in the `tests/` directory.

```bash
# Run all tests
uv run pytest
```

### Building the Package

```bash
uv build
```

## Development Conventions

*   **Immutability:** All layer configurations should be immutable. Do not store mutable state on `self`.
*   **Explicit State:** Pass `Param` and `State` dictionaries explicitly through the call stack.
*   **Type Hinting:** Use strict type hints. Pydantic will validate layer attributes at initialization.
*   **Dispatch:** Use `@dispatch` from `plum` to handle different input types or initialization signatures.
*   **Logging:** Use standard Python `logging`.
*   **Linting:** The project uses `ruff` (configuration in `pyproject.toml`).

## Directory Structure

*   `src/julax/`: Core library source code.
    *   `base.py`: Fundamental type definitions (`Param`, `State`) and `FrozenDict`.
    *   `layers/`: Layer implementations (`core.py`, `commons.py`, `connectors.py`).
    *   `experiment/`: Training loop and experiment management.
*   `experiments/`: Example training scripts (e.g., MNIST).
*   `tests/`: Unit and smoke tests.
*   `.github/`: CI/CD workflows.
