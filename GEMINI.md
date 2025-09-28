# Gemini Code Understanding

This document provides an overview of the `julax` project, its structure, and how to work with the code. It is intended to be used as a context for Gemini to understand the project and assist with development.

## Project Overview

`julax` is a personal research and experimentation workspace for learning and implementing machine learning models using JAX. The project is structured as a collection of self-contained experiments and tutorials, each exploring different aspects of JAX and neural network development.

The codebase demonstrates a progression of complexity, starting with basic JAX concepts and moving towards more advanced models like a nanoGPT implementation. The project is not a library but a hands-on environment for trying out new ideas and techniques.

**Key Technologies:**

*   **JAX:** The core framework for numerical computation and automatic differentiation.
*   **Flax:** A neural network library for JAX.
*   **Optax:** A gradient processing and optimization library for JAX.
*   **Penzai:** A toolkit for structured data and tree manipulation in JAX.
*   **Pydantic:** Used for data validation and creating structured configuration models.
*   **TensorFlow Datasets:** For loading and preparing datasets.
*   **Plum:** For multiple dispatch in Python.
*   **jsonargparse:** For parsing command-line arguments and configuration files.

## Project Structure

The project is organized into the following main directories:

*   `01_jax_basics`: A collection of scripts for learning the fundamentals of JAX, including array manipulation, random number generation, and PyTrees.
*   `02_mnist`: A series of scripts (`v1.py` to `v5.py`) that implement a neural network for MNIST classification. These scripts show an evolution of the code, from a simple implementation to a more structured approach using `pydantic` and a custom `ModelBase` class.
*   `03_nanogpt`: A more complex project that implements a nanoGPT model. This includes data preparation scripts, a rich visualization utility, and a more sophisticated model architecture.
*   `scripts`: Contains utility scripts, configuration files (`config.yaml`), and notebooks for exploring specific libraries and concepts.

## Building and Running

This project does not have a single build process. Instead, each experiment is run as a separate Python script.

**Prerequisites:**

The project uses `uv` for dependency management. To install the required packages, run:

```bash
uv pip install -r requirements.txt
```

**Running Experiments:**

To run an experiment, execute the corresponding Python script. For example:

*   **JAX Basics:**
    ```bash
    python 01_jax_basics/main.py
    ```

*   **MNIST Classification:**
    ```bash
    python 02_mnist/v5.py
    ```

*   **nanoGPT Data Preparation:**
    ```bash
    python 03_nanogpt/data/shakespeare_char/prepare.py
    ```

*   **nanoGPT Training:**
    ```bash
    python 03_nanogpt/v1.py
    ```

## Development Conventions

The project follows a set of conventions that have evolved over time.

*   **Configuration:** The later experiments (`03_nanogpt`, `02_mnist/v5.py`) use `pydantic` models for configuration. This allows for type-safe and well-documented experiment parameters. The `scripts/config.yaml` file suggests a move towards using YAML files for configuration, parsed with `jsonargparse`.
*   **Model Definition:** A custom `ModelBase` class is used to define neural network models. This class provides a consistent way to initialize parameters and states, and to define the forward pass of the model.
*   **PyTrees:** The project makes extensive use of JAX PyTrees for managing model parameters and states. The `Param` and `State` classes are custom PyTree implementations that provide a more structured way to work with these data structures.
*   **Multiple Dispatch:** The `plum` library is used for multiple dispatch, which allows for defining different implementations of a function based on the types of its arguments. This is used in the `03_nanogpt/v1.py` script for the `summary` and `typeof` functions.
*   **Visualization:** The `03_nanogpt/v1.py` script includes a rich visualization utility built with the `rich` library. This utility can be used to display the structure and contents of PyTrees in a human-readable format.
*   **Code Style:** The code is formatted with `black`.
