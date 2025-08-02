import jax
import jax.numpy as jnp
from jax import jit, grad
import numpy as np

import os
import random
import pickle

out_dir = "out-shakespeare-char"
os.makedirs(out_dir, exist_ok=True)

dataset = "shakespeare_char"
data_dir = os.path.join("data", dataset)

wandb_log = True
wandb_project = "shakespeare-char"
wandb_run_name = "mini-gpt"

batch_size = 64
block_size = 256

tokens_per_iter = batch_size * block_size

n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 1e-3
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4
beta2 = 0.99

warmup_iters = 100

vocab_size = pickle.load(open(os.path.join(data_dir, "meta.pkl"), "rb"))["vocab_size"]

def get_batch(split):
    if split == "train":
        data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
    else:
        data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")
    ix = [random.randrange(len(data) - block_size) for _ in range(batch_size)]
    x = np.stack([(data[i : i + block_size]).astype(np.int64) for i in ix])
    y = np.stack([(data[i + 1 : i + 1 + block_size]).astype(np.int64) for i in ix])
    return x, y
