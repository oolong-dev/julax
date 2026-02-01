from jsonargparse import auto_cli

import os
from safetensors import safe_open
import jax.numpy as jnp

from model import create_model

from transformers import AutoModelForCausalLM, AutoTokenizer


def from_hf(p, s, model_path):
    tensors = {}
    with safe_open(model_path, framework="flax", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)

    w_ln1 = []
    w_q = []
    w_k = []
    w_v = []
    w_o = []
    w_ln2 = []
    w_up = []
    w_gate = []
    w_down = []

    for i in range(16):
        w_ln1.append(tensors[f"model.layers.{i}.input_layernorm.weight"])
        w_q.append(tensors[f"model.layers.{i}.self_attn.q_proj.weight"].T)
        w_k.append(tensors[f"model.layers.{i}.self_attn.k_proj.weight"].T)
        w_v.append(tensors[f"model.layers.{i}.self_attn.v_proj.weight"].T)
        w_o.append(tensors[f"model.layers.{i}.self_attn.o_proj.weight"].T)

        w_ln2.append(tensors[f"model.layers.{i}.post_attention_layernorm.weight"])
        w_up.append(tensors[f"model.layers.{i}.mlp.up_proj.weight"].T)
        w_gate.append(tensors[f"model.layers.{i}.mlp.gate_proj.weight"].T)
        w_down.append(tensors[f"model.layers.{i}.mlp.down_proj.weight"].T)

    p["blocks"]["hidden"]["attn"]["processor"]["#0"]["hidden"]["norm"]["scale"] = (
        jnp.stack(w_ln1, axis=0)
    )
    p["blocks"]["hidden"]["attn"]["processor"]["#0"]["hidden"]["qkv_proj"]["q"]["#0"][
        "w"
    ] = jnp.stack(w_q, axis=0)
    p["blocks"]["hidden"]["attn"]["processor"]["#0"]["hidden"]["qkv_proj"]["k"]["#0"][
        "w"
    ] = jnp.stack(w_k, axis=0)
    p["blocks"]["hidden"]["attn"]["processor"]["#0"]["hidden"]["qkv_proj"]["v"]["#0"][
        "w"
    ] = jnp.stack(w_v, axis=0)
    p["blocks"]["hidden"]["attn"]["processor"]["#2"]["w"] = jnp.stack(w_o, axis=0)

    p["blocks"]["hidden"]["ffn"]["processor"]["norm"]["scale"] = jnp.stack(
        w_ln2, axis=0
    )
    p["blocks"]["hidden"]["ffn"]["processor"]["up"]["up_proj"]["w"] = jnp.stack(
        w_up, axis=0
    )
    p["blocks"]["hidden"]["ffn"]["processor"]["up"]["gate_proj"]["proj"]["w"] = (
        jnp.stack(w_gate, axis=0)
    )
    p["blocks"]["hidden"]["ffn"]["processor"]["down"]["w"] = jnp.stack(w_down, axis=0)

    p["emb"]["w"] = tensors["model.embed_tokens.weight"]
    p["out_norm"]["scale"] = tensors["model.norm.weight"]

    return p, s


def main(
    prompt: str = "There's a place where time stands still.",
    max_seq_len=30,
    model_path="models/Llama-3.2-1B-Instruct/",
):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained("models/Llama-3.2-1B-Instruct/")

    tokens = tokenizer.encode(prompt)
    input_ids = jnp.array([tokens])
    m = create_model(is_training=False, cache_size=max_seq_len, batch_size=1)
    p, s = from_hf(*m.init(), model_path=os.path.join(model_path, "model.safetensors"))
    o, s_cached = m(
        {
            "token_ids": input_ids,
            "position": jnp.arange(input_ids.shape[1]).reshape(1, -1),
        },
        p,
        s,
    )

    for _ in range(max_seq_len - input_ids.shape[1]):
        new_token = int(o[0][-1].argmax())
        position = jnp.array([[len(tokens)]])
        tokens.append(new_token)

        input_ids = jnp.array([[new_token]])

        o, s_cached = m({"token_ids": input_ids, "position": position}, p, s_cached)
    response = tokenizer.decode(tokens)
    print("Julax generated response:\n", response)

    model_inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **model_inputs, max_length=max_seq_len, do_sample=False
    )
    print("HuggingFace generated response:\n", tokenizer.decode(generated_ids[0]))


if __name__ == "__main__":
    auto_cli(main)
