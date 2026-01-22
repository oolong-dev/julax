import julax.layers as layers


def test_layers_exposed():
    # base
    assert layers.LayerBase
    assert layers.LayerLike
    assert layers.to_layer

    # commons
    assert layers.Linear
    assert layers.Dropout
    assert layers.Embedding
    assert layers.RotaryEmbedding
    assert layers.LayerNorm
    assert layers.RMSNorm
    assert layers.train_mode
    assert layers.test_mode

    # connectors
    assert layers.F
    assert layers.Select
    assert layers.Repeat
    assert layers.NamedLayers
    assert layers.Chain
    assert layers.Branch
    assert layers.Residual
    assert layers.Parallel

    # einops
    assert layers.Reduce
    assert layers.Rearrange
    assert layers.EinMix

    # core
    assert layers.Learner
    assert layers.Trainer

    # pprint
    assert layers.pprint
