model_params = {
    "optimizer": "Adam",
    "encoder_params": {
        "encoder_block_params": {
            "num_heads": 8, "hidden_size": 512, "dropout_rate": 0.5
        },
        "embed_params": {
            "vocab_size": 32, "embed_dim": 100
        },
        "params": {
            "dropout_rate": 0.5, "num_blocks": 3
        }
    },
    "decoder_params": {
        "decoder_block_params": {
            "num_heads": 8, "hidden_size": 512, "dropout_rate": 0.5
        },

        "embed_params": {
            "vocab_size": 32, "embed_dim": 100
        },

        "params": {
            "dropout_rate": 0.5, "num_blocks": 3, "vocab_size": 32
        }

    }
}
train_params = {
    "learning_rate": 0.001,
    "clip_gradients": 5.0,
    "batch_size": 32,
    "num_epochs": 3,
    "steps": 3
}

test_params = {

}
