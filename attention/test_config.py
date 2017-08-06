model_params = {
    "optimizer": "Adam",

    "encoder_params": {
        "encoder_block_params": {
            "num_heads": 8, "hidden_size": 512, "dropout_rate": 0.5
        },
        "embed_params": {
            "vocab_size": 32, "embed_dim": 128
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
            "vocab_size": 32, "embed_dim": 128
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
    "steps": 20
}

validation_params = {
    "batch_size": 32,
    "num_epochs": 3,
    "steps": 1
}
test_params = {}


estimator_params = {
    "save_summary_steps": 20,
    "save_checkpoints_steps": 20,
}
