import os

NEPTUNE_PROJECT = os.environ.get("NEPTUNE_PROJECT", "default_project")
NEPTUNE_API_TOKEN = os.environ.get("NEPTUNE_API_TOKEN", "default_token")

SPECIAL_TOKENS_DICT = {
    "eos_token": "[EOS]",
    "bos_token": "[BOS]",
    "unk_token": "[UNK]",
    "sep_token": "[SEP]",
    "pad_token": "[PAD]",
    "cls_token": "[CLS]",
    "mask_token": "[MASK]",
}


DEFAULT_ARGS = {
    # GLOBAL PARAMETERS
    "task": "classification",
    "seed": 42,
    "device": "cuda",
    #
    # NEPTUNE PARAMETERS
    "project": NEPTUNE_PROJECT,
    "api_token": NEPTUNE_API_TOKEN,
    "custom_run_id": None,
    "name": None,
    "tags": [],
    #
    # TRAINING PARAMETERS
    "lr": 0.00008,
    "epochs": 6,
    "batch_size": 64,
    "criterion": "cross_entropy",
    "scheduler": False,
    "scheduler_gamma": 0.66,
    "scheduler_milestones": [3, 6, 9, 12, 15],
    #
    # DATASET PARAMETERS
    "dataset": "enwik8",
    "tokenizer": "bert-base-uncased",
    #
    # MULTIHEAD ATTENTION PARAMETERS
    "mha_type": "vanilla",
    "d_model": 512,
    "num_heads": 8,
    "dropout": 0.1,
    "max_length": 512,
    "has_outproj": True,
    "act_fun": "relu",
    "apply_rotary_pos_enc": False,
    "post_norm": True,
    #
    # MODEL PARAMETERS
    "model": "decoder_only",
    "structure": "512x6",
    "pos_enc_type": "learnable",
}
