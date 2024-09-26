# import os

# NEPTUNE_PROJECT = os.environ.get("NEPTUNE_PROJECT", "default_project")
# NEPTUNE_API_TOKEN = os.environ.get("NEPTUNE_API_TOKEN", "default_token")
NEPTUNE_PROJECT = "masters/masters-first-experiments"
NEPTUNE_API_TOKEN = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjMmNlZTYxMi0zZDA5LTQzZTItYmU0YS00ZDI0MDUzMDhiY2UifQ=="

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
    "batch_size": 8,
    "criterion": "cross_entropy",
    # SCHEDULER PARAMETERS
    "scheduler": True,
    "scheduler_lr_warmup_steps": 10000,
    "scheduler_num_all_steps": 1000000,
    "scheduler_final_lr_fraction": 0.1,
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
    "max_length": 4096,
    "has_outproj": True,
    "act_fun": "relu",
    "apply_rotary_pos_enc": False,
    "post_norm": False,
    #
    # MODEL PARAMETERS
    "model": "decoder_only",
    "structure": "512x6",
    "pos_enc_type": "learnable",
}
