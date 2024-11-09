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
    # NEPTUNE PARAMETERS
    "project": NEPTUNE_PROJECT,
    "api_token": NEPTUNE_API_TOKEN,
    "custom_run_id": None,
    "name": None,
    "tags": [],
    #
    # TRAINING PARAMETERS
    "init_lr": 0.00008,
    "epochs": 1,
    "batch_size": 2,
    "gradient_accumulation_steps": 8,
    "criterion": "cross_entropy",
    "use_validation": False,
    # SCHEDULER PARAMETERS
    "scheduler": True,
    "scheduler_lr_warmup_steps": int(1_000_000_000 / 4096) * 0.01,
    "scheduler_num_all_steps": int(1_000_000_000 / 4096),
    "scheduler_final_lr_fraction": 0.1,
    #
    # DATASET PARAMETERS
    "task": "sequence_modelling",
    "dataset": "enwik9",
    "tokenizer": "bert-base-uncased",
    "max_length": 4096,
    #
    # MULTIHEAD ATTENTION PARAMETERS
    "mha_type": "vanilla",
    "d_model": 512,
    "num_heads": 8,
    "dropout": 0,
    "act_fun": "relu",
    "apply_rotary_pos_enc": False,
    "post_norm": False,
    #
    # MODEL PARAMETERS
    "model": "decoder_only",
    "structure": "6x4096",
    "pos_enc_type": "learnable",
    "device": "cuda",
    #
    # HOURGLASS PARAMETERS
    "hourglass_downsampling_type": "avg",
    "hourglass_upsampling_type": "linear",
    "hourglass_attention_downsampling": True,
    "hourglass_attention_upsampling": True,
    "hourglass_upsampling_residual": True,
    "hourglass_sampling_post_norm": True,
    "hourglass_sampling_use_linear": True,
    "hourglass_sampling_use_feedforward": True,
}
