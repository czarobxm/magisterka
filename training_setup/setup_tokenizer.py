import argparse

from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer


from training_setup.config import SPECIAL_TOKENS_DICT


def setup_tokenizer(args: argparse.Namespace) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, resume_download=None)
    if args.tokenizer in ["gpt2", "bert-base-uncased"]:
        tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT)
        tokenizer._tokenizer.post_processor = (  # pylint: disable=protected-access
            TemplateProcessing(
                single=f"{tokenizer.bos_token} $A {tokenizer.eos_token}",
                special_tokens=[
                    (tokenizer.eos_token, tokenizer.eos_token_id),
                    (tokenizer.bos_token, tokenizer.bos_token_id),
                ],
            )
        )
    return tokenizer
