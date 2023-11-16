"""Tokenization classes for CPT."""
import json
from typing import Optional, Tuple

from tokenizers import pre_tokenizers

from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.json", "merges_file": "merges.txt", "tokenizer_file": "tokenizer.json"}

PRETRAINED_VOCAB_FILES_MAP = {
    "tokenizer_file": {
        "EleutherAI/gpt-neox-20b": "https://huggingface.co/EleutherAI/gpt-neox-20b/resolve/main/tokenizer.json",
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "gpt-neox-20b": 2048,
}


class CPTTokenizerFast(PreTrainedTokenizerFast):

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file=None,
        merges_file=None,
        tokenizer_file=None,
        unk_token="<|endoftext|>",
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        add_prefix_space=False,
        **kwargs,
    ):
        super().__init__(
            vocab_file,
            merges_file,
            tokenizer_file=tokenizer_file,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            add_prefix_space=add_prefix_space,
            **kwargs,
        )

        pre_tok_state = json.loads(self.backend_tokenizer.pre_tokenizer.__getstate__())
        if pre_tok_state.get("add_prefix_space", add_prefix_space) != add_prefix_space:
            pre_tok_class = getattr(pre_tokenizers, pre_tok_state.pop("type"))
            pre_tok_state["add_prefix_space"] = add_prefix_space
            self.backend_tokenizer.pre_tokenizer = pre_tok_class(**pre_tok_state)

        self.add_prefix_space = add_prefix_space

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        return tuple(files)

    @property
    # Copied from transformers.models.gpt2.tokenization_gpt2.GPT2Tokenizer.default_chat_template
    def default_chat_template(self):
        """
        A simple chat template that ignores role information and just concatenates messages with EOS tokens.
        """
        logger.warning_once(
            "\nNo chat template is defined for this tokenizer - using the default template "
            f"for the {self.__class__.__name__} class. If the default is not appropriate for "
            "your model, please set `tokenizer.chat_template` to an appropriate template. "
            "See https://huggingface.co/docs/transformers/main/chat_templating for more information.\n"
        )
        return "{% for message in messages %}" "{{ message.content }}{{ eos_token }}" "{% endfor %}"
