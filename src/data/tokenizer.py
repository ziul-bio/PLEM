from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

from tokenizers.decoders import Replace

# Vocabulary — all valid amino acid and special tokens
SEQUENCE_VOCAB = [
    "<cls>", "<pad>", "<eos>", "<unk>", "<mask>",
    "L", "A", "G", "V", "S", "E", "R", "T", "I", "D",
    "P", "K", "Q", "N", "F", "Y", "M", "H", "W", "C",
    "X", "B", "U", "Z", "O",
]


class ProteinTokenizer(PreTrainedTokenizerFast):
    """
    A simple character-level tokenizer for single protein sequences.
    Wraps each sequence with <cls> ... <eos> tokens automatically.
    """

    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        unk_token="<unk>",
        cls_token="<cls>",
        pad_token="<pad>",
        mask_token="<mask>",
        eos_token="<eos>",
        **kwargs,
    ):
        token_to_id = {tok: idx for idx, tok in enumerate(SEQUENCE_VOCAB)}

        # Character-level BPE = BPE with no merges
        bpe = BPE(token_to_id, merges=[], unk_token=unk_token)
        tokenizer = Tokenizer(bpe)

        special_tokens = [cls_token, pad_token, mask_token, eos_token]
        tokenizer.add_special_tokens(special_tokens)

        # Auto-wrap every sequence with <cls> and <eos>
        tokenizer.post_processor = TemplateProcessing(
            single="<cls> $A <eos>",
            special_tokens=[
                ("<cls>", tokenizer.token_to_id("<cls>")),
                ("<eos>", tokenizer.token_to_id("<eos>")),
            ],
        )

        # Join amino acids without spaces on decode
        tokenizer.decoder = Replace(" ", "")

        super().__init__(
            tokenizer_object=tokenizer,
            unk_token=unk_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            eos_token=eos_token,
            **kwargs,
        )

    @property
    def all_token_ids(self):
        return list(range(self.vocab_size))

    @property
    def special_token_ids(self):
        return self.all_special_ids