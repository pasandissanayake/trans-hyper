from typing import Any
from models import register
from transformers import BertTokenizer

TOKENIZER_NAME = "bert_tokenizer"

@register(TOKENIZER_NAME)
class TokenizerForBert():
    def __init__(self, cfg) -> None:
        self.name = TOKENIZER_NAME
        self.cfg = cfg
        self.tokenizer_cfg = self.cfg.tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.tokenizer(*args, 
                               truncation=self.tokenizer_cfg.truncation(),
                               padding=self.tokenizer_cfg.padding(),
                               max_length=self.tokenizer_cfg.max_length(),
                               return_tensors='pt'                 
                             )