from typing import Any
from models import register
from transformers import AutoTokenizer

TOKENIZER_NAME = "t0_tokenizer"

@register(TOKENIZER_NAME)
class TokenizerForT0():
    def __init__(self, cfg) -> None:
        self.name = TOKENIZER_NAME
        self.cfg = cfg
        self.tokenizer_cfg = self.cfg.tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.hypernet.model())

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.tokenizer(*args, 
                               truncation=self.tokenizer_cfg.truncation(),
                               padding=self.tokenizer_cfg.padding(),
                               max_length=self.tokenizer_cfg.max_length(),
                               return_tensors='pt'                 
                             )