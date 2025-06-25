from torch import nn
from transformers import BertModel
import torch.nn.functional as F

from models import register


HYPERNET_NAME = "bert"

@register(HYPERNET_NAME)
class BertRegressionModel(nn.Module):
    def __init__(self, cfg, hyponet, tokenizer):
        super().__init__()
        self.name = HYPERNET_NAME
        self.cfg = cfg
        self.hypernet_cfg = getattr(self.cfg.hypernet, self.name)
        self.debug = self.cfg.debug_hypernet() or self.cfg.debug()
        
        self.bert = BertModel.from_pretrained(self.hypernet_cfg.name())
        self.hyponet = hyponet
        self.tokenizer = tokenizer
        
        total_params = 0
        for name, shape in self.hyponet.param_shapes.items():
            total_params += shape[0] * shape[1]
        
        self.regressor = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, total_params),
        )

        if self.debug:
            print(f"Initializing hypernet {self.name}, name: {self.hypernet_cfg.name()}")
            print(f"{self.name} hypernet hidden size: {self.bert.config.hidden_size}")
            print(f"total hyponet params: {total_params}")
            
    def forward(self, data):
        encodings = self.tokenizer(data)
        outputs = self.bert(input_ids=encodings[["input_ids"]], attention_mask=encodings["attention_mask"])
        outputs = outputs.last_hidden_state[:, 0, :]  # [CLS] token output
        outputs = self.regressor(outputs)

        params = dict()
        start_idx = 0
        for name, shape in self.hyponet.param_shapes.items():
            end_idx = start_idx + shape[0] * shape[1]
            wb = F.normalize(outputs[:, start_idx:end_idx], dim=1)
            params[name] = wb
            start_idx = end_idx
        self.hyponet.set_params(params=params)
        return self.hyponet