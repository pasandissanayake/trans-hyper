from torch import nn
from transformers import AutoModelForSeq2SeqLM
import torch.nn.functional as F

from models import register, make


HYPERNET_NAME = "t0pp"

@register(HYPERNET_NAME)
class T0ppRegression(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.name = HYPERNET_NAME
        self.cfg = cfg
        self.hypernet_cfg = self.cfg.hypernet
        self.debug = self.cfg.debug_hypernet() or self.cfg.debug()
        
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.hypernet_cfg.name())
        self.encoder = self.model.encoder
        
        # fine-tune only a part of the model
        def check_any_substring(target_string, list_of_substrings):
            for substring in list_of_substrings:
                if substring in target_string:
                    return True
            return False
        for name, param in self.model.named_parameters():
            if check_any_substring(name, ["22, 23"]):
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.hyponet = make(model_name=self.cfg.hyponet.model(), cfg=self.cfg, sd=None)
               
        total_params = 0
        for name, shape in self.hyponet.param_shapes.items():
            total_params += shape[0] * shape[1]
        
        self.regressor = nn.Sequential(
            nn.Linear(self.model.config.d_model, total_params),
            nn.LayerNorm(normalized_shape=total_params)
        )

        if self.debug:
            print(f"Initializing hypernet {self.name}, name: {self.hypernet_cfg.name()}")
            print(f"{self.name} hypernet hidden size: {self.model.config.d_model}")
            print(f"total hyponet params: {total_params}")
            
    def forward(self, data):
        outputs = self.encoder(input_ids=data["input_ids"], attention_mask=data["attention_mask"])
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