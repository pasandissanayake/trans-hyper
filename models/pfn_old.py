import torch
from torch import nn
from torch.nn import functional as F
from tabpfn import TabPFNClassifier

from models import register, make
from utils.common import check_any_substring


HYPERNET_NAME = "tabpfn"

@register(HYPERNET_NAME)
class TabPFNModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.name = HYPERNET_NAME
        self.cfg = cfg
        self.hypernet_cfg = self.cfg.hypernet
        self.debug = self.cfg.debug_hypernet or self.cfg.debug
        
        self.classifier = TabPFNClassifier(device="cuda", n_estimators=1)
        
        self.hyponet = make(model_name=self.cfg.hyponet.model, cfg=self.cfg, sd=None)
               
        total_params = 0
        for name, shape in self.hyponet.param_shapes.items():
            total_params += shape[0] * shape[1]
        
        self.regressor = nn.Sequential(
            nn.Linear(192, total_params),
            nn.LayerNorm(normalized_shape=total_params)
        )

        if self.debug:
            print(f"Initializing hypernet {self.name}, name: {self.hypernet_cfg.name}")
            print(f"total hyponet params: {total_params}")
            
    def forward(self, data):
        queries_x = data['queries_x']
        queries_y = data['queries_y']
        
        print(f"q_x shape: {queries_x.shape}")

        self.classifier.fit(queries_x.squeeze(0).cpu(), queries_y.squeeze(0).cpu())
        # self.classifier.fit(queries_x.squeeze(0), queries_y.squeeze(0))

        self.model = self.classifier.model_
        for param in self.model.parameters():
            param.requires_grad = False
        self.activations = {}
        def get_activation(name):
            def hook(model, input, output):
                print(f"in shape: {input[0].shape}, out shape: {output.shape}")
                self.activations[name] = output.detach()
            return hook
        layer_name = "transformer_encoder.layers.11.mlp.linear2" # shape: torch.Size([192, 768])
        # print(dict([*self.model.named_modules()]).keys())
        target_layer = dict([*self.model.named_modules()])[layer_name]
        target_layer.register_forward_hook(get_activation("activations"))

        predictions = self.classifier.predict_proba(queries_x.squeeze(0).cpu())  # (batch_size, n_queries, n_classes)
        print("activations:", self.activations.keys())
        outputs = torch.tensor(self.activations["activations"], dtype=torch.float32)
        print("outputs:", outputs.shape)
        # outputs = outputs.unsqueeze(0)
        outputs = self.regressor(outputs)
        
        outputs = outputs.squeeze(0)
        
        params = dict()
        start_idx = 0
        for name, shape in self.hyponet.param_shapes.items():
            end_idx = start_idx + shape[0] * shape[1]
            wb = F.normalize(outputs[:, start_idx:end_idx], dim=1)
            params[name] = wb
            start_idx = end_idx
        self.hyponet.set_params(params=params)
        return self.hyponet