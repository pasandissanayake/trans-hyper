import torch
from torch import nn
from torch.nn import functional as F
import einops

from tabpfn_extensions import TabPFNClassifier
from tabpfn_extensions.embedding import TabPFNEmbedding

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
        self.extractor = TabPFNEmbedding(tabpfn_clf=self.classifier, n_fold=0)
        
        self.hyponet = make(model_name=self.cfg.hyponet.model, cfg=self.cfg, sd=None)
               
        total_params = 0
        for name, shape in self.hyponet.param_shapes.items():
            total_params += shape[0] * shape[1]
        
        self.regressor = nn.Sequential(
            nn.Linear(192 * self.cfg.datasets.n_queries, total_params),
            nn.LayerNorm(normalized_shape=total_params)
        )

        if self.debug:
            print(f"Initializing hypernet {self.name}, name: {self.hypernet_cfg.name}")
            print(f"total hyponet params: {total_params}")
            
    def forward(self, data):
        queries_x = data['queries_x']
        queries_y = data['queries_y']
        X = queries_x.squeeze(0).cpu()
        y = queries_y.squeeze(0).cpu()

        self.classifier.fit(X, y)
        embeddings = self.extractor.get_embeddings(X, y, X, data_source="test")
        embeddings = einops.rearrange(embeddings, "batch sample features -> batch (sample features)")
        outputs = torch.tensor(embeddings, dtype=torch.float32).cuda()

        # print(f"devices -- embeds: {embeddings.device}, outputs: {outputs.device}")

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