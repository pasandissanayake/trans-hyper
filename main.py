import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, BertConfig
from torch.optim import AdamW
import torch.nn.functional as F

from models.hyponets import HypoMlp

from datahandles import *
from utils import *

import argparse
import os
import einops

from utils import ConfigObject

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True,
                        help='Path to the config file.')
    parser.add_argument('--name', type=str, default=None,
                        help='Experiment name. If not provided, will use the cfg filename.')
    args = parser.parse_args()

    return args


def make_cfg(args):
    cfg = Config(args.cfg)

    if args.name is None:
            exp_name = os.path.basename(args.cfg).split('.')[0]
    else:
        exp_name = args.name

    if args.tag is not None:
        exp_name += '_' + args.tag

    setattr(cfg, "env", ConfigObject())
    setattr(cfg.env, "exp_name", ConfigObject(exp_name))
    setattr(cfg.env, "total_gpus", ConfigObject(torch.cuda.device_count()))
    setattr(cfg.env, "save_dir", ConfigObject(os.path.join(args.save_root, exp_name)))
    setattr(cfg.env, "wandb_upload", ConfigObject(args.wandb_upload))
    setattr(cfg.env, "port", ConfigObject(str(29600 + args.port_offset)))
    setattr(cfg.env, "cudnn", ConfigObject(args.cudnn))
   
    return cfg

class TextRegressionDataset(Dataset):
    def __init__(self, texts, targets, tokenizer, max_length=128):
        self.texts = texts
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encodings = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        item = {key: val.squeeze(0) for key, val in encodings.items()}
        item["labels"] = torch.tensor(self.targets[idx], dtype=torch.float)
        return item
    
class FewshotTokenizedDataset(Dataset):
    def __init__(self, fewshotds, tokenizer, max_length=128):
        self.fewshotds = fewshotds
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.fewshotds)

    def __getitem__(self, idx):
        encodings = self.tokenizer(
            self.fewshotds[idx]['shots'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        item = {key: val.squeeze(0) for key, val in encodings.items()}
        item["queries_x"] = self.fewshotds[idx]["queries_x"]
        item["queries_y"] = self.fewshotds[idx]["queries_y"]
        return item


# 🧠 Model with Regression Head


# 🏋️ Training loop
def train(model, dataloader, optimizer, device):
    model.train()
    criterion = nn.CrossEntropyLoss()
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        queries_x = batch["queries_x"].to(device)
        queries_y = batch["queries_y"].to(device)

        hyponet = model(input_ids, attention_mask)
        loss = criterion(einops.rearrange(hyponet(queries_x), "batch n_queries n_class -> (batch n_queries) n_class"),
                                          einops.rearrange(queries_y, "batch n_queries -> (batch n_queries)"))
        loss.backward()
        optimizer.step()
        print(f"Loss: {loss.item():.4f}")


def main():
    args = parse_args()
    cfg = make_cfg(args)

    if cfg.debug(): print('UNIVERSAL DEBUG MODE ENABLED')

    


    fewshot_dataset = FewshotDataset(cfg, 'train', n_shots=3, n_queries=5)
    print(f"Fewshot dataset length: {len(fewshot_dataset)}") 
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    fewshot_tokenized = FewshotTokenizedDataset(fewshotds=fewshot_dataset,
                                                tokenizer=tokenizer)

    dataloader = DataLoader(fewshot_tokenized, batch_size=128, shuffle=True)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertRegressionModel().to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5)

    for epoch in range(3):
        print(f"Epoch {epoch+1}")
        train(model, dataloader, optimizer, device)


# ⚙️ Usage
if __name__ == "__main__":
    main()
    