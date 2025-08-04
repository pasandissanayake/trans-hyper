from torch.utils.data import Dataset, DataLoader
import numpy as np
from datahandles.dataset_utils import datahandles
from tabllm import load_and_preprocess_dataset, balance_dataset
from datasets import load_from_disk
from pathlib import Path
from datahandles import CombinedDataset, CombinedTextDataset, FewshotDataset
from utils import Config

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Union

import yaml
from jinja2 import Environment, FileSystemLoader, select_autoescape
import os

TEXT_COL_NAME = "note"
TARGET_COL_NAME = "label"


class Template():
    def __init__(self, cfg, dataset_name):
        self.cfg = cfg
        self.dataset_name = dataset_name
        self.template_file_path = Path(self.cfg.datasets.tabllm.template_dir()) / f"templates_{dataset_name}.yaml"

        class CustomTagLoader(yaml.SafeLoader):
            pass

        # Add constructors for the specific tags you want to treat as regular mappings.
        # This tells PyYAML to parse the content under these tags as a standard dictionary.
        CustomTagLoader.add_constructor(u'!Template', CustomTagLoader.construct_mapping)
        CustomTagLoader.add_constructor(u'!TemplateMetadata', CustomTagLoader.construct_mapping)

        with open(self.template_file_path, 'r') as f:
                template_data = yaml.load(f, Loader=CustomTagLoader)

        template_id = list(template_data['templates'].keys())[0] # Get the first (and likely only) template ID
        template_config = template_data['templates'][template_id]

        full_jinja_template_str = template_config['jinja']
        answer_choices_raw = template_config['answer_choices']

        # Split answer choices into a list
        self.answer_choices = [choice.strip() for choice in answer_choices_raw.split('|||')]

        # Split the full Jinja template string into prompt and answer parts
        # This assumes '|||' is a delimiter for the template structure, not literal output.
        jinja_parts = full_jinja_template_str.split('|||', 1) # Split only once

        self.prompt_template_str = jinja_parts[0].strip()
        self.answer_template_str = jinja_parts[1].strip() if len(jinja_parts) > 1 else ''
        
        # Set up Jinja2 environment (no specific loader needed as we have the string directly)
        self.env = Environment(
            loader=FileSystemLoader(os.path.dirname(self.cfg.datasets.tabllm.template_dir()) or './'), # Use FileSystemLoader for relative includes, though not strictly needed here
            autoescape=select_autoescape(['html', 'xml'])
        )

        # Render the prompt part
        self.prompt_template = self.env.from_string(self.prompt_template_str)

    def apply_template(self, note:str, label:int):
        self.rendered_prompt = self.prompt_template.render(note=note) # Only note is relevant for prompt

        # Render the answer part (if it exists)
        rendered_answer = ""
        if self.answer_template_str:
            answer_template = self.env.from_string(self.answer_template_str)
            rendered_answer = answer_template.render(label=label, answer_choices=self.answer_choices)
        else:
            # If no explicit answer template, just use the selected answer choice directly
            if 0 <= label < len(self.answer_choices):
                rendered_answer = self.answer_choices[label]
            else:
                raise IndexError(f"Label {label} is out of bounds for answer choices {self.answer_choices}.")

        # Combine the rendered parts with '|||' as the explicit separator in the final output
        # This ensures '|||' is always present as a separator if the template implies it.
        if rendered_answer:
            rendered_output = f"{self.rendered_prompt}" # Include only the prompt for k-shot fine-tuning
        else:
            rendered_output = self.rendered_prompt # If there's no answer part, just return the prompt

        return rendered_output


class TabLLMDataObject():
    def __init__(self, cfg, set_hyponet_in_dim:bool):
        self.splits = ['train', 'val', 'test']

        self.cfg = cfg
        self.debug = cfg.debug_datasets() or cfg.debug()
        self.raw_data_path = self.cfg.datasets.tabllm.raw_data_path
        self.txt_data_path = self.cfg.datasets.tabllm.txt_data_path
        
        self.n_shots = self.cfg.datasets.n_shots()
                
        self.ds_list_dict = {}
        self.ds_list_dict['train'] = self.cfg.datasets.list_combine_train()
        self.ds_list_dict['val'] = self.cfg.datasets.list_combine_val()
        self.ds_list_dict['test'] = self.cfg.datasets.list_combine_test()
        
        tabllm_ds_list = ['income', 'car', 'heart', 'diabetes', 'creditg', 'blood', 'bank', 'jungle', 'wine', 'calhousing']
        for split in self.splits:
            if not set(self.ds_list_dict[split]).issubset(tabllm_ds_list):
                raise ValueError(f"Invalid dataset list in {split} split: {self.ds_list_dict[split]}. Available datasets: {tabllm_ds_list}")
        
        self.all_ds_list = list(set(self.ds_list_dict['train'] + self.ds_list_dict['val'] + self.ds_list_dict['test']))
        self.raw_datapoints = [load_and_preprocess_dataset(dataset_name=ds_name, data_dir=Path(f"{self.raw_data_path}/{ds_name}")) for ds_name in self.all_ds_list]
        self.txt_datapoints = [ pd.DataFrame(load_from_disk(f"{self.txt_data_path}/{ds_name}")) for ds_name in self.all_ds_list] # type: ignore

        self.n_features = [ds.shape[1]-1 for ds in self.raw_datapoints] # subtract 1 for the label
        self.max_n_features = max(self.n_features)

        if self.debug:
            print((f"TabLLMDataObject initialized with {len(self.ds_list_dict['train'])} training set(s), "
                  f"{len(self.ds_list_dict['val'])} validation set(s), and {len(self.ds_list_dict['test'])} test set(s).\n"
                  f"Maximum number of features across all datasets: {self.max_n_features}"))
            
        if set_hyponet_in_dim:
            self.cfg.hyponet.in_dim(self.max_n_features) # type: ignore
            if self.debug:
                print(f"Hyponet in_dim set to max number of features (={self.max_n_features})")
                
        # create splits -- structure is as follows:
        # self.split_datapoints: {ds_name: {
        #                               'data': {'train': dataframe, 'valid': dataframe, 'test': dataframe},  
        #                               'template': text template
        #                                  }
        #                         }
        self.split_datapoints = {ds_name: {'data': self.split_and_concat_dfs(raw_dps, txt_dps, 
                                                           n_shots=self.n_shots,
                                                           seed=np.random.randint(low=0, high=100)) , 'template': Template(self.cfg, ds_name)}
                                                           for ds_name, raw_dps, txt_dps in zip(self.all_ds_list, self.raw_datapoints, self.txt_datapoints)}
        

        self.data = {}
        for split in self.splits:
            self.data[split] = CombinedTabLLMTextDataset(cfg=cfg, 
                                                         split=split, 
                                                         datapoints=list(self.split_datapoints.values()), 
                                                         max_n_features=self.max_n_features)
        
    def split_and_concat_dfs(
            self,
            df_features: pd.DataFrame,
            df_notes: pd.DataFrame,
            n_shots: int,
            shuffle: bool = True,
            seed: int = 42,
            note_column_name: str = TEXT_COL_NAME
        ) -> dict[str, pd.DataFrame]:
        """
        Splits two aligned DataFrames into train/val/test splits and concatenates the `note` column
        from df_notes into df_features.

        Returns:
            train_df, val_df, test_df — each containing features + a 'note' column
        """
        assert df_features.shape[0] == df_notes.shape[0], f"DataFrames must be the same length. Got {df_features.shape[0]} and {df_notes.shape[0]}"
        assert note_column_name in df_notes.columns, f"'{note_column_name}' column must exist in df_notes"
        assert n_shots >= 0, "Ratios must sum to 1.0"

        n = len(df_features)
        indices = np.arange(n)

        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(indices)

        valid_end = n_shots + int(0.5 * (n - n_shots))
        
        train_idx = indices[:n_shots]
        val_idx = indices[n_shots:valid_end]
        test_idx = indices[valid_end:]

        def concat(df_feat, df_note, idx):
            df_combined = df_feat.iloc[idx].copy().reset_index(drop=True)
            df_combined[note_column_name] = df_note.iloc[idx][note_column_name].reset_index(drop=True)
            return df_combined

        return {
            'train': concat(df_features, df_notes, train_idx),
            'val': concat(df_features, df_notes, val_idx),
            'test': concat(df_features, df_notes, test_idx)
        }


class CombinedTabLLMTextDataset(CombinedTextDataset):
    def __init__(self, cfg, split: str, datapoints:list[dict[str, Union[dict[str, pd.DataFrame],Template]]], max_n_features:int):
        self.cfg = cfg
        self.split = split
        self.debug = cfg.debug_datasets() or cfg.debug()

        # structure of self.datapoints is as follows:
        # self.datapoints = [   ---> a list of dictionaries, one for each dataset
        #     {
        #         'data': {'train': df. 'valid': df, 'test': df}, 
        #         'template': text template
        #     }
        # ]
        self.datapoints = datapoints

        self.balanced = getattr(self.cfg.datasets.balanced, self.split)
        self.balanced = self.balanced()
        
        if self.balanced:
            for ds in self.datapoints:
                ds['data'][self.split] = balance_dataset(ds['data'][self.split], label_column='label') # type: ignore
            if self.debug:
                print(f"Balancing {self.split} split")

        self.lengths = [len(ds['data'][self.split]) for ds in self.datapoints] # type: ignore
        self.total_length = sum(self.lengths)
        if self.debug:
            print(f"split: {self.split}, counts: {[ds['data'][self.split]['label'].value_counts() for ds in self.datapoints]}") # type: ignore

        self.n_features = [ds['data'][self.split].shape[1]-2 for ds in self.datapoints] # type: ignore # subtract 2 for note and label
        if max_n_features < 1:
            self.max_n_features = max(self.n_features)
        else:
            self.max_n_features = max_n_features
            assert np.all(np.array(self.n_features) <= self.max_n_features), f"specified max number of features (={max_n_features}) is less than the available number of features (={self.n_features})"

    def __len__(self):
        return self.total_length
    
    def __getitem__(self, index):
        if isinstance(index, tuple):
            idx = index[0]
            get_text = index[1]
        else:
            idx = index
            get_text = False

        i, ds_idx = self._get_dataset_idx(idx)
        row = self.datapoints[i]['data'][self.split].iloc[ds_idx] # type: ignore
        template = self.datapoints[i]['template']

        txt_x = row[TEXT_COL_NAME]
        new_row = row.drop(TEXT_COL_NAME)
        raw_y = np.int64(new_row[TARGET_COL_NAME])
        new_row = new_row.drop(TARGET_COL_NAME)
        raw_x = np.zeros(self.max_n_features, dtype=np.float32)
        raw_x[:self.n_features[i]] = new_row.to_numpy(dtype=np.float32)

        if get_text:
            return f"Example: {template.apply_template(note=txt_x, label=int(raw_y))}\n" # type: ignore
        else:
            return {'queries_x': raw_x, 
                    'queries_y': raw_y,
                    'shots': f"Example: {template.apply_template(note=txt_x, label=int(raw_y))}\n" # type: ignore
                }  # Return the raw input and text
                
    