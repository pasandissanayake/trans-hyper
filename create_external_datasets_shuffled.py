import argparse
import itertools
import logging
import os
import random
import re
import datetime
from collections import Counter
from pathlib import Path
import math

import numpy as np
from datasets import Dataset

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.trainer_utils import set_seed
from transformers import AutoModelForSeq2SeqLM

from tabllm.helper.note_generator import NoteGenerator
from tabllm.helper.note_template import NoteTemplate
from tabllm.helper.external_datasets_variables import *
from tabllm.helper.preprocess import preprocess
from tabllm.dataset_utils import load_dataset

logger = logging.getLogger(__name__)

cat_idx_dict = {
    "car": [0,1,2,3,4,5],
    "diabetes": [],
    "heart": [1,2,6,8,10],
    "income": [1,2,3,4,5,6,7,11],
    "creditg": [0,2,3,4,5,6,8,9,11,13,14,16,18,19],
    "blood": [],
    "bank": [1,2,3,4,6,7,8,10,15],
    "jungle": [],
    "calhousing": [],
}
bin_num = 10

def main():
    """
    args:
        --seed: random seed for reproducibility
        --datadir: path for raw datasets
        --dataset: dataset names to serialize
        --outdir: path for output directory
        --shuffled: shuffle data
        --tabletotext:
        --t0serialization:
    """

    args = parse_args()
    set_seed(args.seed)
    logging.basicConfig(level=logging.INFO)

    # Configuration
    data_dir = Path(args.datadir)
    data_dir = data_dir / args.dataset
    temp_output = 'dataset-generation-' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(args.outdir) / temp_output
    if not args.debug:
        os.mkdir(output_dir)
    logger.info(f"Generate dataset {args.dataset}.")

    if not args.list and args.permuted:
        raise ValueError("Permuted note is not supported.")
    dataset_name = args.dataset + \
                   ('_list' if args.list else '') + \
                   ('_permuted' if args.permuted else '') + \
                   ('_values' if args.values else '') + \
                   ('_shuffled' if args.shuffled else '') + \
                   ('_importance' if args.feature_importance else '')
    dataset = load_dataset(args.dataset, data_dir)

    # template, template_config = None, None
    template = eval('template_' + dataset_name)
    template_config = eval('template_config_' + dataset_name)
    note_generator = NoteTemplate(template, **template_config)

    # Shuffled: shuffle each feature column separately
    if args.shuffled:
        # np.random.seed(42)
        def derange(n):
            orig = np.arange(n)
            derangement = orig
            while np.any(orig == derangement):
                derangement = np.random.permutation(orig)
            return derangement
        
        def shuffle_dataset(dataset):
            cat_idx = cat_idx_dict[args.dataset]
            derangement_dict = {}
            for column_idx, c in enumerate(dataset.columns):
                if column_idx in cat_idx and c != 'label':
                    derangement_dict[c] = {}
                    value_set = list(set(dataset[c].values))
                    derangement = derange(len(value_set))
                    derangement_dict[c] = {value: value_set[derangement[i]] for i, value in enumerate(value_set)}
                    dataset[c] = [derangement_dict[c][value] for value in dataset[c]]
                if column_idx not in cat_idx and c!= 'label':
                    value_list = dataset[c].values
                    ret_value_list = []
                    num_values = len(value_list)
                    sorted_value_list = sorted(list(value_list))
                    derangement = derange(bin_num)

                    bin_idx_intervals = []
                    bin_idx_endpoints = []
                    factor = num_values / bin_num
                    for bin_idx in range(bin_num):
                        lower_idx, upper_idx = math.floor(bin_idx * factor), math.floor((bin_idx + 1) * factor)
                        bin_idx_intervals.append([lower_idx, upper_idx])
                        bin_idx_endpoints.append([sorted_value_list[lower_idx], sorted_value_list[upper_idx-1]])

                    for value in value_list:
                        for bin_idx, (lower_value, upper_value) in enumerate(bin_idx_endpoints):
                            if value >= lower_value and value <= upper_value:
                                mapped_bin_lower_idx, mapped_bin_upper_idx = bin_idx_intervals[derangement[bin_idx]]
                                sampled_bin_values = sorted_value_list[mapped_bin_lower_idx : mapped_bin_upper_idx]
                                ret_value_list.append(random.choice(sampled_bin_values))
                                break
                    dataset[c] = ret_value_list
            return dataset
        
        dataset = shuffle_dataset(dataset)

    notes = [NoteGenerator.clean_note(note_generator.substitute(r)) for _, r in dataset.iterrows()]
    old_size_notes = len(notes)
    start = 0  # 25000
    end = len(notes)
    notes = notes[start:end]
    dataset = dataset.iloc[start:end]
    print(f"Only consider dataset range between {start} and {end} (total: {old_size_notes})")

    # Apply modifications based on the list format
    # Table-To-Text
    if args.tabletotext or args.t0serialization:
        if args.tabletotext:
            tokenizer = AutoTokenizer.from_pretrained("Narrativaai/bloom-560m-finetuned-totto-table-to-text")
            model = AutoModelForCausalLM.from_pretrained("Narrativaai/bloom-560m-finetuned-totto-table-to-text").to("cuda")
        else:
            tokenizer = AutoTokenizer.from_pretrained('bigscience/T0')
            model = AutoModelForSeq2SeqLM.from_pretrained('bigscience/T0').to("cuda")

        def serialize(ex):
            inputs = tokenizer(ex['text'], return_tensors='pt', padding=True)
            input_ids = inputs.input_ids.to("cuda")
            attention_mask = inputs.attention_mask.to("cuda")
            output = model.generate(input_ids, attention_mask=attention_mask, max_length=len(input_ids[0]) + 50,
                                    eos_token_id=tokenizer.eos_token_id)
            ex['out'] = tokenizer.decode(output[0], skip_special_tokens=False)
            return ex

        if args.tabletotext:
            num_features = len(dataset.columns) - 1

            def write_into_table(name, value):
                example = {}
                example['table_page_title'] = ''
                example['table_section_title'] = ''
                example['table'] = [[{'column_span': 1, 'is_header': True, 'row_span': 1, 'value': name}],
                                    [{'column_span': 1, 'is_header': False, 'row_span': 1, 'value': value}]]
                example['highlighted_cells'] = [[0, 0], [1, 0]]
                return example

            def table_to_text(note):
                re_name_value = re.compile(r"^- (.*):([^:]*)$", re.MULTILINE)
                name_values = re_name_value.findall(note)
                examples = [write_into_table(x[0].strip(), x[1].strip()) for x in name_values]
                return [preprocess(e)['linearized_table'] for e in examples]

            # notes = notes[0:10]
            old_size = len(notes)
            notes = Dataset.from_dict({'text': list(itertools.chain(*[table_to_text(n) for n in notes]))})
            assert notes.shape[0] == num_features * old_size, f"notes.shape[0]: {notes.shape[0]}, num_features: {num_features}, old_size: {old_size}"
            notes = notes.map(serialize)
            # Debug
            notes.save_to_disk(output_dir / (dataset_name + '_debug'))
            notes = [(((ex['out'].split('>')[-2]).split('<')[0]).replace('\n', ' ')).strip() for ex in notes] # type: ignore
            notes = [' '.join(l) for l in [notes[x:x+num_features] for x in range(0, len(notes), num_features)]]

        if args.t0serialization:
            def entry_to_text(note):
                prefix = 'Write this information as a sentence: '
                suffix = '. \n'
                re_name_value = re.compile(r"^- (.*):([^:]*)$", re.MULTILINE)
                name_values = re_name_value.findall(note)
                lines = note.splitlines()[0:len(name_values)]
                lines = [l[2:] for l in lines]
                chunks = [(prefix + ', '.join(lines[k:k + 2]) + suffix) for k in range(0, len(lines), 2)]
                return chunks

            old_size = len(notes)
            num_chunks = int(((len(dataset.columns) - 1) / 2.) + 0.5)
            # notes = notes[0:10]
            notes = Dataset.from_dict({'text': list(itertools.chain(*[entry_to_text(n) for n in notes]))})
            assert notes.shape[0] == old_size * num_chunks
            notes = notes.map(serialize)
            # Debug
            notes.save_to_disk(output_dir / (dataset_name + '_debug'))
            notes = [ex['out'][6:-4] for ex in notes] # type: ignore
            notes = [' '.join(l) for l in [notes[x:x + num_chunks] for x in range(0, len(notes), num_chunks)]]

    for i in range(0, min(10, len(notes))):
        print('----')
        print(notes[i])
    dataset = Dataset.from_dict({'note': notes, 'label': dataset['label'].to_list()})

    if not args.debug:
        logger.info(f"Store generated datasets to {output_dir}/{dataset_name}")
        logger.info(f"\tn={len(dataset)}, feats={dataset.num_columns}, labels={dict(Counter(dataset['label']))}")
        dataset.save_to_disk(output_dir / dataset_name)


def parse_args():
    parser = argparse.ArgumentParser(description="Create note dataset from cohort.")
    parser.add_argument(
        "--debug",
        action="store_true",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--dataset",
        type=str
    )
    parser.add_argument(
        "--datadir",
        type=str
    )
    parser.add_argument(
        "--outdir",
        type=str
    )
    parser.add_argument(
        "--list",
        action="store_true",
    )
    parser.add_argument(
        "--tabletotext",
        action="store_true",
    )
    parser.add_argument(
        "--t0serialization",
        action="store_true",
    )
    parser.add_argument(
        "--permuted",
        action="store_true",
    )
    parser.add_argument(
        "--values",
        action="store_true",
    )
    parser.add_argument(
        "--shuffled",
        action="store_true",
    )
    parser.add_argument(
        "--feature_importance",
        action="store_true",
    )

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    main()