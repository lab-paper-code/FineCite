import pandas as pd
import os
import json
from typing import List
from collections import Counter

import torch
from transformers import PreTrainedTokenizer
from argparse import Namespace

from finecite.data_processing.prompts import PROMPTS
    

class Processor(object):
    def __init__(self, args: Namespace, tokenizer: PreTrainedTokenizer):
        self.args = args
        self.path = args.input_dir
        self.tokenizer = tokenizer
        self.num_special_token = 2
        self.max_len = args.max_len
        self.max_inp_tok = self.max_len - self.num_special_token
        
        # set prompt
        if 'llm2vec' in self.args.model_name and self.args.use_prompt:
            self.prompt = PROMPTS['llm2vec_std']
            self.prompt_token = self.tokenizer.apply_chat_template(self.prompt)[:-1]
            self.prompt_len = len(self.prompt_token) - 1
            self.max_inp_tok = self.max_len - self.prompt_len - self.num_special_token
        
    def load_csv(self, split: str) -> List[dict]:
        data = pd.read_csv(os.path.join(self.path, f'{split}.csv')).to_dict('records')
        if self.args.debug and self.args.debug_size:
            data = data[:self.args.debug_size]
        return data
    
    def load_jsonl(self, split: str) -> List[dict]:
        data = []
        with open(os.path.join(self.path, f'{split}.jsonl')) as f:
            for line in f:
                if not line.strip(): continue
                data.append(json.loads(line))
        if self.args.debug and self.args.debug_size:
            data = data[:self.args.debug_size]
        return data
    
    def _convert_to_iob(self, context: list, num_labels: int):
        prev=0
        for i in range(len(context)):
            l = context[i]
            if prev != 0 and l == prev:
                context[i] = l + num_labels
            prev = l
        return context
    
    def _calc_class_weights(self, flat_labels: list[int]):
        flat_labels = [l for l in flat_labels if l != -100]
        counter = Counter(flat_labels)
        sorted_counter = sorted(counter.items())
        counter_sum = sum(counter.values())
        ratio_scopes = [counter_sum / (len(counter) * c )for l, c in sorted_counter]
        return ratio_scopes

class FineciteProcessor(Processor):
    def read_data(self, split: str) -> List[dict]:
        data = self.load_jsonl(split)
        examples = []
        for idx, sample in enumerate(data):
            word_list =  sample['masked_paragraph']
            text = ' '.join(sample['masked_paragraph'])
            word_labels = sample['annotations']
            examples.append({
                'id': idx,
                'text': text,
                'word_list': word_list,
                'word_labels': word_labels
            })
        return examples

    def create_features(self, data: List[dict]):
        features = []
        for example in data:
            word_list = example['word_list']
            word_labels = example['word_labels']
            annotation_1 = word_labels[0]
            annotation_2 = word_labels[1] if len(word_labels) == 2 else word_labels[0]
            
            # convert word labels to token labels
            input_ids, token_annotation_1, token_annotation_2 = [], [], []
            for word, lbl, lbl2 in zip(word_list, annotation_1, annotation_2):
                word_ids = self.tokenizer.encode(word, add_special_tokens=False)
                input_ids.extend(word_ids)
                token_annotation_1.extend([lbl] * len(word_ids))
                token_annotation_2.extend([lbl2] * len(word_ids))
            
            # convert uniform labeling to IOB labeling
            if self.args.iob_labels:
                token_annotation_1, token_annotation_2 = self._convert_to_iob(token_annotation_1, num_labels=3), self._convert_to_iob(token_annotation_2, num_labels=3)
                
            # restrict input_ids to max len
            if len(input_ids) > self.max_inp_tok:
                dif = len(input_ids) - (self.max_inp_tok)
                nr_ann_first_n = sum([1 if label != 0 else 0 for label in token_annotation_1[:dif]])
                nr_ann_last_n = sum([1 if label != 0 else 0 for label in token_annotation_1[-dif:]])
                if nr_ann_last_n <= nr_ann_first_n:
                    input_ids = input_ids[:-dif]
                    token_annotation_1 = token_annotation_1[:-dif]
                    token_annotation_2 = token_annotation_2[:-dif]
                else:
                    input_ids = input_ids[dif:]
                    token_annotation_1 = token_annotation_1[dif:]
                    token_annotation_2 = token_annotation_2[dif:]
            assert len(input_ids) <= self.max_inp_tok and len(input_ids) == len(token_annotation_1) and len(input_ids) == len(token_annotation_2), f'Missmatch in data length (input_ids, labels, labels2): {len(input_ids), len(token_annotation_1), len(token_annotation_2)}'
            
            #add special token and prompt
            if 'llm2vec' in self.args.model_name and self.args.use_prompt:
                input_ids = self.prompt_token + input_ids + [self.tokenizer.sep_token_id]
                token_annotation_1 = [-100] * (self.prompt_len + 1) + token_annotation_1 + [-100]
                token_annotation_2 = [-100] * (self.prompt_len + 1) + token_annotation_2 + [-100]
            else:
                input_ids = [self.tokenizer.cls_token_id] + input_ids + [self.tokenizer.sep_token_id]
                token_annotation_1 = [-100] + token_annotation_1 + [-100]
                token_annotation_2 = [-100] + token_annotation_2 + [-100]
            assert len(input_ids) == len(token_annotation_1) and len(input_ids) == len(token_annotation_2), f'Missmatch in data length (input_ids, labels, labels2): {len(input_ids), len(token_annotation_1), len(token_annotation_2)}'
            
            # add input mask
            input_mask = [1] * len(input_ids)
            
            # add max padding
            raw_len = len(input_ids)
            input_ids = input_ids + [self.tokenizer.pad_token_id] * (self.max_len - raw_len)
            input_mask = input_mask + [0] * (self.max_len - raw_len)
            token_annotation_1 = token_annotation_1 + [-100] * (self.max_len - raw_len)
            token_annotation_2 = token_annotation_2 + [-100] * (self.max_len - raw_len)
            assert len(input_ids) == self.max_len and len(input_ids) == len(input_mask) and len(input_ids) == len(token_annotation_1) and len(input_ids) == len(token_annotation_2), f'Missmatch in data length (input_ids, labels, labels2): {len(input_ids), len(token_annotation_1), len(token_annotation_2)}'
                
            features.append({
                    'input_ids': torch.tensor(input_ids), 
                    'input_mask': torch.tensor(input_mask), 
                    'token_labels': torch.tensor([token_annotation_1, token_annotation_2], dtype=torch.long) if len(example['word_labels']) == 2 else torch.tensor([token_annotation_1], dtype=torch.long)
                    })
            
        # calculate class weights
        flat_labels = [l for feat in features for l in feat['token_labels'][0].tolist()]
        weights = self._calc_class_weights(flat_labels)
        num_labels = len(weights)
        
        return features, weights, num_labels
    
    
class CLSProcessor(Processor):    
    def read_data(self, split: str)  -> List[dict]:
        data = self.load_csv(split)
        examples = []
        for idx, row in enumerate(data):
            text = row['context']
            intent_labels = eval(row['intent_labels'])
            examples.append({
                'id': idx,
                'text': text,
                'intent_labels': intent_labels,
            })
        return examples

    def create_features(self, examples: List[dict]):
        features = []
        # calculate class weights
        flat_labels = [l for example in examples for l in example['intent_labels']]
        weights = self._calc_class_weights(flat_labels)
        num_labels = len(weights)
        
        for example in examples:
            text = example['text']
            intent_labels = example['intent_labels']
            
            # tokenize txt input
            input_ids = self.tokenizer.encode(text, add_special_tokens=False)
            
            if len(input_ids) > self.max_inp_tok:
                input_ids = input_ids[:self.max_inp_tok]
            assert len(input_ids) <= self.max_inp_tok
            
            #add special token and prompt
            if 'llm2vec' in self.args.model_name and self.args.use_prompt:
                input_ids = self.prompt_token + input_ids + [self.tokenizer.sep_token_id]
            else:
                input_ids = [self.tokenizer.cls_token_id] + input_ids + [self.tokenizer.sep_token_id]
                
            # add input mask
            input_mask = [1] * len(input_ids)
            
            # add max padding
            raw_len = len(input_ids)
            input_ids = input_ids + [self.tokenizer.pad_token_id] * (self.max_len - raw_len)
            input_mask = input_mask + [0] * (self.max_len - raw_len)
            assert len(input_ids) == self.max_len
            
            # make one/multiple hot encoded from labels
            intent_labels = [1 if i in intent_labels else 0 for i in range(num_labels)]
                
            features.append({
                    'input_ids': torch.tensor(input_ids), 
                    'input_mask': torch.tensor(input_mask), 
                    'intent_labels': torch.tensor(intent_labels, dtype = torch.float16), 
                    })
        
        return features, weights, num_labels
    
    
def load_processor(args, tokenizer) -> Processor:
    match args.dataset:
        case 'finecite':
            return FineciteProcessor(args, tokenizer)
        case 'acl-arc'| 'act2'| 'ACT-D'| 'ACT-ND'|'scicite'| 'multicite':
            return CLSProcessor(args, tokenizer)
        case _:
            raise NotImplementedError(f'There is no processor for the dataset {args.dataset}')