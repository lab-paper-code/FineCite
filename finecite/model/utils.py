import torch
from torch import Tensor
from collections import defaultdict
from typing import Dict

MODEL_DESCRIPTION = {
    'scibert': {
        'source': 'allenai/scibert_scivocab_uncased',
    },
    'modern_bert': {
        'source': 'bert-base-uncased'
        },
    'llm2vec_mistral':{
        'source': "mistralai/Mistral-7B-Instruct-v0.2",
        'source_peft_weights1':'McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp',
        'source_peft_weights2': 'McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp-supervised',
        'modify_token': {
            'cls_token': '<s>',
            'sep_token': '</s>',
            'pad_token': '</s>',   
        },
    },
    'llm2vec_llama3':{
        'source': "meta-llama/Meta-Llama-3.1-8B-Instruct",
        'source_peft_weights1':'McGill-NLP/LLM2Vec-Meta-Llama-31-8B-Instruct-mntp',
        'source_peft_weights2': 'McGill-NLP/LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-supervised',
        'modify_token': {
            'cls_token': '<|begin_of_text|>',
            'sep_token': '<|end_of_text|>',
            'pad_token': '<|end_of_text|>',
        },
    }
}

class CustomDefaultdict(defaultdict):
    def extend(self, a: Dict[str, Tensor]):
        for k, v in a.items():
            self[k].extend(v)
    def append(self, a: Dict[str, Tensor]):
        for k, v in a.items():
            self[k].append(v)
    def stack(self):
        for k in self.keys():
            self[k] = torch.stack(self[k], dim=0)
    def sample(self, batch_size, argument):
        min_loss, min_idx = self[argument].min(dim=0)
        max_loss, max_idx = self[argument].max(dim=0)
        dict_min = {k:v[min_idx*batch_size:(min_idx+1) * batch_size] if  not 'loss' in k else v[min_idx:min_idx+1] for k,v in self.items()} 
        dict_max = {k:v[max_idx*batch_size:(max_idx+1) * batch_size] if  not 'loss' in k else v[max_idx:max_idx+1] for k,v in self.items()}
        return {key: torch.cat([dict_min[key], dict_max[key]], dim=0) for key in set(dict_min) & set(dict_max)}