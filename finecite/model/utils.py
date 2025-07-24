import os
import torch
import gc
from torch import Tensor
from collections import defaultdict
from typing import Dict
from transformers import AutoModel, AutoTokenizer, AutoConfig, BitsAndBytesConfig
from peft import PeftConfig, PeftModel, LoraConfig, get_peft_model

MODEL_DESCRIPTION = {
    'scibert': {
        'source': 'allenai/scibert_scivocab_uncased',
        'max_len': 512,
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
        'max_len': 740,
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
        'max_len': 740,
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

def load_scibert(args, pretrained_mode_dir):
    if pretrained_mode_dir: 
        print(f'loading {args.model_name} model from {pretrained_mode_dir}')
        source = pretrained_mode_dir
    else: 
        print(f'loading {args.model_name} model...')
        source = args.model_desc['source']
    config = AutoConfig.from_pretrained(source)
    embedding_model = AutoModel.from_pretrained(
        source,
        trust_remote_code=True,
        config=config, 
        torch_dtype=args.dtype,

    )
    embedding_model.to(args.device)
    return embedding_model

def laod_llm2vec(args, pretrained_mode_dir):
    # we cache the base model, merge the adapters and reload it quantized and with a new adapter for finetuning
    if 'config.json' not in os.listdir(args.base_model_dir):
        print('loading base model...')
        temp_config = AutoConfig.from_pretrained(args.model_desc['source_peft_weights1'])
        temp_model = AutoModel.from_pretrained(
            args.model_desc['source'],
            trust_remote_code=True,
            config=temp_config, 
            torch_dtype=args.dtype, 
    
        )
        temp_model.to(args.device)

        print('merging peft weights...')
        temp_model = PeftModel.from_pretrained(
            temp_model,
            args.model_desc['source_peft_weights1'], 
        )
        temp_model = temp_model.merge_and_unload() 

        # Loading supervised model. This loads the trained LoRA weights on top of MNTP model. 
        # Hence the final weights are -- Base model + MNTP (LoRA) + supervised (LoRA).
        temp_model = PeftModel.from_pretrained(
            temp_model,
            args.model_desc['source_peft_weights2'],
    
        )
        temp_model = temp_model.merge_and_unload()
        
        #save base model
        temp_model.save_pretrained(args.base_model_dir)
        temp_model, temp_config = None, None
        gc.collect()
        torch.cuda.empty_cache()

    # load base model
    print('loading LLM2VEC...')
    config = AutoConfig.from_pretrained(args.base_model_dir)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_storage=torch.bfloat16,
    )
    embedding_model = AutoModel.from_pretrained(
        args.base_model_dir,
        trust_remote_code=True,
        config=config,
        quantization_config = bnb_config,
        torch_dtype=args.dtype,
        device_map="auto",
    )

    print('loading peft adapter...')
    if pretrained_mode_dir:
        print(f'loading lora adapter from {pretrained_mode_dir}')
        embedding_model = PeftModel.from_pretrained(
            embedding_model,
            pretrained_mode_dir
        )
    else: 
        print('initializing lora adapter...')
        peft_config = LoraConfig(target_modules=['embed_tokens',"v_proj", "q_proj", "up_proj", "o_proj", "k_proj", "down_proj", "gate_proj" ], inference_mode=False, r=16, lora_alpha=32, lora_dropout=0.05)         
        embedding_model = get_peft_model(embedding_model, peft_config)
    print(embedding_model.print_trainable_parameters())
    
    return embedding_model

def load_tokenizer_embedding_model(args, pretrained_mode_dir = None):
    
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_desc['source'])
    if 'modify_token' in args.model_desc: # add cls, sep, pad tokenids
        token_desc = args.model_desc['modify_token']
        tokenizer.cls_token = token_desc['cls_token']
        tokenizer.cls_token_id = tokenizer.convert_tokens_to_ids(tokenizer.cls_token)
        tokenizer.sep_token = token_desc['sep_token']
        tokenizer.sep_token_id = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)
        tokenizer.pad_token = token_desc['pad_token']
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    
    #load model
    if args.model_name in ['scibert']:
        embedding_model = load_scibert(args, pretrained_mode_dir)
    elif args.model_name in ['llm2vec_mistral', 'llm2vec_llama3']:
        embedding_model = load_scibert(args, pretrained_mode_dir)
    else:
        raise NotImplementedError(f"Loading {args.model_name} is not implemented.")
    
    return tokenizer, embedding_model

def load_classifier(self, path):
    if 'ext' in self.args.task:
        self.ext_cls.load_state_dict(torch.load(os.path.join(path, 'ext_classifier.pt'), weights_only=True))
        if self.args.ext_type in ['crf', 'bilstm_crf']:
            self.ext_loss.load_state_dict(torch.load(os.path.join(path, 'crf_parameter.pt')))
    if 'cls' in self.args.task:
        self.cls_cls.load_state_dict(torch.load(os.path.join(path, 'cls_classifier.pt'), weights_only=True))