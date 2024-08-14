'''
for segment in sentence_prio sentence_majo token_scibert
do
    for mode in scopes total
    do 
        python3 seq_tagger_dc_extraction.py ${segment} ${mode} sdp_act
    done
done
'''

import os
import json
from tqdm import tqdm
from collections import Counter, defaultdict
import torch
import torchmetrics
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from torch.nn import CrossEntropyLoss
import pandas as pd
from peft import PeftModel, LoraConfig, TaskType, get_peft_model
from datetime import datetime
import sys
import re

SPECIAL_TOKEN = ['#AUTHOR_TAG', '#TAUTHOR_TAG']
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

DATASET = sys.argv[3]
SEGMENT = sys.argv[1]
SEGMENT_TYPE = 'sentence' if 'sentence' in SEGMENT else 'token'
MODEL_NAME =  'scibert' #"allenai/scibert_scivocab_uncased"  # "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp","McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp"
MODE = sys.argv[2]
MODEL_PATH = f'/raid/deallab/CCE_Data/model_training/output/seq_tagger/fine_cite/{MODEL_NAME}/{SEGMENT}_{MODE}_2_1e-05_0.0/model_state_e_9.pt'

# Define the argument values as Python variables
num_labels = 4 if MODE == 'scopes' else 2
batch_size = 2
data_parallel = True

with open(f'/home/deallab/lasse/CCE/postprocessing/output/finecite_{MODE}_weights.json') as weights_file:
    weights = json.load(weights_file)
    
INPUT_DIR = f'/raid/deallab/CCE_Data/model_evaluation/classification_task/data/{DATASET}/'
#INPUT_DIR = f'/raid/deallab/CCE_Data/model_training/data/seq_tagger/fine_cite/sentence_majo_99__07-01-02/'
OUTPUT_DIR = f'/raid/deallab/CCE_Data/model_evaluation/classification_task/data/{DATASET}/finecite/'
os.makedirs(os.path.join(OUTPUT_DIR, 'd_nc', f'{SEGMENT}_{MODE}'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'd_c', f'{SEGMENT}_{MODE}'), exist_ok=True)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if MODEL_NAME == 'scibert':
    tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    tokenizer.add_special_tokens({'additional_special_tokens': SPECIAL_TOKEN})
    config = AutoConfig.from_pretrained("allenai/scibert_scivocab_uncased")
    LMmodel = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased", config=config, torch_dtype=torch.bfloat16)
    LMmodel.resize_token_embeddings(len(tokenizer))
class SeqTagger(torch.nn.Module):
    def __init__(
        self,
        tokenizer,
        model,
        num_labels,
        **kwargs,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.model_config = model.config
        self.model = torch.nn.DataParallel(model) if data_parallel else model
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(self.model_config.hidden_size, num_labels, dtype=torch.bfloat16)
        self.loss_fn = CrossEntropyLoss(weight=torch.BFloat16Tensor(weights[SEGMENT]).to(device))

    
    def predict(self, input_ids, attention_mask):
        ids = input_ids.to(device, dtype = torch.long)
        mask = attention_mask.to(device, dtype = torch.long)
        output = self.model(input_ids= ids, attention_mask = mask)
        if SEGMENT_TYPE == 'sentence':
            cls_output_state = output["last_hidden_state"][ids == self.tokenizer.cls_token_id]
        if SEGMENT_TYPE == 'token':
            cls_output_state = output['last_hidden_state'][attention_mask == 1]
        logits = self.classifier(cls_output_state)
        return logits

model = SeqTagger(model=LMmodel, tokenizer=tokenizer, num_labels=num_labels)
model.to(device)
model.load_state_dict(torch.load(MODEL_PATH))

# helper
def replace_authortags(sent):
    regex1 = r'((?:\( *)?(?:(?:(?:(?:[A-Z][A-Za-z\'\`-]+)(?:,? (?:(?:and |& )?(?:[A-Z][A-Za-z\'\`-]+)|(?:et al.?)))* ?(?:, *\(? *(?:19|20)[0-9][0-9](?:, p.? [0-9]+)?| *\\((?:19|20)[0-9][0-9](?:, p.? [0-9]+)? *\)?\\)))(?: |;|,|and)*)+)?#AUTHOR_TAG(?:(?: |;|,|and)*(?:(?:(?:[A-Z][A-Za-z\'\`-]+)(?:,? (?:(?:and |& )?(?:[A-Z][A-Za-z\'\`-]+)|(?:et al.?)))* ?(?:, *\(? *(?:19|20)[0-9][0-9](?:, p.? [0-9]+)?| *\\((?:19|20)[0-9][0-9](?:, p.? [0-9]+)? *\)?\\)))(?: |;|,|and)*)+)?(?: *\))?)'
    regex2 = r'((?:\( *)?(?:(?:[A-Z][a-z\'\`-]+)(?:,? (?:(?:and |& )?(?:[A-Z][a-z\'\`-]+)|(?:et al.?)))* ?(?:, *\(? *(?:19|20)[0-9][0-9](?:, p.? [0-9]+)?| *\\((?:19|20)[0-9][0-9](?:, p.? [0-9]+)? *\)?\\))(?: |;|,|and)*)+(?: *\))?|(?:(?:[A-Z][a-z\'\`-]+)(?:,? (?:(?:and |& )?(?:[A-Z][a-z\'\`-]+)|(?:et al.?)))* *\( *(?:19|20)[0-9][0-9](?:, p.? [0-9]+)? *\)?))'
    tauthro_tags = [match[0] for match in re.findall(regex1, sent)]
    sent = re.sub(regex1, '#TAUTHOR_TAG ', sent)
    author_tags = [match[0] for match in re.findall(regex2, sent)]
    sent =  re.sub(regex2, '#AUTHOR_TAG ', sent)
    assert len(re.findall(r'#T?AUTHOR_TAG',sent)) == len(tauthro_tags) + len(author_tags), f'{sent}, {tauthro_tags}, {author_tags}'
    return sent, tauthro_tags, author_tags

def find_cite_sent_id(par_list):
    cite_sent_id = -1
    for i, sent in enumerate(par_list):
        if re.search(r'#AUTHOR_TAG', sent):
            cite_sent_id = i
            break
    return cite_sent_id

def build_contiguous_context(segments, labels, start_id):
    context=[segments[start_id]]
    prev_id = start_id -1
    next_id = start_id +1
    while prev_id >=0:
        if labels[prev_id] != 0:
            context.insert(0,segments[prev_id])
            prev_id -= 1
        else: break
    while next_id < len(segments):
        if labels[next_id] != 0:
            context.append(segments[next_id])
            next_id += 1
        else: break
    return context
    
def build_sentence_context(row):
    par_list = eval(row['cite_context_paragraph'])
    cite_sent_id = find_cite_sent_id(par_list)
    if cite_sent_id == -1:
        print('no author tag')
        return [],[]
    
    # prepare for tokinization
    inputs = [replace_authortags(sent)[0] for sent in par_list]
    inputs = [seg.replace(tokenizer.cls_token, 'cls_token') for seg in inputs] # replace [CLS] / <s> token in text
    inputs = f' {tokenizer.cls_token} '.join(inputs)
    input_ids = tokenizer.encode(inputs)
    
    # ignore sentences over the 512 token limit, delete them but make sure the cls token are still there
    if len(input_ids) > 512:
        cls_token_list = []
        while len(input_ids) + len(cls_token_list) > 512:
            item = input_ids.pop()
            if item == tokenizer.cls_token_id:
                cls_token_list.insert(0, item)
        input_ids.extend(cls_token_list)
    assert len(input_ids) <= 512, f'the input ids are longer than the max amount of token: {len(input_ids), len(input_ids) > 512}'
    
    input_mask = torch.tensor([[1] * len(input_ids) + [0] * (512 - len(input_ids))])
    input_ids = torch.tensor([input_ids + [tokenizer.pad_token_id] * (512 - len(input_ids))])
    model.eval()
    output = model.predict(input_ids, input_mask)
    preds = output.argmax(-1)
    if preds[cite_sent_id] == 0:
        preds[cite_sent_id] = 1
        
    context_c = build_contiguous_context(par_list, preds, cite_sent_id)
    context_nc = [sent for i, sent in enumerate(par_list) if preds[i] != 0]
    return context_nc, context_c
    
    
# build context for token segmentation
def build_token_context(row):
    par_list = eval(row['cite_context_paragraph'])
    cite_sent_id = find_cite_sent_id(par_list)
    if cite_sent_id == -1:
        print('no author tag')
        return [],[]
    par_string = ' '.join(par_list)
    inputs, tauthor_token, author_token = replace_authortags(par_string)      
    input_ids = tokenizer.encode(inputs, add_special_tokens=False)
    if len(input_ids) > 510:
        input_ids = input_ids[:510]
        deleted_token = ' '.join(tokenizer.convert_ids_to_tokens(input_ids[510:]))
        print(deleted_token)
        if re.search(r'#T?AUTHOR_TAG', deleted_token):
            print('deleted author tag')
            for _ in re.findall(r'#AUTHOR_TAG', deleted_token):
                author_token.pop()
            for _ in re.findall(r'#AUTHOR_TAG', deleted_token):
                tauthor_token.pop()
    #add special token to end and start
    input_ids = [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]
    #check input length < 512 token 
    assert len(input_ids) <= 512, f'the input ids are longer than the max amount of token: {len(input_ids), len(input_ids) > 512}'
    
    input_mask_tensor = torch.tensor([[1] * len(input_ids) + [0] * (512 - len(input_ids))])
    input_ids_tensor = torch.tensor([input_ids + [tokenizer.pad_token_id] * (512 - len(input_ids))])
    
    model.eval()
    output = model.predict(input_ids_tensor, input_mask_tensor)
    preds = output.argmax(-1)
    assert len(preds) == len(input_ids), f'the len of preds is {len(preds)} and the len of the input ids is {len(input_ids)}'
    
    input_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    author_tag_ids = [i for i, token in enumerate(input_tokens) if token=='#AUTHOR_TAG']
    tauthor_tag_ids = [i for i, token in enumerate(input_tokens) if token=='#TAUTHOR_TAG']
    assert len(author_tag_ids) <= len(author_token) and len(tauthor_tag_ids) <= len(tauthor_token), f'{input_tokens}, {tauthor_token}, {author_token}, {len(tauthor_tag_ids)}, {len(author_tag_ids)}'
    for i, id in enumerate(author_tag_ids):
        input_tokens[id] = author_token[i]
    for i, id in enumerate(tauthor_tag_ids):
        input_tokens[id] = tauthor_token[i]
    if tauthor_tag_ids:
        context_c = build_contiguous_context(input_tokens, preds, tauthor_tag_ids[0])
    else: context_c = []
    context_nc = [sent for i, sent in enumerate(input_tokens) if preds[i] != 0]
    return context_nc, context_c
    
for dataset in ['train', 'test']:
    print(SEGMENT, dataset)
    data_df = pd.read_csv(INPUT_DIR + f"/{dataset}_raw.txt", sep="\t", engine="python", dtype=object)
    res_df_nc = pd.DataFrame(columns=['CC','label'])
    res_df_c = pd.DataFrame(columns=['CC','label'])
    for idx, row in tqdm(data_df.iterrows()):
        label = row['citation_class_label']
        if SEGMENT_TYPE == 'sentence':
            try: 
                context_nc, context_c = build_sentence_context(row)
                if context_nc:
                    res_df_nc.loc[len(res_df_nc)] = [context_nc, label]
                if context_c:
                    res_df_c.loc[len(res_df_c)] = [context_c, label]
            except: 
                print('there was a issue with the context')
                continue
        elif SEGMENT_TYPE == 'token':
            # try: 
            context_nc, context_c = build_token_context(row)
            if context_nc:
                res_df_nc.loc[len(res_df_nc)] = [context_nc, label]
            if context_c:
                res_df_c.loc[len(res_df_c)] = [context_c, label]
            # except: 
            #     print('there was a issue with the context')
            #     continue
    res_df_nc.to_csv(os.path.join(OUTPUT_DIR, 'd_nc', f'{SEGMENT}_{MODE}', f'{dataset}.csv'), index=False)
    res_df_c.to_csv(os.path.join(OUTPUT_DIR, 'd_c', f'{SEGMENT}_{MODE}', f'{dataset}.csv'), index=False)