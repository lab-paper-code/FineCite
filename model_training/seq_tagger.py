# region imports
import os
import json
from tqdm import tqdm
from collections import Counter, defaultdict
import torch
import torchmetrics
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_linear_schedule_with_warmup, AutoModelForCausalLM
from torch.nn import CrossEntropyLoss
import pandas as pd
from peft import PeftModel, LoraConfig, TaskType, get_peft_model
from datetime import datetime
import bitsandbytes as bnb
from transformers import BitsAndBytesConfig
import argparse
import gc
import random
import numpy as np
#endregion
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed_value=42):
    # Set seed for reproducibility.
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
set_seed()

#Use the ArgumentParser library
parser = argparse.ArgumentParser(description='Seq_tagger parser')

# ---input arguments---

#choose the model
parser.add_argument('--model_name', required=True, help='scibert llm2vec_mistral llm2vec_llama3')

#choose the segment type 
parser.add_argument('--segment', required=True, help='sentence_prio sentence_majo token_scibert token_mistral token_llama')

#choose the model running tpye by scopes or total
parser.add_argument('--mode', required=True, help='scopes total')

#choose the batch size of input
parser.add_argument('--batch_size', default=2, help='1 2 4')

#choose the learning rate of model training
parser.add_argument('--learning_rate', default=1e-05, help='3e-05 5e-5 8e-5 1e-04')

#choose the dropout parameter during model training
parser.add_argument('--dropout', default=0.0, help='0.04 0.05 0.06 0.07 0.1 0.12 0.15 0.18 0.2')

#if you want the result of the full data training, tag the '--full_trainging' on right side of the execution
parser.add_argument('--full_training', action='store_true', help='')

parser.add_argument('--debug', action='store_true', help='')
parser.add_argument('--debug_size', default=5, help='')

#if you want the reesult with out background scope, tag the '--no_background' on right side of the execution
parser.add_argument("--no_background", action='store_true', help='')

#execution example
'''
python3 -u seq_tagger.py --model_name scibert --segment token_scibert --mode scopes --learning_rate 5e-05 --dropout 0.1 --batch_size 2 --full_training
'''

#static arguments
args = parser.parse_args()

args.batch_size = int(args.batch_size)
args.learning_rate = float(args.learning_rate)
args.dropout = float(args.dropout)

args.segment_type = args.segment.split('_')[0]

#settings of the dir path
args.data_dir = f"../data/model_training/data/seq_tagger/fine_cite/{args.segment}_69__07-01-02/"
args.base_model_dir = f'../data/model_training/llm2vec_models/{args.model_name}/'
args.model_output_dir = f"../data/model_training/output/seq_tagger/fine_cite/{args.model_name}/{args.segment}_{args.mode}_{args.batch_size}_{args.learning_rate}_{args.dropout}/"
if args.debug:
    args.log_output_dir = f"./output/debug/{args.model_name}/{args.segment}_{args.mode}_{args.batch_size}_{args.learning_rate}_{args.dropout}_{datetime.now().strftime('%m_%d_%H_%M_%S')}/"
elif args.full_training:
    args.log_output_dir = f"./output/{args.model_name}/_full_{args.segment}_{args.mode}_{args.batch_size}_{args.learning_rate}_{args.dropout}/"
elif args.no_background:
    args.log_output_dir = f"./output/{args.model_name}/{args.segment}_{args.mode}_{args.batch_size}_{args.learning_rate}_{args.dropout}_no_background/"
else: 
    args.log_output_dir = f"./output/{args.model_name}/{args.segment}_{args.mode}_{args.batch_size}_{args.learning_rate}_{args.dropout}/"
    
os.makedirs(args.data_dir, exist_ok=True)
os.makedirs(args.base_model_dir, exist_ok=True)
os.makedirs(args.model_output_dir,exist_ok=True)
os.makedirs(args.log_output_dir, exist_ok=True)

args.special_token = ['#AUTHOR_TAG','#TAUTHOR_TAG']
args.num_labels = 4 if args.mode == 'scopes' else 2
with open(f'./finecite_{args.mode}_weights.json') as weights_file:
    weights = json.load(weights_file)
args.label_weights = weights[args.segment]

args.max_epochs = 20
args.patients = 2
args.adam_epsilon = 1e-08
args.weight_decay = 0.0

class MyDataset(Dataset):
    def __init__(self, data, tokenizer, args, max_len = 512):
        self.args = args
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # load labels
        labels = eval(self.data.loc[idx, "labels"])
        if self.args.mode == 'total':
            labels = self.convert_labels_two_class(labels)
        if args.no_background:
            labels = [l if l != 3 else 0 for l in labels]
        
        # load paragraph
        CP = eval(self.data.loc[idx, 'segments'])
        
        if self.args.segment_type == 'sentence':
            CP = [seg.replace(self.tokenizer.cls_token, 'cls_token') for seg in CP] # replace [CLS] / <s> token in text
            CP = f' {self.tokenizer.cls_token} '.join(CP)
            input_ids = self.tokenizer.encode(CP)
            # ignore sentences over the 512 token limit, delete them but make sure the cls token are still there
            if len(input_ids) > 512:
                cls_token_list = []
                while len(input_ids) + len(cls_token_list) > 512:
                    item = input_ids.pop()
                    if item == self.tokenizer.cls_token_id:
                        cls_token_list.insert(0, item)
                input_ids.extend(cls_token_list)
            assert len(labels) == input_ids.count(self.tokenizer.cls_token_id), f'the lenght of the labels is {len(labels)} while the number of [CLS] token is {input_ids.count(tokenizer.cls_token_id)}'
        
        if self.args.segment_type == 'token':     
            input_ids = self.tokenizer.convert_tokens_to_ids(CP)
            if len(input_ids) > 510:
                dif = len(input_ids) - 510
                nr_ann_first_n = sum([1 if label != 0 else 0 for label in labels[:dif]])
                nr_ann_last_n = sum([1 if label != 0 else 0 for label in labels[-dif:]])
                if nr_ann_last_n <= nr_ann_first_n:
                    input_ids = input_ids[:-dif]
                    labels = labels[:-dif]
                else:
                    input_ids = input_ids[dif:]
                    labels = labels[dif:]
            #add special token to end and start
            labels = [-100] + labels + [-100]
            input_ids = [self.tokenizer.cls_token_id] + input_ids + [self.tokenizer.sep_token_id]
            assert len(input_ids) == len(labels), f'the labels ({len(input_ids)}) do not have the same length as the input ids ({len(input_ids)})'
        #check input length < 512 token 
        assert len(input_ids) <= 512, f'the input ids are longer than the max amount of token: {len(input_ids), len(input_ids) > 512}'
        
        #pad input and add generate input mask
        input_mask = [1] * len(input_ids) + [0] * (512 - len(input_ids))
        input_ids = input_ids + [self.tokenizer.pad_token_id] * (512 - len(input_ids))
        labels = labels + [-100] * (512 - len(labels))
        assert len(input_ids) == len(input_mask) and len(input_ids) == len(labels) and len(input_ids) == 512
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            'input_mask': torch.tensor(input_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }
    
    def convert_labels_two_class(self, labels):
        return [0 if label == 0 else 1 for label in labels]
 
class SeqTagger(torch.nn.Module):
    def __init__(
        self,
        model_args,
    ):
        super().__init__()
        self.args = model_args
        self.current_epoch = 0
        self.global_step = 0
        self.dropout = torch.nn.Dropout(self.args.dropout)
        self.loss_fn = CrossEntropyLoss(weight=torch.BFloat16Tensor(self.args.label_weights).to(DEVICE))
        self.metric_multi_acc = torchmetrics.Accuracy(task='multiclass', num_classes=self.args.num_labels)
        self.metric_acc = torchmetrics.Accuracy(task='binary', num_classes=2)
        self.metric_macro_f1 = torchmetrics.F1Score(average='macro', num_classes=self.args.num_labels, task='multiclass')
        self.metric_f1 = torchmetrics.F1Score(average='macro', num_classes=2, task='binary')
    
    def load_pretrained(self, model_name):
        # load the right model
        model_desc_dict = {
            'scibert': {
                'source': 'allenai/scibert_scivocab_uncased',
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
                'source': "meta-llama/Meta-Llama-3-8B-Instruct",
                'source_peft_weights1':'McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp',
                'source_peft_weights2': 'McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised',
                'modify_token': {
                    'cls_token': '<|begin_of_text|>',
                    'sep_token': '<|end_of_text|>',
                    'pad_token': '<|end_of_text|>',
                },
            }
        }

        model_desc = model_desc_dict[model_name]

        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_desc['source'])
        
        self.tokenizer.add_special_tokens({'additional_special_tokens': self.args.special_token})
        if 'modify_token' in model_desc:
            token_desc = model_desc['modify_token']
            self.tokenizer.cls_token = token_desc['cls_token']
            self.tokenizer.sep_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.cls_token)
            self.tokenizer.sep_token = token_desc['sep_token']
            self.tokenizer.sep_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.sep_token)
            self.tokenizer.pad_token = token_desc['pad_token']
            self.tokenizer.sep_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        
        #load model
        if model_name == 'scibert':
            self.config = AutoConfig.from_pretrained(model_desc['source'])

            self.model = AutoModel.from_pretrained(
                model_desc['source'],
                trust_remote_code=True,
                config=self.config, 
                torch_dtype=torch.bfloat16
            )
            self.model.resize_token_embeddings(len(self.tokenizer))

        else: 
            if 'config.json' not in os.listdir(self.args.base_model_dir):
                # At the firt time, Make the model only once
                #load base model
                print('downloading base model')
                config = AutoConfig.from_pretrained(model_desc['source_peft_weights1'])

                LMmodel = AutoModel.from_pretrained(
                    model_desc['source'],
                    trust_remote_code=True,
                    config=config, 
                    torch_dtype=torch.bfloat16
                )
                LMmodel.to(DEVICE)

                #add peft model
                LMmodel = PeftModel.from_pretrained(
                LMmodel,
                model_desc['source_peft_weights1']
                )
                # Merge Peft weights into base model
                LMmodel = LMmodel.merge_and_unload()

                # Loading supervised model. 
                # This loads the trained LoRA weights on top of MNTP model. 
                # Hence the final weights are -- Base model + MNTP (LoRA) + supervised (LoRA).
                LMmodel = PeftModel.from_pretrained(
                    LMmodel,
                    model_desc['source_peft_weights2'],
                )
                # Merge Peft weights into base model
                LMmodel = LMmodel.merge_and_unload()

                #Save the model weights on base_model_dir
                LMmodel.save_pretrained(self.args.base_model_dir)
                LMmodel = None
                #Perform garbage collection to free circular reference objects from memory.
                gc.collect()

            # If model in dir_path, Implement the quantization
            self.config = AutoConfig.from_pretrained(self.args.base_model_dir)

            # Try to apply QLoRA
            self.bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_storage=torch.bfloat16,
            )
            # Quantize the base model
            self.model = AutoModel.from_pretrained(
                self.args.base_model_dir,
                trust_remote_code=True,
                config=self.config,
                quantization_config = self.bnb_config,
                torch_dtype=torch.bfloat16,
                device_map="cuda" if torch.cuda.is_available() else "cpu",
            )
            #load peft model
            self.model.resize_token_embeddings(len(self.tokenizer))
            
            #instantiate a LoraConfig class
            self.peft_config = LoraConfig(target_modules=['embed_tokens',"v_proj", "q_proj", "up_proj", "o_proj", "k_proj", "down_proj", "gate_proj" ], inference_mode=False, r=16, lora_alpha=32, lora_dropout=0.05)         
            
            # QLoRA : Quantized base model + LoRA
            # The get_peft_model function will take the model and prepare it for training with the PEFT method
            self.model = get_peft_model(self.model, self.peft_config)
            print(self.model.print_trainable_parameters())
        
        #load empty classifier
        self.classifier = torch.nn.Linear(self.config.hidden_size, self.args.num_labels, dtype=torch.bfloat16)

    #save the model weights
    def save_pretrained(self, path):
        self.model.save_pretrained(path, save_embedding_layers=True)
        torch.save(self.classifier.state_dict(),os.path.join(path, 'classifier.pt'))
    
    #load the classifier
    def load_classifier(self, path):
        self.classifier.load_state_dict(torch.load(os.path.join(path, 'classifier.pt'), weights_only=True))

    def get_tokenizer(self):
        return self.tokenizer
    
    # write the metadata of model setup
    def print_model_setup(self):
        print(f'Logging model_setup')
        with open(os.path.join(self.args.log_output_dir, f'model_setup.json'), 'w') as f_out:
            json.dump({
                'full_training': self.args.full_training,
                'segment': self.args.segment,
                'mode': self.args.mode,
                'model_name': self.args.model_name,
                'batch_size': self.args.batch_size,
                'max_epoch': self.args.max_epochs,
                'learning_rate': self.args.learning_rate,
                'dropout': self.args.dropout,
                }, f_out, indent=4)

    def step(self):
        self.global_step += 1
    
    def epoch(self, epoch):
            self.current_epoch += 1
            
    def configure_optimizers(self, num_training_steps):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        self.args.num_training_steps = num_training_steps
        print(num_training_steps)
        # Define parameters for no weight decay
        no_decay = ["bias", "LayerNorm.weight"]
        # Group parameters for optimizer
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = bnb.optim.AdamW(optimizer_grouped_parameters, lr=float(self.args.learning_rate), eps=self.args.adam_epsilon)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps= 100,
            num_training_steps=self.args.num_training_steps,
        )
        return

    # Calcutae the logits, loss by classifier and CrossEntropyLoss
    def forward(self, **inputs):
        # load data to device
        ids = inputs["input_ids"].to(DEVICE, dtype = torch.long)
        mask = inputs['input_mask'].to(DEVICE, dtype = torch.long)
        labels = inputs["labels"][inputs["labels"] != -100].to(DEVICE, dtype = torch.long)
        
        #calculate logits
        output = self.model(input_ids=ids, attention_mask=mask) # get model output
        if self.args.segment_type == 'sentence':
            cls_output_state = output["last_hidden_state"][ids == self.tokenizer.cls_token_id]
            cls_output_state = cls_output_state[:len(labels)] #exclude pad tokens for mistral
        if self.args.segment_type == 'token':
            cls_output_state = output['last_hidden_state'][inputs['labels'] != -100]
        cls_output_state = self.dropout(cls_output_state)
        logits = self.classifier(cls_output_state)

        #calculate loss by CrossEntropyLoss
        loss = self.loss_fn(logits, labels,) if "labels" in inputs else None
        return loss, logits, labels,
    
    def train_epoch(self, train_dataloader):
        self.train()
        pbar = tqdm(train_dataloader)
        for _,data in enumerate(pbar, 0):
            outputs = self(**data)
            loss = outputs[0]
            pbar.set_description(f'Loss: {loss}')
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.step()
        return
    
    def evaluate(self, eval_dataloader):
        self.eval()
        output = []
        with torch.no_grad():
            for _, data in enumerate(eval_dataloader, 0):
                outputs = self(**data)
                val_loss, logits, labels = outputs
                preds = torch.argmax(logits, axis=1)
                res = {"loss": val_loss, "preds": preds, "labels": labels, 'logits': logits}
                output.append(res)
        res = self.on_evaluation_epoch_end(output)
        return  res

    def on_evaluation_epoch_end(self, outputs):
        preds = torch.cat([x["preds"] for x in outputs]).detach()
        labels = torch.cat([x["labels"] for x in outputs]).detach()
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        val_res = {}
        val_res[f'loss']= loss.tolist()
        # split the evaluation metrics by mode
        if self.args.mode == 'scopes':

            val_res[f'acc'] = self.metric_multi_acc(preds, labels).tolist()
            val_res[f'macro_f1'] = self.metric_macro_f1(preds, labels).tolist()
            
            binary_labels = torch.tensor([1 if label != 0 else 0 for label in labels])
            binary_preds = torch.tensor([1 if label != 0 else 0 for label in preds])
            val_res[f'total_f1'] = self.metric_f1(binary_preds, binary_labels).tolist()
            
            inf_labels = torch.tensor([1 if label == 1 else 0 for label in labels])
            inf_preds = torch.tensor([1 if label == 1 else 0 for label in preds])
            val_res[f'inf_f1'] = self.metric_f1(inf_preds, inf_labels).tolist()
            
            perc_labels = torch.tensor([1 if label == 2 else 0 for label in labels])
            prec_preds = torch.tensor([1 if label == 2 else 0 for label in preds])
            val_res[f'perc_f1'] = self.metric_f1(prec_preds, perc_labels).tolist()
            
            backg_labels = torch.tensor([1 if label == 3 else 0 for label in labels])
            backg_preds = torch.tensor([1 if label == 3 else 0 for label in preds])
            val_res[f'backg_f1'] = self.metric_f1(backg_preds, backg_labels).tolist()

            early_stopping_metric = val_res[f'macro_f1']
        
        elif self.args.mode == 'total':
            val_res[f'acc'] = self.metric_acc(preds, labels).tolist()
            val_res[f'total_f1'] = self.metric_f1(preds, labels).tolist()

            early_stopping_metric = val_res[f'total_f1']

        print(f'Logging validation scores for epoch {self.current_epoch}')
        with open(os.path.join(self.args.log_output_dir, f'eval_metrics{self.current_epoch}.json'), 'w') as f_out:
            json.dump(val_res, f_out, indent=4)

    
        return early_stopping_metric
    

#region train val test
def train(model, train_dataloader, optimizer, scheduler):
    model.train()
    pbar = tqdm(train_dataloader)
    for _,data in enumerate(pbar, 0):
        loss = model.training_step(data)
        pbar.set_description(f'Loss: {loss}')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        model.step()
    return

def evaluate(model, val_dataloader):
    model.eval()
    output = []
    with torch.no_grad():
        for _, data in enumerate(val_dataloader, 0):
            res = model.validation_step(data)
            output.append(res)
    res = model.on_validation_epoch_end(output)
    return res

#load model and tokenizer
model = SeqTagger(args)
model.load_pretrained(args.model_name)
tokenizer = model.get_tokenizer()

print(model)

#region load data into dataloader and check data
train_data = pd.read_csv(os.path.join(args.data_dir, 'train.csv'))
val_data = pd.read_csv(os.path.join(args.data_dir, 'val.csv'))
test_data = pd.read_csv(os.path.join(args.data_dir, 'test.csv'))
if args.debug:
    train_data = train_data[:args.debug_size]
    val_data = val_data[:args.debug_size]
    test_data = test_data[:args.debug_size]

if args.full_training:
    print(len(train_data), len(val_data))
    train_data = pd.concat([train_data, val_data], axis=0, ignore_index=True)
    print(len(train_data))
    eval_data = test_data
else:
    eval_data = val_data

#load model into dataset
train_dataset = MyDataset(train_data, tokenizer, args)
eval_dataset = MyDataset(eval_data, tokenizer, args)

#Dataloader
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, num_workers=0) 
val_dataloader =  DataLoader(eval_dataset, shuffle=True, batch_size=args.batch_size, num_workers=0)

# double-check data
first_example = train_dataset[0]
sample_text = tokenizer.convert_ids_to_tokens(ids=first_example['input_ids'])
print(f'First example input text: {sample_text}')
if args.segment_type == 'sentence':
    num_pred_targets = len([token for token in first_example['input_ids'] if token == tokenizer.cls_token_id])
elif args.segment_type == 'token':
    num_pred_targets = len([token for token in first_example['input_ids'] if token not in [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]])
print(f'Num pred targets (cls: {tokenizer.cls_token}, sep: {tokenizer.sep_token}, pad: {tokenizer.pad_token}): {num_pred_targets}')
special_token_ids = [token for token in first_example['input_ids'] if token in tokenizer.additional_special_tokens_ids]
print(f'Special tokens in input: {tokenizer.convert_ids_to_tokens(ids=special_token_ids)}')
labels = sorted(Counter([label.item() for label in first_example['labels']]).items())
print(f'Labels {labels}')
#endregion

#region load model
model.print_model_setup()
num_training_steps = len(train_dataloader) * args.max_epochs
model.configure_optimizers(num_training_steps)

model.to(DEVICE)

#endregion
best_value = 0
best_value_epoch = 0
for epoch in tqdm(range(args.max_epochs)):
    model.epoch(epoch)
    model.train_epoch(train_dataloader)
    val_metric = model.evaluate(val_dataloader)
    if val_metric > best_value:
        best_value = val_metric
        best_value_epoch = epoch
        if args.full_training:
            model.save_pretrained(args.model_output_dir)
    else:
        if epoch >= best_value_epoch + args.patients:
            break