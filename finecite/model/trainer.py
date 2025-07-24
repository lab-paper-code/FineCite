# region imports
import os
import torch
import torchmetrics
import json
from torch.nn.modules.loss import CrossEntropyLoss, BCEWithLogitsLoss
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_linear_schedule_with_warmup, BitsAndBytesConfig
from torch.nn import CrossEntropyLoss
from peft import PeftModel, LoraConfig, get_peft_model
import bitsandbytes as bnb
import gc
from math import ceil
import re
from statistics import mean
from finecite.model.utils import CustomDefaultdict
from finecite.model.classifier import ClsClassifier, ExtClassifier, CRF
from finecite.model.model import ExtractionModel, ClassificationModel

class CustomTrainer():
    def __init__(
        self,
        args,
        model,
        tokenizer,
        train_dataloader,
        test_dataloader
    ):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.device = args.device
        self.current_epoch = 0
        self.global_step = 0
        self._extract = False
        
        # add evaluation metrics 
        self.metric_multi_acc = torchmetrics.Accuracy(task='multiclass', num_classes=self.args.num_labels).to(self.device)
        self.metric_acc = torchmetrics.Accuracy(task='binary', num_classes=2).to(self.device)
        self.metric_binary_f1 = torchmetrics.F1Score(task="binary", num_classes=2).to(self.device)
        
        if self.args.dataset == 'multicite':
            self.metric_macro_f1 = torchmetrics.F1Score(average='macro', num_labels=self.args.num_labels, task='multilabel').to(self.device)
            self.metric_micro_f1 = torchmetrics.F1Score(average='micro', num_labels=self.args.num_labels, task='multilabel').to(self.device)
            self.metric_precision = torchmetrics.Precision(average=None, task="multilabel", num_labels=self.args.num_labels).to(self.device)
            self.metric_recall = torchmetrics.Recall(average=None, task="multilabel", num_labels=self.args.num_labels).to(self.device)
            self.metric_f1 = torchmetrics.F1Score(average=None, task="multilabel", num_labels=self.args.num_labels).to(self.device)
        else:
            self.metric_macro_f1 = torchmetrics.F1Score(average='macro', num_classes=self.args.num_labels, task='multiclass').to(self.device)
            self.metric_micro_f1 = torchmetrics.F1Score(average='micro', num_classes=self.args.num_labels, task='multiclass').to(self.device)
            self.metric_precision = torchmetrics.Precision(average=None, task="multiclass", num_classes=self.args.num_labels).to(self.device)
            self.metric_recall = torchmetrics.Recall(average=None, task="multiclass", num_classes=self.args.num_labels).to(self.device)
            self.metric_f1 = torchmetrics.F1Score(average=None, task="multiclass", num_classes=self.args.num_labels).to(self.device)

        # configure optimizser
        print('configuring optimizer...')
        self._configure_optimizer()

    def save_pretrained(self, path):
        torch.save(self.model.state_dict(), path)
            
    def step(self):
        self.global_step += 1
    
    def epoch(self):
            self.current_epoch += 1
            
    def log(self, **kwargs):
        with open(os.path.join(self.args.output_dir, f'train.log'), 'a') as f_out:
            f_out.write(f'{str(kwargs)}\n')

    def _configure_optimizer(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        # Define parameters for no weight decay
        no_decay = ["bias", "LayerNorm.weight"]
        # Group parameters for optimizer
        crf_param = 'ext_loss'
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay) and  crf_param not in n],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay) and  crf_param not in n],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = bnb.optim.AdamW(optimizer_grouped_parameters, lr=float(self.args.learning_rate), eps=self.args.adam_epsilon)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps= 0.05 * self.args.num_training_steps,
            num_training_steps=self.args.num_training_steps,
        )
        if self.args.ext_type in ['crf', 'bilstm_crf']:
            crf_optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if crf_param in n],
                    "weight_decay": self.args.weight_decay,
                }, ]
            self.crf_optimizer = bnb.optim.AdamW(crf_optimizer_grouped_parameters, lr=float(self.args.crf_learning_rate), eps=self.args.adam_epsilon)
            self.crf_scheduler = get_linear_schedule_with_warmup(
                self.crf_optimizer,
                num_warmup_steps= 0.05 * self.args.num_training_steps,
                num_training_steps=self.args.num_training_steps,
            )
    
    def train_epoch(self, train_dataloader):
        agg_loss = []
        self.model.train()
        for _,batch in enumerate(train_dataloader, 0):
            hs, output, loss = self.model(**batch)
            agg_loss.append(loss)
            self.optimizer.zero_grad()
            loss.backward() 
            self.optimizer.step()
            self.scheduler.step()
            if self.args.ext_type in ['crf', 'bilstm_crf']:
                self.crf_optimizer.step()
                self.crf_scheduler.step()
            self.step()
            
            if self.global_step%100 == 0:
                print({'current_epoch':self.current_epoch, 'current_step':self.global_step, 'avg_loss': round(torch.mean(torch.stack(agg_loss)).item(), 5), 'max_loss': round(torch.max(torch.stack(agg_loss)).item(), 5),'min_loss': round(torch.min(torch.stack(agg_loss)).item(), 5), })
                agg_loss = []
                self.log(loss=round(torch.mean(loss).item(), 5), current_step=self.global_step, )
        return
    
    def evaluate(self, eval_dataloader):
        self.model.eval()
        eval_data = CustomDefaultdict(list)
        with torch.no_grad():
            for i, data in enumerate(eval_dataloader, 0):
                hidden_state, output, loss = self.model(**data)
                eval_data.append({'loss': loss})
                eval_data.extend({'input_ids': data['input_ids'].to(self.device), })
                if self.args.task == 'ext':
                    eval_data.extend({'ext_out': output, 'tok_lbl': data['token_labels'][:, 0].to(self.device), 'tok_lbl2': data['token_labels'][:, 1].to(self.device)})
                if self.args.task=='cls': 
                    eval_data.extend({'cls_out': output, 'int_lbl': data['intent_labels'].to(self.device)})    
        eval_data.stack()
        early_stopping_metric, val_res = self.on_evaluation_epoch_end(eval_data)
        samples = eval_data.sample(self.args.batch_size, 'loss')
        output_samples = self.create_output_samples(samples)
        return  early_stopping_metric, val_res, output_samples
    
    def create_output_samples(self, samples):
        output_samples = ''
        if 'ext' in self.args.task:
            output_samples += f"#### Extraction Task ####\n\nMin Loss (first {self.args.batch_size}): {samples['loss'][0]}\nMax Loss (last {self.args.batch_size}): {samples['loss'][1]}\n\n"
            output_samples += self.ext_sample_string(samples)
        if 'cls' in self.args.task:
            output_samples += f"#### Classification Task ####\n\nMin Loss (first {self.args.batch_size}): {samples['loss'][0]}\nMax Loss (last {self.args.batch_size}): {samples['loss'][1]}\n\n"
            for idx, (ids, preds, int_lbl) in enumerate(zip(samples['input_ids'], samples['cls_out'], samples['int_lbl'])):
                output_samples += f'## SAMPLE {idx} ##\n\n'
                output_samples += 'Input:  ' + self.tokenizer.decode(ids) + '\n'
                if self.args.task == 'multicite':
                    preds, int_lbl = preds.tolist(), int_lbl.tolist()
                    output_samples += 'Preds:  '+ [idx for idx in range(self.args.num_labels) if preds[idx] >= 0.5] + '\n'
                    output_samples += 'Labels1:' + [idx for idx in range(self.args.num_labels) if int_lbl[idx] >= 0.5] + '\n\n'
                else:
                    output_samples += 'Preds:  '+ str(torch.argmax(preds, dim=0).item()) + '\n'
                    output_samples += 'Labels1:' + str(torch.argmax(int_lbl, dim=0).item()) + '\n\n'
        return output_samples
    
    def create_label_string(self, labels, token_len_list):
        res_string = ''
        for label, length in zip(labels, token_len_list):
            left = ceil((length-1)/2)
            right = length-1-left
            res_string += ' '*left + str(label) + ' '*right
        return res_string
    
    def evaluate_sample(self, preds, lbl1, lbl2):
        val_res = {}
        f1_lbl1 = self.metric_f1(preds, lbl1).tolist()
        f1_lbl2 = self.metric_f1(preds, lbl2).tolist()
        f1 = list(zip(f1_lbl1, f1_lbl2))
        if self.args.dataset == 'multicite_extraction':
            return f1
        val_res[f'inf_f1'] = round(mean(f1[1]), 3)
        val_res[f'perc_f1'] = round(mean(f1[2]), 3)
        val_res[f'back_f1'] = round(mean(f1[3]), 3)
        
        val_res[f'macro_f1'] = round(mean([val_res[f'inf_f1'],val_res[f'perc_f1'], val_res[f'back_f1']]),3)
        return val_res
    
    def ext_sample_string(self, samples):
        sample_string = '\n'
        for idx, (ids, logits, lbl1, lbl2) in enumerate(zip(samples['input_ids'], samples['ext_out'], samples['tok_lbl'], samples['tok_lbl2'])):
            preds = self.model.decode_logits_to_labels(logits[None,:,:],lbl1[None,:])
            ids, lbl1, lbl2 = ids[lbl1!=-100], lbl1[lbl1!=-100], lbl2[lbl1!=-100]
            val_res = self.evaluate_sample(preds, lbl1, lbl2)
            preds, ids, lbl1, lbl2 = preds.tolist(), ids.tolist(), lbl1.tolist(), lbl2.tolist()
            tokens = self.tokenizer.convert_ids_to_tokens(ids)
            if self.args.model_name == 'scibert':
                tokens = [token[2:] if re.match('##', token) else ' ' + token  for token in tokens]
            elif 'llm2vec' in self.args.model_name:
                tokens = [' ' + token[1:] if re.match('_', token) else token  for token in tokens]
            token_len = [len(token) for token in tokens]
            
            sample_string += f'## SAMPLE {idx} {val_res} ## \n\n'
            sample_string += 'Input:  ' + ''.join(tokens) + '\n'
            sample_string += 'Preds:  '+ self.create_label_string(preds, token_len) + '\n'
            sample_string += 'Labels1:'+ self.create_label_string(lbl1, token_len)+ '\n'
            sample_string += 'Labels2:'+ self.create_label_string(lbl2, token_len)+ '\n\n'
        return sample_string
    

    def on_evaluation_epoch_end(self, eval_data):
        val_res = {}
        val_res['epoch'] = self.current_epoch
        if 'ext' in self.args.task and self.args.dataset == 'finecite':
            val_res['loss'] = round(torch.mean(eval_data['loss'], dim=0).item(), 3)
            preds = self.model.decode_logits_to_labels(eval_data['ext_out'], eval_data['tok_lbl'])
            preds = torch.tensor([l if l <=3 else l-3 for l in preds]).to(self.device)
            
            labels1, labels2 = eval_data['tok_lbl'][eval_data['tok_lbl'] != -100], eval_data['tok_lbl2'][eval_data['tok_lbl'] != -100]
            labels1, labels2 = torch.tensor([l if l <=3 else l-3 for l in labels1]).to(self.device), torch.tensor([l if l <=3 else l-3 for l in labels2]).to(self.device)

            val_res[f'acc'] = [round(self.metric_multi_acc(preds, labels1).item(), 3), round(self.metric_multi_acc(preds, labels2).item(), 3)]
            val_res[f'macro_f1'] = []         
            binary_labels1 = torch.where(labels1 == 0, 0, 1)
            binary_labels2 = torch.where(labels2 == 0, 0, 1)
            binary_preds = torch.where(preds == 0, 0, 1)
            val_res[f'total_f1'] = mean([round(self.metric_binary_f1(binary_preds, binary_labels1).item(), 3), round(self.metric_binary_f1(binary_preds, binary_labels2).item(), 3)])
            
            f1_lbl1 = self.metric_f1(preds, labels1).tolist()
            f1_lbl2 = self.metric_f1(preds, labels2).tolist()
            f1 = list(zip(f1_lbl1, f1_lbl2))
            val_res[f'inf_f1'] = round(mean(f1[1]), 3)
            val_res[f'perc_f1'] = round(mean(f1[2]), 3)
            val_res[f'back_f1'] = round(mean(f1[3]), 3)
            
            val_res[f'macro_f1'] = round(mean([val_res[f'inf_f1'],val_res[f'perc_f1'], val_res[f'back_f1']]),3)
            
        if 'ext' in self.args.task and self.args.dataset == 'multicite_extraction':
            val_res['loss'] = round(torch.mean(eval_data['loss'], dim=0).item(), 3)
            preds = self.model.decode_logits_to_labels(eval_data['ext_out'], eval_data['tok_lbl'])
            preds = torch.tensor([l if l <=1 else l-1 for l in preds]).to(self.device)
            
            labels1 = eval_data['tok_lbl'][eval_data['tok_lbl'] != -100]
            labels1 = torch.tensor([l if l <=1 else l-1 for l in labels1]).to(self.device)
            
            val_res['macro_f1'] = round(self.metric_binary_f1(preds, labels1).item(), 3)
            
            
        if 'cls' in self.args.task:
            val_res['loss'] = round(torch.mean(eval_data['loss'], dim=0).item(), 3)
            if self.args.dataset == 'multicite':
                preds_04 = eval_data['cls_out'] >= 0.4
                preds_05 = eval_data['cls_out'] >= 0.5
                preds_06 = eval_data['cls_out'] >= 0.6
                preds_07 = eval_data['cls_out'] >= 0.7
                labels = eval_data['int_lbl']
                val_res['macro_f1'] = round(self.metric_macro_f1(preds_05, labels).item(), 3)
                val_res['macro_f1_06'] = round(self.metric_macro_f1(preds_06, labels).item(), 3)
                val_res['macro_f1_07'] = round(self.metric_macro_f1(preds_07, labels).item(), 3)
                val_res['macro_f1_04'] = round(self.metric_macro_f1(preds_04, labels).item(), 3)
                
                val_res['precision_05'] = [round(x, 3) for x in self.metric_precision(preds_05, labels).tolist()]
                val_res['recall_05'] = [round(x, 3) for x in self.metric_recall(preds_05, labels).tolist()]
                val_res['f1_05'] = [round(x, 3) for x in self.metric_f1(preds_05, labels).tolist()]
                
                val_res['precision_04'] = [round(x, 3) for x in self.metric_precision(preds_04, labels).tolist()]
                val_res['recall_04'] = [round(x, 3) for x in self.metric_recall(preds_04, labels).tolist()]
                val_res['f1_04'] = [round(x, 3) for x in self.metric_f1(preds_04, labels).tolist()]
                
            else:
                preds = torch.argmax(eval_data['cls_out'], dim=1)
                labels = torch.argmax(eval_data['int_lbl'], dim=1)
                val_res['macro_f1'] = round(self.metric_macro_f1(preds, labels).item(), 3)
                val_res['micro_f1'] = round(self.metric_micro_f1(preds, labels).item(), 3)
                
                val_res['precision'] = [round(x, 3) for x in self.metric_precision(preds, labels).tolist()]
                val_res['recall'] = [round(x, 3) for x in self.metric_recall(preds, labels).tolist()]
                val_res['f1'] = [round(x, 3) for x in self.metric_f1(preds, labels).tolist()]
            

        early_stopping_metric = val_res['macro_f1']

        print(val_res)
        self.log(**val_res)
        return early_stopping_metric, val_res
    
    
    def train(self):
        print('\nstarting training...')
        self.args.best_value = 0
        self.args.best_value_epoch = 0
        best_val_res = {}
        best_output_samples = ''

        for epoch in range(self.args.max_epochs):
            self.epoch()
            self.train_epoch(self.train_dataloader)
            val_metric, val_res, output_samples = self.evaluate(self.test_dataloader)
            if val_metric > self.args.best_value:
                self.args.best_value = val_metric
                self.args.best_value_epoch = epoch
                best_val_res = val_res
                best_output_samples = output_samples
                if self.args.save_model:
                    self.save_pretrained(self.args.model_output_file)
            else:
                if epoch >= self.args.best_value_epoch + self.args.patients:
                    break
                
        print(f'Logging validation scores for best epoch')
        with open(os.path.join(self.args.output_dir, f'best_scores.json'), 'w') as f_out:
                json.dump(best_val_res, f_out, indent=4)
                
        print(f'Logging best output samples')
        with open(os.path.join(self.args.output_dir, f'output_samples.txt'), 'w') as f_out:
                f_out.write(best_output_samples)