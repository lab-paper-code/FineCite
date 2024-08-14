import pandas as pd
import torch
import transformers
from torch.utils.data import Dataset, DataLoader
from torch import cuda
import sys
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from ast import literal_eval
from tqdm import tqdm
import random
import numpy as np

'''
python3 classification.py acl_arc fc_t 

data_set='acl_arc'
for context in t bt tn btn bbt tnn
do
    CUDA_VISIBLE_DEVICES=2,3  python3 classification.py acl_arc fc ${context} 
done

data_set='acl_arc'
for context in sentence_prio_scopes sentence_prio_total sentence_majo_scopes sentence_majo_total token_scibert_scopes token_scibert_total
do
    for type in d_nc d_c
    do
        CUDA_VISIBLE_DEVICES=2,3  python3 classification.py acl_arc ${type} ${context} 
    done
done

for segment in sentence_prio sentence_majo
do
    for mode in scopes total
    do
        CUDA_VISIBLE_DEVICES=2,3 python3 classification.py ${segment} ${mode} 
    done
done

'''

# load model and tokenizer
LMTokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
LMModel = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET = sys.argv[1]
TYPE = sys.argv[2]
CONTEXT = sys.argv[3]

#load inpput dataset
if TYPE == 'fc':
    train_dataset = pd.read_csv(f'/raid/deallab/CCE_Data/model_evaluation/classification_task/data/{DATASET}/{TYPE}/{CONTEXT}/train.csv', sep=',')
    testing_dataset = pd.read_csv(f'/raid/deallab/CCE_Data/model_evaluation/classification_task/data/{DATASET}/{TYPE}/{CONTEXT}/test.csv', sep=',')
elif TYPE in ['d_nc', 'd_c']:
    train_dataset = pd.read_csv(f'/raid/deallab/CCE_Data/model_evaluation/classification_task/data/{DATASET}/finecite/{TYPE}/{CONTEXT}/train.csv', sep=',')
    testing_dataset = pd.read_csv(f'/raid/deallab/CCE_Data/model_evaluation/classification_task/data/{DATASET}/finecite/{TYPE}/{CONTEXT}/test.csv', sep=',')


#Model parameter
MAX_LEN = 512
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 8
LEARNING_RATE = 1e-5
drop_out = 0.4
EPOCHS = 10
tokenizer = LMTokenizer
CLS_COUNT = 8 if sys.argv[1] == 'multicite' else 6

output_file_name = f'results/{DATASET}/{TYPE}_{CONTEXT}_{TRAIN_BATCH_SIZE}_{LEARNING_RATE}_{drop_out}.txt'
file = open(output_file_name,'w')

def set_seed(seed_value=99):
    # Set seed for reproducibility.
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
set_seed()


class Triage(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __getitem__(self, index):
        
        CC = literal_eval(self.data.CC[index])
        CC = ' [SEP] '.join(CC)
        CC = " ".join(CC.split())
        inputs = self.tokenizer.encode_plus(
            CC,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )
        CC_ids = inputs['input_ids']
        CC_mask = inputs['attention_mask']

        return {       
            'CC_ids': torch.tensor(CC_ids, dtype=torch.long),
            'CC_mask': torch.tensor(CC_mask, dtype=torch.long),
            
            'targets': torch.tensor(self.data.label[index], dtype=torch.long)
        } 
    
    def __len__(self):
        return self.len


training_set = Triage(train_dataset, tokenizer, MAX_LEN)
testing_set = Triage(testing_dataset, tokenizer, MAX_LEN)

train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

test_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)

class LMClass(torch.nn.Module):
    def __init__(self):
        super(LMClass, self).__init__()
        self.l1 = torch.nn.DataParallel(LMModel)
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(drop_out)
        self.classifier = torch.nn.Linear(768, CLS_COUNT)

    def forward(self, data):
        input_ids = data['CC_ids'].to(device, dtype = torch.long)
        attention_mask = data['CC_mask'].to(device, dtype = torch.long)
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state1 = output_1[0]
        pooler = hidden_state1[:, 0, :]
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output

model = LMClass()
model.to(device)

#class weights of the different datasets
if DATASET == 'sdp_act':
    weights = [0.30435841, 1.34843581, 2.91375291, 7.57575758, 1.78062678, 1.06837607] #sdp_act
elif DATASET == 'acl_arc':
    weights = [0.32256169, 0.92424242, 4.65254237, 4.81578947, 3.8125, 0.88263666]  # acl_arc
elif DATASET == 'multicite':
    weights = [0.39627130681818185, 2.27178338762215, 17.88301282051282, 1.1386734693877552, 0.9527834699453552, 0.42179467795585124, 2.15258487654321, 4.603547854785479]
else: 
    sys.exit('unknown dataset')
    
class_weights = torch.FloatTensor(weights).to(device)
loss_function = torch.nn.CrossEntropyLoss(weight=class_weights)

# load optimizer
no_decay = ['bias', 'gamma', 'beta']
optimizer_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        'weight_decay_rate': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        'weight_decay_rate': 0.0}
]
optimizer = torch.optim.AdamW(optimizer_parameters, lr=LEARNING_RATE, eps=1e-8)


def calcuate_accu(big_idx, targets):
    n_correct = (big_idx==targets).sum().item()
    return n_correct

def train(epoch):
    tr_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    model.train()
    pred = []
    act = []
    for _,data in enumerate(tqdm(training_loader), 0):
        targets = data['targets'].to(device, dtype = torch.long)

        outputs = model(data)
        loss = loss_function(outputs, targets)
        tr_loss += loss.item()
        big_val, big_idx = torch.max(outputs.data, dim=1)
        n_correct += calcuate_accu(big_idx, targets)
        pred += big_idx.tolist()
        act += targets.tolist()

        nb_tr_steps += 1
        nb_tr_examples+=targets.size(0)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()

    file.write(f'The Total Accuracy for Epoch {epoch}: {(n_correct*100)/nb_tr_examples}\n')
    print(f'The Total Accuracy for Epoch {epoch}: {(n_correct*100)/nb_tr_examples}\n')
    epoch_loss = tr_loss/nb_tr_steps
    epoch_accu = (n_correct*100)/nb_tr_examples
    file.write(f"Training Loss Epoch: {epoch_loss}\n")
    file.write(f"Training Accuracy Epoch: {epoch_accu}\n")
    ma_f1 = f1_score(act, pred, average='macro')
    mi_f1 = f1_score(act, pred, average='micro')
    file.write(f"Train Macro F1: {ma_f1}\n")
    file.write(f"Train Micro F1: {mi_f1}\n")
    file.write("\n")
    return

def valid(model, testing_loader):
    model.eval()
    n_correct = 0; n_wrong = 0; tr_loss = 0
    nb_tr_steps =0
    nb_tr_examples =0
    pred = []
    act = []
    val_targets = []
    val_outputs = []
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            targets = data['targets'].to(device, dtype = torch.long)
            val_targets.extend(targets.tolist())
            outputs = model(data)
            val_outputs.extend(outputs.argmax(-1).tolist())
            loss = loss_function(outputs, targets)
            tr_loss += loss.item()
            big_val, big_idx = torch.max(outputs.data, dim=1)
            n_correct += calcuate_accu(big_idx, targets)
            pred += big_idx.tolist()
            act += targets.tolist()
            nb_tr_steps += 1
            nb_tr_examples+=targets.size(0)
    
    epoch_loss = tr_loss/nb_tr_steps
    epoch_accu = (n_correct*100)/nb_tr_examples
    file.write(f"Validation Loss Epoch: {epoch_loss}\n")
    file.write(f"Validation Accuracy Epoch: {epoch_accu}\n")
    ma_f1 = f1_score(act, pred, average='macro')
    mi_f1 = f1_score(act, pred, average='micro')
    file.write(f"Validation Macro F1: {ma_f1}\n")
    file.write(f"Validation Micro F1: {mi_f1}\n")
    i_f1_string = 'Class F1:'
    for i in range(6):
        i_target = [1 if t == i else 0 for t in val_targets]
        i_preds = [1 if p == i else 0 for p in val_outputs]
        i_f1 = f1_score(i_target, i_preds)
        i_f1_string += f' {i}:{i_f1},'
    file.write(i_f1_string + '\n')
    print(f'Validation Macro: {ma_f1}   Validation Micro: {mi_f1}')
    return ma_f1, mi_f1, epoch_accu,

best_ma_f1 = {
    'epoch': 0,
    'score': 0
}
best_mi_f1 = {
    'epoch': 0,
    'score': 0
}
best_acc = {
    'epoch': 0,
    'score': 0
}

for epoch in tqdm(range(EPOCHS)):
    train(epoch)
    ma_f1, mi_f1 ,acc = valid(model, testing_loader)
    if ma_f1 > best_ma_f1['score']:
        best_ma_f1['score'] = ma_f1
        best_ma_f1['epoch'] = epoch
    if mi_f1 > best_mi_f1['score']:
        best_mi_f1['score'] = mi_f1
        best_mi_f1['epoch'] = epoch
    if acc > best_acc['score']:
        best_acc['score'] = acc
        best_acc['epoch'] = epoch


    file.write("\n\n")
    file.close()
    file = open(output_file_name,'a')

file.write(f'Best:\n    Accuracy in Epoch {best_acc["epoch"]}: {best_acc["score"]}\n    Macro F1 in Epoch {best_ma_f1["epoch"]}: {best_ma_f1["score"]}\n    Micro F1 in Epoch {best_mi_f1["epoch"]}: {best_mi_f1["score"]}')
file.close()