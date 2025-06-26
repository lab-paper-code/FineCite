# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: lasse
#     language: python
#     name: python3
# ---
'''
CUDA_VISIBLE_DEVICES=1 nohup python3 -u train.py --model_name llm2vec_llama3 --dataset finecite --task ext --ext_type linear --debug> output/linear.txt &
CUDA_VISIBLE_DEVICES=2 nohup python3 -u train.py --model_name scibert --dataset finecite --task ext --ext_type bilstm --save_model> output/bilstm.txt &
CUDA_VISIBLE_DEVICES=2 nohup python3 -u train.py --model_name scibert --dataset finecite --task ext --ext_type crf --save_model> output/crf.txt &
CUDA_VISIBLE_DEVICES=2 nohup python3 -u train.py --model_name scibert --dataset finecite --task ext --ext_type bilstm_crf --save_model> output/bilstm_crf.txt &

CUDA_VISIBLE_DEVICES=2 nohup bash -c 'for ext_type in linear bilstm crf
    do python3 -u train.py --model_name llm2vec_mistral --dataset finecite --task ext --ext_type ${ext_type} --save_model --batch_size 3;
done'> output/llm2vec_mistral.txt &

CUDA_VISIBLE_DEVICES=2 nohup bash -c 'for seed in 10 11 12 13 14
        do python3 -u train.py --model_name scibert --dataset finecite --task ext --ext_type linear --save_model --batch_size 4 --learning_rate 5e-05 --seed ${seed} --train_size -1;
done'> output/sample_ablation.txt &

CUDA_VISIBLE_DEVICES=0 nohup python3 -u train.py --model_name scibert --dataset multicite_extraction --task ext --ext_type linear --save_model --batch_size 4 --learning_rate 5e-05 --seed 10 > output/extraction_ablation.txt &

Hyper Parameter:
acl-arc: --batch_size 4 --learning_rate 5e-05
act2: --batch_size 16 --learning_rate 3e-05
sciicte: --batch_size 16 --learning_rate 3e-05
multicite: --batch_size 8 --learning_rate 5e-05

CUDA_VISIBLE_DEVICES=2 nohup python3 -u train.py --model_name scibert --dataset ACT-ND --task cls --ext_type linear > output/ACT-ND.txt &
CUDA_VISIBLE_DEVICES=2 nohup python3 -u train.py --model_name scibert --dataset act2 --task cls --ext_type bilstm > output/bilstm.txt &
CUDA_VISIBLE_DEVICES=2 nohup python3 -u train.py --model_name scibert --dataset act2 --task cls --ext_type crf > output/crf.txt &
CUDA_VISIBLE_DEVICES=2 nohup python3 -u train.py --model_name scibert --dataset act2 --task cls --ext_type bilstm_crf > output/bilstm_crf.txt &

CUDA_VISIBLE_DEVICES=3 nohup python3 -u train.py --model_name scibert --dataset act2 --task cls --ext_type linear --heal_token word > output/linear_word.txt &
CUDA_VISIBLE_DEVICES=3 nohup python3 -u train.py --model_name scibert --dataset act2 --task cls --ext_type linear --heal_token phrase > output/linear_phrase.txt &

CUDA_VISIBLE_DEVICES=1 nohup bash -c 'for seed in 10 11 12 13 14
    do for ext_type in linear bilstm crf
        do python3 -u train.py --model_name scibert --dataset multicite --task cls --ext_type ${ext_type} --seed ${seed}  --batch_size 16 --learning_rate 3e-05;
    done;
done'> output/scibert_multicite.txt & 

CUDA_VISIBLE_DEVICES=3 nohup bash -c 'for seed in 10 11   
    do for cls_type in linear
        do python3 -u train.py --model_name scibert --dataset acl-arc --task cls --ext_type linear --cls_type ${cls_type} --seed ${seed} --batch_size 4 --learning_rate 5e-05;
    done;
done'> output/scibert_acl_arc_ablation.txt & 

CUDA_VISIBLE_DEVICES=2 nohup bash -c 'for seed in 10 11 12 13 14   
    do for dataset in ACT-D ACT-ND
        do python3 -u train.py --model_name scibert --dataset ${dataset} --task cls --ext_type linear --cls_type linear --seed ${seed} --batch_size 16 --learning_rate 3e-05;
    done;
done'> output/scibert_act2_domain_ablation.txt & 
'''

import os
#os.environ['CUDA_VISIBLE_DEVICES']='2'
import json
from tqdm import tqdm
from collections import Counter
import torch
from torch.utils.data import DataLoader
import pandas as pd
from datetime import datetime
import argparse
import random
import numpy as np
import importlib
import getpass
import sys

sys.path.append('/home/dacslab/lassejantsch/CCE')

import finecite
from finecite import set_seed, DATA_DIR, get_class_weights, CustomDataset, load_processor
from finecite.model import MODEL_DESCRIPTION
from finecite.model.finecite_linear import CCAModel
#fix sample extraction (only possible to extract one)

def main():
    
    parser = argparse.ArgumentParser(description='Seq_tagger parser')

    #input arguments
    parser.add_argument('--model_name', required=True, help='scibert llm2vec_mistral llm2vec_llama3')
    parser.add_argument('--dataset', required=True, help='acl-arc, act2, scicite, multicite, finecite')
    parser.add_argument('--task', required=True, help='ext cls')
    parser.add_argument('--ext_type', required=True, help='linear, bilstm, crf, bilstm_crf')
    
    parser.add_argument('--heal_token', default='word')
    parser.add_argument('--cls_type', default='linear', help='weighted, balanced, linear, auto_wighted, inf, perc, back')
    parser.add_argument('--cls_weights', type=list[int], default=[1, 1, 1])
    

    parser.add_argument('--batch_size', type=int, default=4, help='')
    parser.add_argument('--learning_rate', type=float, default=2e-05, help='')
    parser.add_argument('--dropout', type=float, default=0.1, help='')

    parser.add_argument('--reext_data', action='store_true')
    parser.add_argument('--save_model', action='store_true', help='')
    parser.add_argument('--debug', action='store_true', help='')
    parser.add_argument('--debug_size', type=int, default=None, help='')
    parser.add_argument('--train_size', type=int, default=None, help='')
    parser.add_argument('--print_test', action='store_true')
    parser.add_argument("--seed", type=int, default=4455, help='')
    args = parser.parse_args()
    
    #static arguments
    args.max_epochs = 20
    args.patients = 5
    args.adam_epsilon = 1e-08
    args.weight_decay = 0.0
    args.use_prompt = False #args.model_name != 'scibert'
    args.max_len = 512 if args.model_name == 'scibert' else 740
    args.dtype = torch.float32
    
    #temp
    args.crf_learning_rate = 0.001

    # model description
    args.model_desc = MODEL_DESCRIPTION[args.model_name]

    # input directory
    args.input_dir = f'{DATA_DIR}/model_training/{args.dataset}/'

    # output directory
    if args.debug:
        args.output_dir = f"./output/_debug/{args.dataset}/{args.model_name}/{args.batch_size}_{args.learning_rate}_{args.dropout}_{datetime.now().strftime('%m_%d_%H_%M_%S')}/"
    elif args.cls_type != 'linear':
        args.output_dir = f"./output/{args.dataset}/{args.model_name}/{args.ext_type}_{args.cls_type}__seed_{args.seed}__bs_{args.batch_size}__lr_{args.learning_rate}__do_{args.dropout}__time_{datetime.now().strftime('%m_%d_%H_%M_%S')}/"   
    elif args.train_size:
        args.output_dir = f"./output/{args.dataset}/{args.model_name}/sample_ablation/{args.ext_type}__size_{args.train_size}__seed_{args.seed}__bs_{args.batch_size}__lr_{args.learning_rate}__do_{args.dropout}__time_{datetime.now().strftime('%m_%d_%H_%M_%S')}/"   
    elif args.print_test:
        if not args.train_size:
            args.trin_size = -1
        args.output_dir = f"./output/{args.dataset}/{args.model_name}/print_test/{args.ext_type}__size_{args.train_size}__seed_{args.seed}__bs_{args.batch_size}__lr_{args.learning_rate}__do_{args.dropout}__time_{datetime.now().strftime('%m_%d_%H_%M_%S')}/" 
    else: 
        args.output_dir = f"./output/{args.dataset}/{args.model_name}/{args.ext_type}__seed_{args.seed}__bs_{args.batch_size}__lr_{args.learning_rate}__do_{args.dropout}__time_{datetime.now().strftime('%m_%d_%H_%M_%S')}/"   
        
    os.makedirs(args.output_dir, exist_ok=True)
    if args.save_model:
        args.model_output_dir = f"{DATA_DIR}/model_training/output/{args.dataset}/{args.model_name}/{args.ext_type}__seed_{args.seed}__bs_{args.batch_size}__lr_{args.learning_rate}__do_{args.dropout}__time_{datetime.now().strftime('%m_%d_%H_%M_%S')}/"
        os.makedirs(args.model_output_dir, exist_ok=True)

    # model cache dir
    if args.model_name != 'scibert':
        args.base_model_dir =  f'{DATA_DIR}/model_training/llm2vec_models/{args.model_name}/'
        os.makedirs(args.base_model_dir, exist_ok=True)
    
    #data cache dir
    if args.task == 'cls':
        args.chache_dir =  f'.cache/{args.dataset}/'
        os.makedirs(args.chache_dir, exist_ok=True)

    # model input dir
    args.trained_model_dir = f'{DATA_DIR}/model_training/output/finecite/{args.model_name}/{args.ext_type}' if args.dataset != 'finecite' else None

    set_seed(args.seed)
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #load model and tokenizer
    print('loading model...')
    model = CCAModel(args)
    model.load_pretrained()
    tokenizer = model.tokenizer
    print('done')
    
    #load data processor
    processor = load_processor(args)
    dataset = CustomDataset(args, tokenizer)

    # load data
    train_data = processor.load_train_data()
    test_data = processor.load_test_data()

    # create dataset
    train_ds, weights1, num_labels = dataset.load_data(train_data)
    test_ds, weights2, num_labels = dataset.load_data(test_data)
    weights = [(w1 * len(train_ds) + w2 * len(test_ds)) / (len(train_ds) + len(test_ds)) for w1, w2 in zip(weights1, weights2)]
    num_training_steps = int(len(train_data) / args.batch_size) * args.max_epochs
    print(num_labels)
    model.configurate(weights, num_labels, num_training_steps)

        # add context labels if cls
    if args.task == 'cls':
        train_file = f'{args.model_name}_{args.ext_type}_{args.heal_token}_train.pt'
        test_file = f'{args.model_name}_{args.ext_type}_{args.heal_token}_test.pt'
        # check cached examples
        if not args.reext_data and train_file in os.listdir(args.chache_dir) and test_file in os.listdir(args.chache_dir):
            train_ds = torch.load(os.path.join(args.chache_dir, train_file), weights_only=False)
            test_ds = torch.load(os.path.join(args.chache_dir, test_file), weights_only=False)
        else:
        
            # dataloader
            train_dl = DataLoader(train_ds, batch_size = args.batch_size, shuffle=False)
            test_dl = DataLoader(test_ds, batch_size = args.batch_size, shuffle=False)
            
            #extract context
            train_lbls = model.extract(train_dl)
            test_lbls = model.extract(test_dl)
            
            #add context to dataset
            train_ds = dataset.add_context_lbls(train_ds, train_lbls)
            test_ds = dataset.add_context_lbls(test_ds, test_lbls)
            
            #cache data
            torch.save(train_ds, os.path.join(args.chache_dir, train_file))
            torch.save(test_ds, os.path.join(args.chache_dir, test_file))
            
        #reload model from_pretrained
        model.reload()

    print(len(train_ds), len(test_ds))

    #Dataloader
    train_dataloader = DataLoader(train_ds, shuffle=True, batch_size=args.batch_size, num_workers=0) 
    val_dataloader =  DataLoader(test_ds, shuffle=False, batch_size=args.batch_size, num_workers=0)
        
    # log model setup
    print(f'Logging run_setup')
    print_args = {k: str(v) for k,v in vars(args).items()}
    with open(os.path.join(args.output_dir, f'run_setup.json'), 'w') as f_out:
        json.dump(print_args, f_out, indent=4)

    # log imput sample
    print(f'Logging input sample')
    input_sample = [tokenizer.convert_ids_to_tokens(ids=train_ds[i]['input_ids']) for i in range(3)]

    with open(os.path.join(args.output_dir, f'input_sample.json'), 'w') as f_out:
        json.dump(input_sample, f_out, indent=4)
        
    # print sample text
    first_example = train_ds[0]
    sample_text = tokenizer.convert_ids_to_tokens(ids=first_example['input_ids'])
    print(f'First example input text: {sample_text}')
    #print number of predicting targets
    num_pred_targets = len([token for token in first_example['input_ids'] if token not in [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]])
    print(f'Num pred targets (cls: {tokenizer.cls_token}, sep: {tokenizer.sep_token}, pad: {tokenizer.pad_token}): {num_pred_targets}')
    #print special token in example
    special_token_ids = [token for token in first_example['input_ids'] if token in tokenizer.additional_special_tokens_ids]
    print(f'Special tokens in input: {tokenizer.convert_ids_to_tokens(ids=special_token_ids)}')
    #print labels in example
    if 'ext' in args.task:
        labels = first_example['tok_lbl'].tolist()
    if 'cls' in args.task:
        labels = first_example['int_lbl'].tolist()
    print(f'Labels {labels}')
    
    # start training
    print('\nstarting training...')
    model.to(args.device)
    args.best_value = 0
    args.best_value_epoch = 0
    best_val_res = {}
    best_output_samples = ''
    for epoch in range(args.max_epochs):
        model.epoch()
        model.train_epoch(train_dataloader)
        val_metric, val_res, output_samples = model.evaluate(val_dataloader)
        if val_metric > args.best_value:
            args.best_value = val_metric
            args.best_value_epoch = epoch
            best_val_res = val_res
            best_output_samples = output_samples
            if args.save_model:
                model.save_pretrained(args.model_output_dir)
        else:
            if epoch >= args.best_value_epoch + args.patients:
                break
            
    print(f'Logging validation scores for best epoch')
    with open(os.path.join(args.output_dir, f'best_scores.json'), 'w') as f_out:
            json.dump(best_val_res, f_out, indent=4)
            
    print(f'Logging best output samples')
    with open(os.path.join(args.output_dir, f'output_samples.txt'), 'w') as f_out:
            f_out.write(best_output_samples)

if __name__ == "__main__":
    main()