
import os
os.environ['CUDA_VISIBLE_DEVICES']='2'
import json
import re
from datetime import datetime
import sys
import argparse

import torch
from torch.utils.data import DataLoader
from dotenv import load_dotenv

load_dotenv()
DATA_DIR = os.getenv('DATA_DIR')
CACHE_DIR = os.getenv('CACHE_DIR')
OUT_DIR = os.getenv('OUT_DIR')
FINECITE_PATH = os.getenv('FINECITE_PATH')
if FINECITE_PATH not in sys.path:
    sys.path.append(FINECITE_PATH)

from finecite.utils import set_seed, get_class_weights
from finecite.data_processing import load_processor
from finecite.model import CustomTrainer, ExtractionModel, ClassificationModel, load_classifier, load_tokenizer_embedding_model, MODEL_DESCRIPTION

import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Seq_tagger parser')

    parser.add_argument('--model_name', type=str, default='scibert',
                        help='Model used for tagging: scibert, llm2vec_mistral, llm2vec_llama3')

    parser.add_argument('--ext_model', type=str, default='scibert',
                        help='Model used for external features: scibert, llm2vec_mistral, llm2vec_llama3')

    parser.add_argument('--dataset', type=str, default='acl-arc',
                        help="Dataset to use: 'acl-arc', 'act2', 'scicite', 'multicite'")

    parser.add_argument('--ext_type', type=str, default='bilstm_crf',
                        help="Extension model type: 'linear', 'bilstm', 'crf', 'bilstm_crf'")

    parser.add_argument('--cls_type', type=str, default='linear',
                        help="Classification strategy: 'weighted', 'balanced', 'linear', 'inf', 'perc', 'back'")

    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training')

    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='Learning rate for the base model')

    parser.add_argument('--crf_learning_rate', type=float, default=0.005,
                        help='Learning rate for the CRF layer (if used)')

    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate applied to the model')

    parser.add_argument('--save_model', action='store_true',
                        help='If set, the model will be saved after training')

    parser.add_argument('--cached_data', action='store_true',
                        help='Use cached preprocessed data if available')

    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode (loads a small subset of data)')

    parser.add_argument('--debug_size', type=int, default=100,
                        help='Number of examples to use in debug mode')

    parser.add_argument('--seed', type=int, default=4455,
                        help='Random seed for reproducibility')

    args =  parser.parse_args()
    
    args.task = 'cls'
    args.max_epochs = 20
    args.patients = 5
    args.adam_epsilon = 1e-08
    args.weight_decay = 0.0
    args.use_prompt = 'llm2vec' in args.model_name
    args.dtype = torch.float32

    # model description
    args.model_desc = MODEL_DESCRIPTION[args.model_name]
    args.max_len = args.model_desc['max_len']

    # input directory
    args.input_dir = f'{DATA_DIR}/{args.dataset}/'

    # output directory
    args.output_dir = re.sub(r'[.:*?"<>|\s-]','_',(
        f"{OUT_DIR}/"
        f"{'_debug/' if args.debug else ''}"
        f"{args.dataset}/{args.model_name}/"
        f"{args.ext_type}__{args.batch_size}__{args.learning_rate}__{args.dropout}"
        f"{'__' + datetime.now().strftime('%m_%d_%H_%M_%S') if args.debug else ''}/"
    ))
    args.model_output_file = re.sub(r'[.:*?"<>|\s]','_',(
        f"{OUT_DIR}/"
        f"{'_debug/' if args.debug else ''}"
        f"{args.dataset}/{args.model_name}/"
        f"safetensors__{args.batch_size}__{args.learning_rate}__{args.dropout}.pt"
    ))

    os.makedirs(args.output_dir, exist_ok=True)

    # model cache dir
    if 'llm2vec' in args.model_name:
        args.base_model_dir =  f'{CACHE_DIR}/{args.model_name}/'
        os.makedirs(args.base_model_dir, exist_ok=True)

    # data cache dir
    args.data_cache_dir =  f'{CACHE_DIR}/{args.dataset}/'
    os.makedirs(args.data_cache_dir, exist_ok=True)

    # ext model dir
    args.ext_model_dir = re.sub(r'[.:*?"<>|\s-]','_',(
        f"{OUT_DIR}/finecite/{args.ext_model}/{args.ext_type}"
    ))
    set_seed(args.seed)
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    return args

def main():
    args = get_args()
    
    with open(f'{args.ext_model_dir}/run_setup.json', 'r') as f:
        ext_args =  argparse.Namespace(**json.load(f))
        
    ext_args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ext_args.dtype = args.dtype

    print('loading extraction model...')
    tokenizer, embedding_model = load_tokenizer_embedding_model(ext_args)
    ext_args.cls_token_id = tokenizer.cls_token_id
    ext_args.sep_token_id = tokenizer.sep_token_id
    ext_args.pad_token_id = tokenizer.pad_token_id

    ext_model = ExtractionModel(ext_args, embedding_model)

    ext_model.load_state_dict(torch.load(f'{args.ext_model_dir}/safetensors.pt'))
    #load data processor
    processor = load_processor(args, tokenizer)

    # load data
    train_data = processor.read_data('train')
    test_data = processor.read_data('test')

    # create dataset
    train_ds, weights, num_labels = processor.create_features(train_data)
    test_ds, _ , _ = processor.create_features(test_data)
    args.label_weights = weights
    args.num_labels = num_labels
    args.num_training_steps = int(len(train_data) / args.batch_size) * args.max_epochs

    train_file = f'{args.ext_model}_{args.ext_type}_train.pt'
    test_file = f'{args.ext_model}_{args.ext_type}_test.pt'
    # check cached examples
    if args.cached_data and train_file in os.listdir(args.data_cache_dir) and test_file in os.listdir(args.data_cache_dir):
        train_ds = torch.load(os.path.join(args.data_cache_dir, train_file), weights_only=False)
        test_ds = torch.load(os.path.join(args.data_cache_dir, test_file), weights_only=False)
    else:

        # dataloader
        train_dl = DataLoader(train_ds, batch_size = args.batch_size, shuffle=False)
        test_dl = DataLoader(test_ds, batch_size = args.batch_size, shuffle=False)
        
        #extract context
        train_lbls = ext_model.extract_token_labels(train_dl)
        test_lbls = ext_model.extract_token_labels(test_dl)
        
        assert len(train_ds) == len(train_lbls) and len(test_ds) == len(test_lbls) 
        
        #add context to dataset
        train_ds = [{**example, 'token_labels': torch.tensor(tok_lbl, dtype=torch.long)} for example, tok_lbl in zip(train_ds, train_lbls)]
        test_ds = [{**example, 'token_labels': torch.tensor(tok_lbl, dtype=torch.long)}for example, tok_lbl in zip(test_ds, test_lbls)]
        
        #cache data
        torch.save(train_ds, os.path.join(args.data_cache_dir, train_file))
        torch.save(test_ds, os.path.join(args.data_cache_dir, test_file))

    #Dataloader
    train_dataloader = DataLoader(train_ds, shuffle=True, batch_size=args.batch_size, num_workers=0) 
    val_dataloader =  DataLoader(test_ds, shuffle=True, batch_size=args.batch_size, num_workers=0)

    #load model and tokenizer
    print('loading model embedding model...')
    tokenizer, embedding_model = load_tokenizer_embedding_model(args)
    print('loading extraction model...')
    ext_model = ClassificationModel(args, embedding_model)
    
    # log model setup
    print(f'Logging run_setup')
    def is_json_serializable(value):
        try:
            json.dumps(value)
            return True
        except (TypeError, OverflowError):
            return False

    args_dict = vars(args)
    filtered_args = {k: v for k, v in args_dict.items() if is_json_serializable(v)}
    with open(os.path.join(args.output_dir, f'run_setup.json'), 'w') as f_out:
        json.dump(filtered_args, f_out, indent=4)

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
    labels = first_example['token_labels'].tolist()
    print(f'Labels {labels}')
    labels = first_example['intent_labels'].tolist()
    print(f'Labels {labels}')

    # start training
    trainer = CustomTrainer(
        args=args,
        model=ext_model,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        test_dataloader=val_dataloader,
    )
    trainer.train()
