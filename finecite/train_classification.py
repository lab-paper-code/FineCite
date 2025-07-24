
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