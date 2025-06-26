import random
import numpy as np
import torch
from collections import defaultdict, Counter

def set_seed(seed_value=42):
    # Set seed for reproducibility.
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    
def get_class_weights(flat_labels: list[str]):
    counter = Counter(flat_labels)
    sorted_counter = sorted(counter.items())
    counter_sum = sum(counter.values())
    ratio_scopes = [counter_sum / (len(counter) * c )for l, c in sorted_counter]
    print(ratio_scopes)
    return ratio_scopes
            

    
