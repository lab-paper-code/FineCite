# *FINECite : A novel three-class framework for fine-grained citation context analysis*

We realize this framework by constructing a novel corpus containing 1,056 manually annotated fine-grained citation contexts. Next, we establish baseline models for two important applications in citation context analysis: citation context extraction and citation context classification. Importantly, our experiments demonstrate the positive impact of our finer-grained context definition leading to an increase in performance on both tasks compared to previous approaches.


<p align="center">
  <img src="https://github.com/user-attachments/assets/2096d5f0-91bd-4133-9880-4eda813aa822" width="75%" alt="Comparing_table"/>
</p>


## The Explanation of the FINCite code.
1. data : The storage of datas for our paper
2. corpus_construction : Data processing
3. model_training
    1. seq_tagger.py : model training and evaluation
    2. output : the storage of reults after model running
    3. extract_resulty : Aggregate the results to show which result is the best performance
    4. finecite_scopes_weights.json & finecite_total_weights.json : It is used to train the model.
4. classification_task

## Model training
- We use the pre-trained weights of SciBERT, LLM2VEC Mistra 7B, and LLM2VEC LLAMA 3 8B from huggingface transformers
- We use a QLoRA method to fine-tuning the model
- Evaluate the result by mode
### Model Configurations
```python
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
```

### QLoRA
1. Quantize a model to 4-bits by configuring the BitsAndBytesConfig class
```python
import torch
from transformers import BitsAndBytesConfig, AutoModel

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModel.from_pretrained(
                base_model_dir,
                trust_remote_code=True,
                config=self.config,
                quantization_config = bnb_config,
                torch_dtype=torch.bfloat16,
            )
```
2. Create a LoraConfig and use the get_peft_model() function to create a PeftModel from the quantized model and configuration.
```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    target_modules=['embed_tokens',"v_proj", "q_proj", "up_proj", "o_proj", "k_proj", "down_proj", "gate_proj" ], 
    inference_mode=False, 
    r=16, 
    lora_alpha=32, 
    lora_dropout=0.05
    )         

model = get_peft_model(model, config)
```

### Evaluate the results
```python
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

```

## How to use:

1. Create virtual environment:
    ```
    virtualenv venv
    ```

2. Activate virtual environsment
    ```
    source venv/bin/activate
    ```

3. install requirements_finecite.txt
    ```
    pip install -r requirements_finecite.txt
    ```

4. Update requirement.txt
    ```
    pip freeze >requirements_finecite.txt
    ```