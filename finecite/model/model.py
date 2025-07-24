import torch
from torch.nn.modules.loss import CrossEntropyLoss, BCEWithLogitsLoss
from finecite.model.classifier import ClsClassifier, ExtClassifier, CRF

class ExtractionModel(torch.nn.Module):
    def __init__(
        self,
        args,
        embedding_model,
    ):
        super().__init__()
        self.args = args
        self.model_name = args.model_name
        self.device = args.device
        self.embedding_model = embedding_model
        self.classifier = ExtClassifier(args, embedding_model.config)
        
        if self.args.ext_type in ['crf', 'bilstm_crf']:
            self.crf = CRF(self.args.num_labels, self.args.device, 1)
        else:
            self.loss_fn = CrossEntropyLoss(weight=torch.FloatTensor(self.args.label_weights).to(self.device))
        
    def forward(self, **inputs):
        ids = inputs['input_ids'].to(self.device, dtype = torch.long)
        mask = inputs['input_mask'].to(self.device, dtype = torch.long)
        token_labels = inputs['token_labels'][:,0,:].to(self.device, dtype = torch.long)
        
        embedding = self.embedding_model(input_ids=ids, attention_mask=mask)
        hidden_state = embedding['last_hidden_state'].to(self.args.dtype)
        
        if 'llm2vec' in self.model_name: # the llm2vec model have a next token embedding, thus to align the model with the labels it has to be rolled forward one postion
            hidden_state = torch.roll(hidden_state, shifts=1, dims=1)

        output = self.classifier(hidden_state)
        output = output.float()
        
        if self.args.ext_type in ['crf', 'bilstm_crf']:
            loss = self.crf(output, token_labels) 
        else:
            loss = self.loss_fn(output, token_labels)
            
        return hidden_state, output, loss
    
    def predict(self, **inputs):
        ids = inputs['input_ids'].to(self.device, dtype = torch.long)
        mask = inputs['input_mask'].to(self.device, dtype = torch.long)
        
        embedding = self.embedding_model(input_ids=ids, attention_mask=mask)
        hidden_state = embedding['last_hidden_state'].to(self.args.dtype)
        
        if 'llm2vec' in self.model_name: # the llm2vec model have a next token embedding, thus to align the model with the labels it has to be rolled forward one postion
            hidden_state = torch.roll(hidden_state, shifts=1, dims=1)

        output = self.classifier(hidden_state)
        output = output.float()
            
        return hidden_state, output
    
    def extract_token_labels(self, data_loader):
        data_agg = []
        self.eval()
        for _,data in enumerate(data_loader, 0):
            with torch.no_grad():
                input_ids = data['input_ids']
                mask =(1-((input_ids == self.args.pad_token_id).to(torch.int) | (input_ids == self.args.cls_token_id).to(torch.int) | (input_ids == self.args.sep_token_id).to(torch.int))).to(self.device)
                hs, token_labels = self.predict(**data)
                if self.args.ext_type in ['crf', 'bilstm_crf']:
                    output = self.crf.predict(token_labels, mask == 1)
                    preds = torch.full(mask.shape, -100, dtype=torch.long) 
                    for i,( m, o) in enumerate(zip(mask, output)):
                        preds[i, m.bool()] = torch.tensor(o, dtype=torch.long)
                else:
                    preds = [[p if m else -100 for p, m in zip(preds, mask)] for preds, mask  in zip(torch.argmax(token_labels, dim = 1).tolist(), (mask == 1).tolist())]
                data_agg.extend(preds)
        self._extract = False
        return data_agg
    
    def decode_logits_to_labels(self, logits, labels):
        if self.args.ext_type in ['crf', 'bilstm_crf']:
            preds = self.crf.decode(logits, labels != -100) 
        else:
            preds = torch.argmax(logits, dim = 1)[labels != -100]
        return preds
    
    
class ClassificationModel(torch.nn.Module):
    def __init__(
        self,
        args,
        embedding_model,
    ):
        super().__init__()
        self.args = args
        self.model_name = args.model_name
        self.device = args.device
        self.embedding_model = embedding_model
        self.classifier = ClsClassifier(args, embedding_model.config)
        
        if self.args.dataset == 'multicite':
            self.loss_fn = BCEWithLogitsLoss(pos_weight=torch.BFloat16Tensor(self.args.label_weights).to(self.device))
        else:
            self.loss_fn = CrossEntropyLoss(weight=torch.BFloat16Tensor(self.args.label_weights).to(self.device))
        
    def forward(self, **inputs):
        ids = inputs['input_ids'].to(self.device, dtype = torch.long)
        mask = inputs['input_mask'].to(self.device, dtype = torch.long)
        token_labels = inputs['token_labels'].to(self.device)
        intent_labels = inputs['intent_labels'].to(self.device)
        
        embedding = self.embedding_model(input_ids=ids, attention_mask=mask)
        hidden_state = embedding['last_hidden_state'].to(self.args.dtype)
        
        if 'llm2vec' in self.model_name: # the llm2vec model have a next token embedding, thus to align the model with the labels it has to be rolled forward one postion
            hidden_state = torch.roll(hidden_state, shifts=1, dims=1)

        output = self.classifier(hidden_state, token_labels)
        
        loss = self.loss_fn(output, intent_labels)
                
        return hidden_state, output, loss
    