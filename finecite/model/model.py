import torch

class ExtractionModel(torch.nn.Module):
    def __init__(
        self,
        args,
        embedding_model,
        classifier,
    ):
        super().__init__()
        self.args = args
        self.model_name = args.model_name
        self.device = args.device
        self.embedding_model = embedding_model
        self.classifier = classifier
        
    def forward(self, **inputs):
        ids = inputs['input_ids'].to(self.device, dtype = torch.long)
        mask = inputs['input_mask'].to(self.device, dtype = torch.long)
        
        embedding = self.embedding_model(input_ids=ids, attention_mask=mask)
        hidden_state = embedding['last_hidden_state'].to(self.args.dtype)
        
        if 'llm2vec' in self.model_name: # the llm2vec model have a next token embedding, thus to align the model with the labels it has to be rolled forward one postion
            hidden_state = torch.roll(hidden_state, shifts=1, dims=1)

        output = self.classifier(hidden_state)
        return hidden_state, output
    
    def extract(self, data_loader):
        data_agg = []
        self.eval()
        for _,data in enumerate(data_loader, 0):
            with torch.no_grad():
                input_ids = data['input_ids']
                mask =(1-((input_ids == self.args.pad_token_id).to(torch.int) | (input_ids == self.args.cls_token_id).to(torch.int) | (input_ids == self.args.sep_token_id).to(torch.int))).to(self.device)
                hs, token_labels = self(**data)
                preds = self.predict(token_labels, mask == 1)
                data_agg.extend(preds)
        self._extract = False
        return data_agg
    
    
    
class ClassificationModel(torch.nn.Module):
    def __init__(
        self,
        args,
        embedding_model,
        classifier,
    ):
        super().__init__()
        self.args = args
        self.model_name = args.model_name
        self.device = args.device
        self.embedding_model = embedding_model
        self.classifier = classifier
        
    def forward(self, **inputs):
        ids = inputs['input_ids'].to(self.device, dtype = torch.long)
        mask = inputs['input_mask'].to(self.device, dtype = torch.long)
        token_labels = inputs['tok_lbl'].to(self.device)
        
        embedding = self.embedding_model(input_ids=ids, attention_mask=mask)
        hidden_state = embedding['last_hidden_state'].to(self.args.dtype)
        
        if 'llm2vec' in self.model_name: # the llm2vec model have a next token embedding, thus to align the model with the labels it has to be rolled forward one postion
            hidden_state = torch.roll(hidden_state, shifts=1, dims=1)

        output = self.classifier(hidden_state, token_labels)
        return hidden_state, output
    