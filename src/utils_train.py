import torch
from torch import nn
import torchmetrics
import pytorch_lightning as pl
from transformers import BertForSequenceClassification, BertModel, BertConfig
# from utils_metrics import *


class LitBertForSequenceClassification(pl.LightningModule):
    def __init__(self, model_name:str, dirpath, lr:float, dropout:float=0., fold_id:int=0, num_labels:int=4):
        super().__init__()
        self.save_hyperparameters()

        # load BERT model
        # self.bert_sc = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        bert_config = BertConfig.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name)
        self.accuracy = torchmetrics.Accuracy()
        
        self.linear = nn.Linear(bert_config.hidden_size, num_labels)
        self.dropout = nn.Dropout(dropout)
        
        
    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        bout = self.bert(input_ids, attention_mask, token_type_ids)
        logits = self.linear(self.dropout(bout[1]))
        loss = None
        if labels is not None:
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        return loss, logits
        

    def training_step(self, batch, batch_idx):
        loss, logits = self._shared_loss(batch, batch_idx, "train")
        self._shared_eval(batch, logits, "train")
        return loss
        

    def validation_step(self, batch, batch_idx):
        val_loss, logits = self._shared_loss(batch, batch_idx, "valid")
        self._shared_eval(batch, logits, "valid")
        
        
    def test_step(self, batch, batch_idx):
        output = self.bert_sc(**batch)
        labels_predicted = output.logits.argmax(-1)
        return labels_predicted
        
    
    def _shared_loss(self, batch, batch_idx, prefix):
        # output = self.bert_sc(**batch)
        # loss = output.loss
        loss, logits = self(**batch)
        self.log(f'{prefix}_loss{self.hparams.fold_id}', loss)
        return loss, logits
    
        
    def _shared_eval(self, batch, logits, prefix):
        labels = batch["labels"]
        labels_predicted = logits.argmax(-1)
        self.accuracy(labels_predicted, labels)
        self.log(f'{prefix}_accuracy{self.hparams.fold_id}', self.accuracy, on_epoch=True)
        
        
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        _, logits = self(**batch)
        labels_predicted = logits.argmax(-1)
        return logits

        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)