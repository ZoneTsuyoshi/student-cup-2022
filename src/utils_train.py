import torch
from torch import nn
import torchmetrics as tm
import pytorch_lightning as pl
from transformers import AutoModel, AutoConfig
from utils_loss import get_loss_fn
# from utils_metrics import *


class LitBertForSequenceClassification(pl.LightningModule):
    def __init__(self, model_name:str, dirpath, lr:float, dropout:float=0., weight_decay=0.01, 
                 loss:str="CEL", gamma:float=1, alpha:float=1, weight=None, fold_id:int=0, num_labels:int=4):
        super().__init__()
        self.save_hyperparameters()

        # load BERT model
        # self.bert_sc = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        bert_config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        self.metrics = tm.MetricCollection([tm.Accuracy(), tm.F1Score(num_classes=num_labels, average="macro")])
        # self.accuracy = tm.Accuracy()
        self.confmat = tm.ConfusionMatrix(num_labels)
        
        self.linear = nn.Linear(bert_config.hidden_size, num_labels)
        self.dropout = nn.Dropout(dropout)
        self.loss_fn = get_loss_fn(loss, gamma, alpha, weight)
        
        
    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        bout = self.bert(input_ids, attention_mask, token_type_ids)
        logits = self.linear(self.dropout(bout[1]))
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        return loss, logits
        

    def training_step(self, batch, batch_idx):
        loss, logits = self._shared_loss(batch, batch_idx, "train")
        self._shared_eval(batch, logits, "train")
        return loss
        

    def validation_step(self, batch, batch_idx):
        val_loss, logits = self._shared_loss(batch, batch_idx, "valid")
        self._shared_eval(batch, logits, "valid")
        
        
#     def test_step(self, batch, batch_idx):
#         _, logits = self(**batch)
#         labels_predicted = logits.argmax(-1)
#         confmat = self.confmat(labels_predicted, batch["labels"])
#         return confmat
    
    
#     def test_step_end(self, test_step_outputs):
#         return test_step_outputs
    
    
#     def test_epoch_end(self, test_epoch_outputs):
#         return torch.stack(test_epoch_outputs).sum(0)
        
    
    def _shared_loss(self, batch, batch_idx, prefix):
        # output = self.bert_sc(**batch)
        # loss = output.loss
        loss, logits = self(**batch)
        self.log(f'{prefix}_loss{self.hparams.fold_id}', loss)
        return loss, logits
    
        
    def _shared_eval(self, batch, logits, prefix):
        labels = batch["labels"]
        labels_predicted = logits.argmax(-1)
        records = self.metrics(labels_predicted, labels)
        self.log_dict({f"{prefix}_{k}{self.hparams.fold_id}":v for k,v in records.items()}, prog_bar=False, logger=True, on_epoch=True)
        # self.accuracy(labels_predicted, labels)
        # self.log(f'{prefix}_accuracy{self.hparams.fold_id}', self.accuracy, on_epoch=True)
        
        
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        _, logits = self(**batch)
        # if "labels" in batch.keys():
        #     labels_predicted = logits.argmax(-1)
        #     return self.confmat(labels_predicted, batch["labels"])
        # else:
        return logits

    
    # def on_predict_epoch_end(self, predict_epoch_outputs):
    #     logits = []
    #     confmat = []
    #     for outputs in predict_epoch_outputs[0]:
    #         logits.append(outputs[0])
    #         confmat.append(outputs[1])
    #         print(outputs[0], outputs[1])
    #     return torch.cat(logits), torch.stack(confmat).sum(0)
    
    
        
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)