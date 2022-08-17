import json
import torch
from torch import nn
import torchmetrics as tm
import pytorch_lightning as pl
from transformers import AutoModel, AutoConfig, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from utils_loss import get_loss_fn
# from utils_metrics import *


class LitBertForSequenceClassification(pl.LightningModule):
    def __init__(self, model_name:str, dirpath, lr:float, dropout:float=0., weight_decay=0.01, 
                 beta1:float=0.9, beta2:float=0.99, epsilon:float=1e-8,
                 loss:str="CEL", gamma:float=1, alpha:float=1, lb_smooth:float=0.1, weight=None,
                 scheduler=None, num_warmup_steps:int=100, num_training_steps:int=1000,
                 fold_id:int=0, num_labels:int=4):
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
        self.loss_fn = get_loss_fn(loss, gamma, alpha, lb_smooth, num_labels, weight)
        
        
        
    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        bout = self.bert(input_ids, attention_mask, token_type_ids)
        logits = self.linear(self.dropout(bout[0][:,0]))
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
    
    
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay,
                                     betas=(self.hparams.beta1, self.hparams.beta2), eps=self.hparams.epsilon)
        if self.hparams.scheduler is None:
            return optimizer
        elif self.hparams.scheduler in ["LSW", "LSwW", "linear", "Linear"]:
            scheduler = get_linear_schedule_with_warmup(optimizer, self.hparams.num_warmup_steps, self.hparams.num_training_steps)
        elif self.hparams.scheduler in ["CSW", "CSwW", "cosine", "Cosine"]:
            scheduler = get_cosine_schedule_with_warmup(optimizer, self.hparams.num_warmup_steps, self.hparams.num_training_steps)
        return [optimizer], [scheduler]
    
    
    
    
def select_hyperparameters(config):
    model_name = config["network"]["model_name"]
    
    if "roberta" in model_name:
        default_config_path = "default_cfgs/roberta-base.json"
    elif "deberta-v3" in model_name:
        default_config_path = "default_cfgs/deberta-v3-base.json"
        
    f = open(default_config_path, "r")
    default_config = json.load(f)
    f.close()
        
    for fkey in config.keys():
        for key in config[fkey].keys():
            if config[fkey][key] is None and key in default_config.keys():
                config[fkey][key] = default_config[key]
                
    return config
    
        