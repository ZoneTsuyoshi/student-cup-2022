import json
import torch
from torch import nn
import torchmetrics as tm
import pytorch_lightning as pl
from transformers import AutoModel, AutoConfig, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from utils_loss import get_loss_fn
from utils_at import AWP
# from utils_metrics import *


class SequenceClassification(nn.Module):
    def __init__(self, model_name:str, dropout:float=0., num_labels:int=4):
        super().__init__()
        bert_config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        self.linear = nn.Linear(bert_config.hidden_size, num_labels)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        bout = self.bert(input_ids, attention_mask, token_type_ids)
        return self.linear(self.dropout(bout[0][:,0]))
        


class LitBertForSequenceClassification(pl.LightningModule):
    def __init__(self, model_name:str, dirpath, lr:float, dropout:float=0., weight_decay:float=0.01, 
                 beta1:float=0.9, beta2:float=0.99, epsilon:float=1e-8, gradient_clipping:float=1.0,
                 loss:str="CEL", gamma:float=1, alpha:float=1, lb_smooth:float=0.1, weight:torch.tensor=None,
                 scheduler:str=None, num_warmup_steps:int=100, num_training_steps:int=1000,
                 awp:bool=False, awp_lr:float=1e-4, awp_eps:float=1e-2, awp_start_epoch:int=1, awp_steps:int=1,
                 fold_id:int=0, num_labels:int=4):
        super().__init__()
        self.save_hyperparameters()

        # load BERT model
        self.sc_model = SequenceClassification(model_name, dropout, num_labels)
        self.metrics = tm.MetricCollection([tm.Accuracy(), tm.F1Score(num_classes=num_labels, average="macro")])
        self.confmat = tm.ConfusionMatrix(num_labels)
        self.loss_fn = get_loss_fn(loss, gamma, alpha, lb_smooth, num_labels, weight)
        self.using_awp = awp
        if self.using_awp:
            self.awp = AWP(self.sc_model, self.loss_fn, awp_lr, awp_eps, awp_start_epoch, awp_steps)
            self.automatic_optimization = False
        
        
        
    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        logits = self.sc_model(input_ids, attention_mask, token_type_ids)
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        return loss, logits
        

    def training_step(self, batch, batch_idx):
        if self.using_awp:
            loss, logits = self._training_awp_step(batch, batch_idx)
        else:
            loss, logits = self._shared_loss(batch, batch_idx, "train")
        self._shared_eval(batch, logits, "train")
        return loss
        

    def validation_step(self, batch, batch_idx):
        val_loss, logits = self._shared_loss(batch, batch_idx, "valid")
        self._shared_eval(batch, logits, "valid")
        
        
    def _training_awp_step(self, batch, batch_idx):
        opt = self.optimizers(use_pl_optimizer=True)
        loss, logits = self(**batch)
        opt.zero_grad()
        self.manual_backward(loss)

        if self.current_epoch >= self.hparams.awp_start_epoch:
            adv_loss = self.awp.attack_backward(**batch, optimizer=opt, epoch=self.current_epoch)
            self.manual_backward(adv_loss)
            self.awp._restore()
        
        nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.hparams.gradient_clipping)
        opt.step()
        self.log(f'train_loss{self.hparams.fold_id}', loss, on_step=True, on_epoch=True, logger=True)
        sch = self.lr_schedulers()
        sch.step()
        lr = float(sch.get_last_lr()[0])
        self.log('lr', lr, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        return loss, logits
        
        
    
    def _shared_loss(self, batch, batch_idx, prefix):
        loss, logits = self(**batch)
        self.log(f'{prefix}_loss{self.hparams.fold_id}', loss)
        return loss, logits
    
        
    def _shared_eval(self, batch, logits, prefix):
        labels = batch["labels"]
        labels_predicted = logits.argmax(-1)
        records = self.metrics(labels_predicted, labels)
        self.log_dict({f"{prefix}_{k}{self.hparams.fold_id}":v for k,v in records.items()}, prog_bar=False, logger=True, on_epoch=True)
        
        
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
    
        