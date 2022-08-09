import torch
import torchmetrics
import pytorch_lightning as pl
from transformers import BertForSequenceClassification, BertModel
# from utils_metrics import *


class LitBertForSequenceClassification(pl.LightningModule):
    def __init__(self, model_name:str, dirpath, lr:float, num_labels:int=4):
        super().__init__()
        self.save_hyperparameters()

        # load BERT model
        self.bert_sc = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.accuracy = torchmetrics.Accuracy()
        

    def training_step(self, batch, batch_idx):
        output = self.bert_sc(**batch)
        loss = output.loss
        self.log('train_loss', loss)
        # loss = self._shared_loss(batch, batch_idx, "train")
        self._shared_eval(batch, batch_idx, "train")
        return loss
        

    def validation_step(self, batch, batch_idx):
        output = self.bert_sc(**batch)
        val_loss = output.loss
        self.log('valid_loss', val_loss)
        # val_loss = self._shared_loss(batch, batch_idx, "valid")
        self._shared_eval(batch, batch_idx, "valid")
        
        
    def test_step(self, batch, batch_idx):
        output = self.bert_sc(**batch)
        labels_predicted = output.logits.argmax(-1)
        return labels_predicted
    
#     def test_step_end(self, test_outputs):
#         return torch.cat(test_outputs.labels), torch.cat(test_outputs.labels_predicted)
        
        
#     def test_epoch_end(self, test_step_outputs):
#         labels = torch.cat(test_step_outputs.labels)
#         labels_predicted = torch.cat(test_step_outputs.labels_predicted)
#         plot_confusion_matrix(labels, labels_predicted, self.dirpath)
        
    
    # def _shared_loss(self, batch, batch_idx, prefix):
    #     output = self.bert_sc(**batch)
    #     loss = output.loss
    #     self.log(f'{prefix}_loss', loss)
    #     return loss
    
        
    def _shared_eval(self, batch, batch_idx, prefix):
        labels = batch["labels"]
        output = self.bert_sc(**batch)
        labels_predicted = output.logits.argmax(-1)
        self.accuracy(labels_predicted, labels)
        self.log(f'{prefix}_accuracy', self.accuracy, on_epoch=True)
        
        
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        output = self.bert_sc(**batch)
        labels_predicted = output.logits.argmax(-1)
        return labels_predicted

        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)