import torch
from transformers import BertTokenizerFast as BertTokenizer, BertModel, BertForSequenceClassification
from torch import nn
from torch.nn import functional as F



class BertMultiClassifier(nn.Module):
    def __init__(self, bert_model_path, labels_count, hidden_dim=768, dropout=0.1):
        super().__init__()

        self.config = {
            'bert_model_path': bert_model_path,
            'labels_count': labels_count,
            'hidden_dim': hidden_dim,
            'dropout': dropout,
        }

        self.bert = BertModel.from_pretrained(bert_model_path)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_dim, labels_count)
        self.sigmoid = nn.Sigmoid()
        

    def forward(self, tokens, masks):
        _, pooled_output = self.bert(tokens, attention_mask=masks, output_all_encoded_layers=False)
        dropout_output = self.dropout(pooled_output)

        linear_output = self.linear(dropout_output)
        proba = self.sigmoid(linear_output)

        return 


