import bs4, copy, re
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
import torch
from torch.utils.data import DataLoader, random_split, Dataset, Subset
from transformers import AutoTokenizer



class DescriptionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer:AutoTokenizer, max_token_len:int=512):
        self.tokenizer = tokenizer
        self.texts = texts
        self.labels = labels
        self.max_token_len = max_token_len
        
        
    def __len__(self):
        return len(self.texts)
    
    
    def __getitem__(self, index: int):
        data = self.texts[index]
        encoding = self.tokenizer(data, add_special_tokens=True, max_length=self.max_token_len, return_token_type_ids=False,
                                  padding="max_length", truncation=True, return_attention_mask=True, return_tensors='pt')
        encoding = {k:v[0] for k,v in encoding.items()}
        if self.labels is not None:
            encoding["labels"] = torch.tensor(self.labels[index])
        return encoding



def get_dataset(config):
    model_name = config["network"]["model_name"]
    valid_rate = config["train"]["valid_rate"]
    batch_size = config["train"]["batch_size"]
    kfolds = config["train"]["kfolds"]
    seed = config["train"]["seed"]
    
    train_df = pd.read_csv("../data/train.csv", index_col=0) # id, description, jopflag
    test_df = pd.read_csv("../data/test.csv", index_col=0) # id, description
    train_texts = remove_html_tags(train_df)
    test_texts = remove_html_tags(test_df)
    train_labels = train_df["jobflag"].values - 1
    
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # loader
    if kfolds==1:
        train_texts, valid_texts, train_labels, valid_labels = train_test_split(train_texts, train_labels, test_size=valid_rate, stratify=train_labels)
        train_dataset = DescriptionDataset(train_texts, train_labels, tokenizer)
        valid_dataset = DescriptionDataset(valid_texts, valid_labels, tokenizer)
        train_loader = [DataLoader(train_dataset, batch_size=batch_size, shuffle=True)]
        valid_loader = [DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)]
        valid_labels = [valid_labels]
    elif kfolds>1:
        train_loader = []
        valid_loader = []
        valid_labels = []
        skf = StratifiedKFold(n_splits=kfolds, random_state=seed, shuffle=True)
        for train_indices, valid_indices in skf.split(train_texts, train_labels):
            train_loader.append(DataLoader(DescriptionDataset(train_texts[train_indices], train_labels[train_indices], tokenizer), batch_size=batch_size, shuffle=True))
            valid_loader.append(DataLoader(DescriptionDataset(train_texts[valid_indices], train_labels[valid_indices], tokenizer), batch_size=batch_size, shuffle=True))
            valid_labels.append(train_labels[valid_indices])
            
    test_dataset = DescriptionDataset(test_texts, None, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, valid_loader, test_loader, valid_labels
    
    
    
    
    
def remove_html_tags(df, parser="lxml"):
    results = copy.deepcopy(df["description"].values)
    for i in range(len(results)):
        results[i] = re.sub("([a-z])</li>", "\\1.</li>", results[i].replace("<li> ", "<li>").replace(" </li>", "</li>")).replace("</li>", " </li>").replace("\\", "")
        results[i] = bs4.BeautifulSoup(results[i], parser).get_text()
    return results