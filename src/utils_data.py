import bs4, copy, re, random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
import nlpaug.augmenter.word as naw
import torch
from torch.utils.data import DataLoader, random_split, Dataset, Subset
from transformers import AutoTokenizer



class DescriptionDataset(Dataset):
    def __init__(self, input_ids:torch.Tensor, attention_mask:torch.Tensor, labels=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels
        
        
    def __len__(self):
        return len(self.input_ids)
    
    
    def __getitem__(self, index: int):
        encoding = {"input_ids":self.input_ids[index], "attention_mask":self.attention_mask[index]}
        if self.labels is not None:
            encoding["labels"] = self.labels[index]
        return encoding



def get_dataset(config):
    model_name = config["network"]["model_name"]
    weight_on = config["train"]["weight"]
    valid_rate = config["train"]["valid_rate"]
    batch_size = config["train"]["batch_size"]
    kfolds = config["train"]["kfolds"]
    seed = config["train"]["seed"]
    da_method = config["train"]["da"]
    mask_ratio = config["train"]["mask_ratio"]
    random.seed(seed)
    
    train_df = pd.read_csv("../data/train.csv", index_col=0) # id, description, jopflag
    test_df = pd.read_csv("../data/test.csv", index_col=0) # id, description
    train_texts = adjust_texts(train_df["description"].values)
    test_texts = adjust_texts(test_df["description"].values)
    train_labels = train_df["jobflag"].values - 1
    
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # loader
    if kfolds==1:
        train_texts, valid_texts, train_labels, valid_labels = train_test_split(train_texts, train_labels, test_size=valid_rate, stratify=train_labels)
        train_dataset = DescriptionDataset(**embed_and_augment(tokenizer, train_texts, train_labels, da_method, mask_ratio))
        valid_dataset = DescriptionDataset(**embed_and_augment(tokenizer, valid_texts, valid_labels))
        train_loader = [DataLoader(train_dataset, batch_size=batch_size, shuffle=True)]
        valid_loader = [DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)]
        valid_labels = [valid_labels]
        if weight_on:
            weight = [torch.tensor(compute_class_weight("balanced", classes=np.arange(4), y=train_labels), dtype=torch.float32)]
        else:
            weight = [None]
    elif kfolds>1:
        train_loader, valid_loader, valid_labels, weight = [], [], [], []
        skf = StratifiedKFold(n_splits=kfolds, random_state=seed, shuffle=True)
        for train_indices, valid_indices in skf.split(train_texts, train_labels):
            train_loader.append(DataLoader(DescriptionDataset(**embed_and_augment(tokenizer, train_texts[train_indices], train_labels[train_indices], da_method, mask_ratio)), batch_size=batch_size, shuffle=True))
            valid_loader.append(DataLoader(DescriptionDataset(**embed_and_augment(tokenizer, train_texts[valid_indices], train_labels[valid_indices])), batch_size=batch_size, shuffle=False))
            valid_labels.append(train_labels[valid_indices])
            if weight_on: 
                weight.append(torch.tensor(compute_class_weight("balanced", classes=np.arange(4), y=train_labels[train_indices]), dtype=torch.float32))
            else:
                weight.append(None)
            
    test_dataset = DescriptionDataset(**embed_and_augment(tokenizer, test_texts))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, valid_loader, test_loader, valid_labels, weight
    
    
    
def embed_and_augment(tokenizer, texts, labels=None, da_method=None, mask_ratio=0.1):
    """
    texts, labels: ndarray
    
    da_method
        l: shuffle list items
        s: transform synonyms
        b: transform by BERT
        r: transform by RoBERTa
        m: mask words
    """
    # tokenizer
    tokenizer_setting = {"add_special_tokens":True, "max_length":512, "return_token_type_ids":False,
                         "padding":"max_length", "truncation":True, "return_attention_mask":True, "return_tensors":'pt'}
    
    if labels is None:
        encoding = tokenizer(remove_html_tags(texts).tolist(), **tokenizer_setting)
        return encoding
    if da_method is None:
        encoding = tokenizer(remove_html_tags(texts).tolist(), **tokenizer_setting)
        encoding["labels"] = torch.tensor(labels.tolist())
        return encoding
    else:
        encoding = tokenizer(remove_html_tags(texts).tolist(), **tokenizer_setting)
        input_ids = [i for i in encoding["input_ids"]]
        attention_mask = [i for i in encoding["attention_mask"]]
        new_labels = labels.tolist()
        unique_labels, counts = np.unique(labels, return_counts=True)
        max_count = counts.max()
        if "s" in da_method: aug = naw.SynonymAug(aug_src='wordnet')
        if "b" in da_method: aug = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action="substitute")
        if "r" in da_method: aug = naw.ContextualWordEmbsAug(model_path='roberta-base', action="substitute")
        for u, c in zip(unique_labels, counts):
            now_texts = texts[labels==u]
            counter = c
            inner_counter = 0
            while counter < max_count:
                counter_flag = True
                da_text = now_texts[inner_counter]
                if "l" in da_method:
                    da_text = da_text.split("<li>")
                    if da_text is not None:
                        da_text = random.sample(da_text, len(da_text))
                        da_text = "".join(da_text)
                    else:
                        counter_flag = False
                da_text = remove_html_tags(da_text)
                if np.any([a in da_method for a in ["s", "b", "r"]]):
                    da_text = aug.augment(da_text)[0]
                    counter_flag = True
                encoding = tokenizer(da_text, **tokenizer_setting)
                current_input_ids = encoding["input_ids"][0]
                if "m" in da_method:
                    perms = np.random.choice(len(current_input_ids), int(mask_ratio*len(current_input_ids)), replace=False)
                    for p in perms:
                        current_input_ids[p] = tokenizer.mask_token_id
                    counter_flag = True
                
                input_ids.append(current_input_ids)
                attention_mask.append(encoding["attention_mask"][0])
                new_labels.append(u)
                inner_counter += 1
                if inner_counter == c:
                    inner_counter = 0
                if counter_flag:
                    counter += 1
        encoding = {"input_ids":torch.stack(input_ids), "attention_mask":torch.stack(attention_mask), "labels":torch.tensor(new_labels)}
        return encoding
                

def adjust_texts(arr):
    results = copy.deepcopy(arr)
    for i in range(len(results)):
        results[i] = re.sub("([a-z])</li>", "\\1.</li>", results[i].replace("<li> ", "<li>").replace(" </li>", "</li>")).replace("</li>", " </li>").replace("\\", "")
    return results
    
    
def remove_html_tags(arr, parser="lxml"):
    if type(arr)==str:
        return bs4.BeautifulSoup(arr, parser).get_text()
    else:
        results = copy.deepcopy(arr)
        for i in range(len(results)):
            # results[i] = re.sub("([a-z])</li>", "\\1.</li>", results[i].replace("<li> ", "<li>").replace(" </li>", "</li>")).replace("</li>", " </li>").replace("\\", "")
            results[i] = bs4.BeautifulSoup(results[i], parser).get_text()
        return results