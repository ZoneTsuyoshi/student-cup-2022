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
    
    
    
def create_cv_data(config):
    kfolds = config["kfolds"]
    seed = config["seed"]
    
    train_df = pd.read_csv("../data/train.csv", index_col=0) # id, description, jopflag
    train_texts = adjust_texts(train_df["description"].values)
    train_labels = train_df["jobflag"].values
    cv_df = train_df.copy()
    cv_df["description"] = train_texts
    cv_df["jobflag"] = train_labels
    cv_df["fold"] = np.zeros(len(cv_df), dtype=int)
    
    skf = StratifiedKFold(n_splits=kfolds, random_state=seed, shuffle=True)
    for i, (train_indices, valid_indices) in enumerate(skf.split(train_texts, train_labels)):
        cv_df.loc[valid_indices, "fold"] = i
    cv_df.to_csv(f"../data/{kfolds}fold-seed{seed}.csv", index=True)
    
    
def get_train_data_for_mlm(config, return_dataset=["fold", "all"]):
    seed = config["base"]["seed"]
    kfolds = config["base"]["kfolds"]
    model_name = config["base"]["model_name"]
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset, valid_dataset = [], []
    
    if "fold" in return_dataset:
        train_df = pd.read_csv(f"../data/{kfolds}fold-seed{seed}.csv", index_col=0)
        train_texts = train_df["description"].values
        all_indices = np.arange(len(train_df))
        for i in range(kfolds):
            train_indices = all_indices[train_df["fold"]!=i]
            valid_indices = all_indices[train_df["fold"]==i]
            train_dataset.append(DescriptionDataset(**embed_and_augment(tokenizer, train_texts[train_indices])))
            valid_dataset.append(DescriptionDataset(**embed_and_augment(tokenizer, train_texts[valid_indices])))
    if "all" in return_dataset:
        train_df = pd.read_csv("../data/train.csv", index_col=0) # id, description, jopflag
        train_texts = adjust_texts(train_df["description"].values)
        train_dataset.append(DescriptionDataset(**embed_and_augment(tokenizer, train_texts)))
        valid_dataset.append(None)
    return train_dataset, valid_dataset
    
    
def get_train_data(config, debug=False, return_loader=["fold", "all"]):
    model_name = config["network"]["model_name"]
    weight_on = config["train"]["weight"]
    valid_rate = config["train"]["valid_rate"]
    batch_size = config["train"]["batch_size"]
    kfolds = config["train"]["kfolds"]
    seed = config["train"]["seed"]
    da_method = config["train"]["da"]
    mask_ratio = config["train"]["mask_ratio"]
    random.seed(seed)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_loader, valid_loader, valid_labels, valid_indices_list, weight = [], [], [], [], []
    
    if "fold" in return_loader:
        train_df = pd.read_csv(f"../data/{kfolds}fold-seed{seed}.csv", index_col=0)
        train_texts = train_df["description"].values
        train_labels = train_df["jobflag"].values - 1
        all_indices = np.arange(len(train_df))
        for i in range(kfolds):
            train_indices = all_indices[train_df["fold"]!=i]
            valid_indices = all_indices[train_df["fold"]==i]
            if debug:
                train_indices = train_indices[:32]
                valid_indices = valid_indices[:32]
            train_loader.append(DataLoader(DescriptionDataset(**embed_and_augment(tokenizer, train_texts[train_indices], train_labels[train_indices], da_method, mask_ratio)), batch_size=batch_size, shuffle=True))
            valid_loader.append(DataLoader(DescriptionDataset(**embed_and_augment(tokenizer, train_texts[valid_indices], train_labels[valid_indices])), batch_size=batch_size, shuffle=False))
            valid_labels.append(train_labels[valid_indices])
            valid_indices_list.append(valid_indices)
            if weight_on: 
                weight.append(torch.tensor(compute_class_weight("balanced", classes=np.arange(4), y=train_labels[train_indices]), dtype=torch.float32))
            else:
                weight.append(None)
    if "all" in return_loader:
        train_df = pd.read_csv("../data/train.csv", index_col=0) # id, description, jopflag
        train_texts = adjust_texts(train_df["description"].values)
        train_labels = train_df["jobflag"].values - 1
        if debug:
            train_texts = train_texts[:16]
            train_labels = train_labels[:16]
        train_loader.append(DataLoader(DescriptionDataset(**embed_and_augment(tokenizer, train_texts, train_labels, da_method, mask_ratio)), batch_size=batch_size, shuffle=True))
        valid_loader.append(None)
        valid_indices_list.append(None)
        valid_labels.append(None)
        if weight_on: 
            weight.append(torch.tensor(compute_class_weight("balanced", classes=np.arange(4), y=train_labels), dtype=torch.float32))
        else:
            weight.append(None)
    return train_loader, valid_loader, valid_labels, valid_indices_list, weight
    
    
    
def get_test_data(config, debug=False):
    model_name = config["network"]["model_name"]
    batch_size = config["train"]["batch_size"]
    
    test_df = pd.read_csv("../data/test.csv", index_col=0) # id, description
    test_texts = adjust_texts(test_df["description"].values)
    if debug:
        test_texts = test_texts[:16]
    
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    test_dataset = DescriptionDataset(**embed_and_augment(tokenizer, test_texts))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return test_loader
    
    
    
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
                    current_length = (current_input_ids > 0).sum().item() - 2
                    perms = np.random.choice(current_length, int(mask_ratio*current_length), replace=False)
                    for p in perms:
                        current_input_ids[p+1] = tokenizer.mask_token_id
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
        if "http" in results[i] or "u202f" in results[i]:
            words = results[i].split(" ")
            for j,w in enumerate(words):
                if "http" in w:
                    words[j] = "URL"
                if "u202f" in w:
                    words[j] = " ".join(w.split("u202f"))
            results[i] = " ".join(words)
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