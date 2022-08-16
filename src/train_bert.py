import os, json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
import torch
import pytorch_lightning as pl
from utils_data import get_train_data
from utils_train import LitBertForSequenceClassification, select_hyperparameters

def train(config, dirpath):
    gpu = config["train"]["gpu"]
    seed = config["train"]["seed"]
    
    if not os.path.exists(dirpath.rsplit("/",1)[0]):
        os.mkdir(dirpath.rsplit("/",1)[0])
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)
    with open(os.path.join(dirpath, "config_bert.json"), "w") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    print("set result directory to {}".format(dirpath))
    
    cuda = torch.cuda.is_available()
    print("cuda is avaiable: {}".format(cuda))
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    
    n_labels = 4
    job_list = ["DS", "MLE", "SE", "C"]
    config = select_hyperparameters(config)
    
    epoch = config["train"]["epoch"]
    kfolds = config["train"]["kfolds"]
    warmup_rate = config["train"]["warmup_rate"]
    gradient_clip_val = config["train"]["gradient_clipping"]
    
    train_loader_list, valid_loader_list, valid_labels_list, weight_list = get_train_data(config)
    comet_logger = pl.loggers.CometLogger(workspace=os.environ.get("zonetsuyoshi"), save_dir=dirpath, project_name="student-cup-2022")
    comet_logger.log_hyperparams(config["train"])
    confmat = np.zeros([n_labels, n_labels])
    f1macro = 0
    for i, (train_loader, valid_loader, valid_labels, weight) in enumerate(zip(train_loader_list, valid_loader_list, valid_labels_list, weight_list)):
        total_steps = epoch * len(train_loader)
        warmup_steps = int(warmup_rate * total_steps)
        ckpt_name = f"fold{i}_best"
        model = LitBertForSequenceClassification(**config["network"], dirpath=dirpath, fold_id=i, weight=weight, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        checkpoint = pl.callbacks.ModelCheckpoint(monitor=f'valid_loss{i}', mode='min', save_top_k=1, save_weights_only=True, dirpath=dirpath, filename=ckpt_name)
        trainer = pl.Trainer(accelerator="gpu", devices=[gpu], max_epochs=epoch, gradient_clip_val=gradient_clip_val, callbacks=[checkpoint], logger=comet_logger)
        trainer.fit(model, train_loader, valid_loader)
        
        model = LitBertForSequenceClassification.load_from_checkpoint(os.path.join(dirpath, ckpt_name + ".ckpt"))
        valid_logits = torch.cat(trainer.predict(model, valid_loader)).detach().cpu().numpy()
        valid_predicted_labels = np.argmax(valid_logits, -1)
        confmat += metrics.confusion_matrix(valid_labels, valid_predicted_labels)
        f1macro += metrics.f1_score(valid_labels, valid_predicted_labels, average="macro")
    comet_logger.log_metrics({"f1macro":f1macro/kfolds})
    fig, ax = plt.subplots(figsize=(5,5))
    sns.heatmap(confmat / confmat.sum(1)[:,None], cmap="Blues", ax=ax, vmin=0, vmax=1, square=True, annot=True, fmt=".2f")
    ax.set_xticklabels(job_list); ax.set_yticklabels(job_list)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); comet_logger.experiment.log_figure("confusion matrix", fig)
    
    