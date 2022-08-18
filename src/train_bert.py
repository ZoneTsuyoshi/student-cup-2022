import os, json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
import torch
import torch.nn.functional as F
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
    mlm_id = config["train"]["mlm_id"]
    gradient_clip_val = config["network"]["gradient_clipping"]
    manual_optimization = config["network"]["at"] is not None
    if manual_optimization: gradient_clip_val = None
    ckpt_name = config["test"]["ckpt"] # best / last
    
    train_loader_list, valid_loader_list, valid_labels_list, weight_list = get_train_data(config)
    logger = pl.loggers.CometLogger(workspace=os.environ.get("zonetsuyoshi"), save_dir=dirpath, project_name="student-cup-2022")
    logger.log_hyperparams(config["train"])
    confmat = np.zeros([n_labels, n_labels])
    f1macro = 0; valid_probs_list = []
    for i, (train_loader, valid_loader, valid_labels, weight) in enumerate(zip(train_loader_list, valid_loader_list, valid_labels_list, weight_list)):
        fold_id = i if valid_loader is not None else "A"
        total_steps = epoch * len(train_loader)
        warmup_steps = int(warmup_rate * total_steps) if warmup_rate < 1 else warmup_rate
        mlm_path = f"../pretrained_models/mlm-k{kfolds}-s{seed}-deberta-v3-base{mlm_id}/fold{fold_id}" if mlm_id is not None else None
        model = LitBertForSequenceClassification(**config["network"], dirpath=dirpath, fold_id=fold_id, weight=weight, num_warmup_steps=warmup_steps, num_training_steps=total_steps, mlm_path=mlm_path)
        checkpoint = pl.callbacks.ModelCheckpoint(monitor=f'valid_loss{fold_id}' if valid_loader is not None else f"train_loss{fold_id}", mode='min', save_last=True, save_top_k=1, save_weights_only=True, dirpath=dirpath, filename=f"fold{fold_id}_best")
        checkpoint.CHECKPOINT_NAME_LAST = f"fold{fold_id}_last"
        trainer = pl.Trainer(accelerator="gpu", devices=[gpu], max_epochs=epoch, gradient_clip_val=gradient_clip_val, callbacks=[checkpoint], logger=logger)
        trainer.fit(model, train_loader, valid_loader)
        
        if valid_loader is not None:
            model = LitBertForSequenceClassification.load_from_checkpoint(os.path.join(dirpath, f"fold{fold_id}_{ckpt_name}.ckpt"))
            valid_logits = torch.cat(trainer.predict(model, valid_loader)).detach().cpu().numpy()
            valid_probs_list.append(F.softmax(valid_logits, dim=-1).numpy())
            valid_predicted_labels = np.argmax(valid_logits.numpy(), -1)
            confmat += metrics.confusion_matrix(valid_labels, valid_predicted_labels)
            f1macro += metrics.f1_score(valid_labels, valid_predicted_labels, average="macro")
    logger.log_metrics({"f1macro":f1macro/kfolds})
    
    fig, ax = plt.subplots(figsize=(5,5))
    sns.heatmap(confmat / confmat.sum(1)[:,None], cmap="Blues", ax=ax, vmin=0, vmax=1, square=True, annot=True, fmt=".2f")
    ax.set_xticklabels(job_list); ax.set_yticklabels(job_list)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); logger.experiment.log_figure("confusion matrix", fig)
    
    fig, ax = plt.subplots(2,2,figsize=(10,10))
    ax = ax.ravel()
    valid_probs = np.concatenate(valid_probs_list)
    np.save(os.path.join(dirpath, "valid_probs.npy"), valid_probs)
    valid_labels = np.concatenate(valid_labels_list[:kfolds])
    for i, job_name in enumerate(job_list):
        target = valid_labels==i
        valid_bool = np.argmax(valid_probs[target], -1) == valid_labels[target]
        for area in [valid_bool, ~valid_bool]:
            sns.histplot(valid_probs[target][area], bins=np.linspace(0,1,11), ax=ax[i], stat="proportion")
        ax[i].set_xlabel(job_name)
    logger.experiment.log_figure("probs histogram", fig)
        