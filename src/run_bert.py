import argparse, datetime, json, os, glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from utils_data import get_dataset
from utils_train import LitBertForSequenceClassification


def main(config, dirpath):
    n_labels = 4
    job_list = ["DS", "MLE", "SE", "C"]
    epoch = config["train"]["epoch"]
    gpu = config["train"]["gpu"]
    seed = config["train"]["seed"]
    kfolds = config["train"]["kfolds"]
    # config["network"]["dirpath"] = dirpath
    
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
    
    train_loader_list, valid_loader_list, test_loader, valid_labels_list = get_dataset(config)
    comet_logger = pl.loggers.CometLogger(workspace=os.environ.get("zonetsuyoshi"), save_dir=dirpath, project_name="student-cup-2022")
    best_model_path_list = []
    for i, (train_loader, valid_loader) in enumerate(zip(train_loader_list, valid_loader_list)):
        model = LitBertForSequenceClassification(**config["network"], dirpath=dirpath, fold_id=i)
        checkpoint = pl.callbacks.ModelCheckpoint(monitor=f'valid_loss{i}', mode='min', save_top_k=1, save_weights_only=True, dirpath=dirpath, filename=f"fold{i}" + "{epoch}-{step}.ckpt")
        trainer = pl.Trainer(accelerator="gpu", devices=[gpu], max_epochs=epoch, callbacks=[checkpoint], logger=comet_logger)
        trainer.fit(model, train_loader, valid_loader)
        best_model_path_list.append(checkpoint.best_model_path)
    
    test_probs = []
    confmat = np.zeros([n_labels, n_labels])
    f1macro = 0
    # best_model_path_list = glob.glob(os.path.join(dirpath, "*.ckpt"))
    for model_path, valid_loader, valid_labels in zip(best_model_path_list, valid_loader_list, valid_labels_list):
        model = LitBertForSequenceClassification.load_from_checkpoint(model_path)
        # model.bert.save_pretrained(dirpath)
        trainer = pl.Trainer()
        valid_logits = torch.cat(trainer.predict(model, valid_loader)).detach().cpu().numpy()
        valid_predicted_labels = np.argmax(valid_logits, -1)
        confmat += metrics.confusion_matrix(valid_labels, valid_predicted_labels)
        f1macro += metrics.f1_score(valid_labels, valid_predicted_labels, average="macro")
        test_probs.append(F.softmax(torch.cat(trainer.predict(model, test_loader))).detach().cpu().numpy())
    comet_logger.log_metrics({"f1macro":f1macro/kfolds})
    fig, ax = plt.subplots(figsize=(5,5))
    sns.heatmap(confmat / confmat.sum(1)[:,None], cmap="Blues", ax=ax, vmin=0, vmax=1, square=True, annot=True, fmt=".2f")
    ax.set_xticklabels(job_list); ax.set_yticklabels(job_list)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); comet_logger.experiment.log_figure("confusion matrix", fig)
    labels_predicted = np.argmax(np.array(test_probs).sum(0), -1)
    pd.DataFrame(np.array([np.arange(1516, 3033), labels_predicted+1]).T).to_csv(os.path.join(dirpath, "submission.csv"), header=False, index=False)
    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Main Routine for Swithing Trajectory Network', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--config',
        type=str,
        default='config_bert.json',
        help='specify the config file')
    args = parser.parse_args()
    
    f = open(args.config, "r")
    config = json.load(f)
    f.close()
    
    model_name = config["network"]["model_name"]
    number_of_date = config["train"]["number_of_date"]
    dt_now = datetime.datetime.now()
    dirpath = os.path.join("../results", "{:02}-{}{}".format(dt_now.day, model_name, number_of_date))
    # dirpath = os.path.join("../results", "{:02}".format(dt_now.day))
    main(config, dirpath)