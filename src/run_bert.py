import argparse, datetime, json, os
import pandas as pd
import numpy as np
import torch
import pytorch_lightning as pl

from utils_data import get_dataset
from utils_train import LitBertForSequenceClassification


def main(config, dirpath):
    model_name = config["pretrained_model"]
    epoch = config["epoch"]
    lr = config["lr"]
    gpu = config["gpu"]
    seed = config["seed"]
    
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
    
    train_loader, valid_loader, test_loader = get_dataset(config)
    model = LitBertForSequenceClassification(model_name, dirpath, lr)
    checkpoint = pl.callbacks.ModelCheckpoint(monitor='valid_loss', mode='min', save_top_k=1, save_weights_only=True, dirpath=dirpath)
    comet_logger = pl.loggers.CometLogger(workspace=os.environ.get("zonetsuyoshi"), save_dir=dirpath, project_name="student-cup-2022")
    trainer = pl.Trainer(accelerator="gpu", devices=[gpu], max_epochs=epoch, callbacks=[checkpoint], logger=comet_logger)
    trainer.fit(model, train_loader, valid_loader)
    
    # best_model_path = os.path.join("../results/09/epoch=0-step=202.ckpt")
    # model = LitBertForSequenceClassification(model_name, dirpath, lr).load_from_checkpoint(best_model_path)
    model = LitBertForSequenceClassification.load_from_checkpoint(checkpoint.best_model_path)
    model.bert_sc.save_pretrained(dirpath)
    trainer = pl.Trainer()
    labels_predicted = torch.cat(trainer.predict(model, test_loader))
    pd.DataFrame(np.array([np.arange(1516, 3033), labels_predicted.detach().cpu().numpy()+1]).T).to_csv(os.path.join(dirpath, "submission.csv"), header=False, index=False)
    
    
    
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
    
    model_name = config["pretrained_model"]
    dt_now = datetime.datetime.now()
    # dirpath = os.path.join("../results", "{:02}{:02}{:02}-{}".format(dt_now.day, dt_now.hour, dt_now.minute, model_name))
    dirpath = os.path.join("../results", "{:02}".format(dt_now.day))
    main(config, dirpath)