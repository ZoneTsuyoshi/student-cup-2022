import os, json, pathlib, argparse
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from utils_data import get_test_data
from utils_train import LitBertForSequenceClassification


def test(dirpath, ckpt_name=None):
    f = open(os.path.join(dirpath, "config_bert.json"), "r")
    config = json.load(f)
    f.close()
    
    gpu = config["train"]["gpu"]
    seed = config["train"]["seed"]
    kfolds = config["train"]["kfolds"]
    if ckpt_name is None:
        ckpt_name = config["test"]["ckpt"]
    
    cuda = torch.cuda.is_available()
    print("cuda is avaiable: {}".format(cuda))
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    
    test_loader = get_test_data(config)
    test_probs = []
    trainer = pl.Trainer(accelerator="gpu", devices=[gpu])
    for i in range(kfolds):
        model = LitBertForSequenceClassification.load_from_checkpoint(os.path.join(dirpath, f"fold{i}_{ckpt_name}.ckpt"))
        test_probs.append(F.softmax(torch.cat(trainer.predict(model, test_loader))).detach().cpu().numpy())
    labels_predicted = np.argmax(np.array(test_probs).sum(0), -1)
    pd.DataFrame(np.array([np.arange(1516, 3033), labels_predicted+1]).T).to_csv(os.path.join(dirpath, "submission.csv"), header=False, index=False)
    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='bert', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dir', type=pathlib.Path)
    parser.add_argument("--best", action="use best model")
    parser.add_argument("--last", action="use last model")
    parser.add_argument("-d", "--dir")
    parser.add_argument("-b", "--best")
    parser.add_argument("-l", "--last")
    args = parser.parse_args()
    ckpt_name = "best"
    if args.last:
        ckpt_name = "last"
    test(args.dir, ckpt_name)