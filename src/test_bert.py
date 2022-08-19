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
    debug = config["train"]["debug"]
    all_weight = config["test"]["all_weight"]
    if ckpt_name is None:
        ckpt_name = config["test"]["ckpt"]
    
    cuda = torch.cuda.is_available()
    print("cuda is avaiable: {}".format(cuda))
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    
    test_loader = get_test_data(config, debug)
    test_probs = []
    trainer = pl.Trainer(accelerator="gpu", devices=[gpu])
    for i in range(kfolds+1):
        fold_id = i if i<kfolds else "A"
        ckpt_path = os.path.join(dirpath, f"fold{fold_id}_{ckpt_name}.ckpt")
        if os.path.exists(ckpt_path):
            model = LitBertForSequenceClassification.load_from_checkpoint(ckpt_path)
            test_probs.append(F.softmax(torch.cat(trainer.predict(model, test_loader))).detach().cpu().numpy())
    test_probs = np.array(test_probs)
    np.save(os.path.join(dirpath, "test_probs.npy"), test_probs)
    
    # test_probs = np.load(os.path.join(dirpath, "test_probs.csv"))
    test_weights = np.ones(kfolds) if len(test_probs)==kfolds else np.concatenate([np.ones(kfolds), all_weight * np.ones(1)])
    test_probs = (np.array(test_probs) * test_weights[:,None,None]).sum(0) / (all_weight + kfolds)
    np.save(os.path.join(dirpath, "test_aggregated_probs.npy"), test_probs)
    labels_predicted = np.argmax(test_probs, -1)
    ids = np.arange(1516, 1532) if debug else np.arange(1516, 3033)
    pd.DataFrame(np.array([ids, labels_predicted+1]).T).to_csv(os.path.join(dirpath, "submission.csv"), header=False, index=False)
    
    
    
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
