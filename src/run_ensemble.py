import os, datetime
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.utils.class_weight import compute_class_weight
from scipy.optimize import minimize


def get_loss_fn(probs, valid_labels, sample_weight=None, loss_name="CE"):
    if loss_name in ["CE", "CEL"]:
        def loss_fn(weight):
            final_probs = (probs * weight[:,None,None]).sum()
            return metrics.log_loss(valid_labels, final_probs, sample_weight=sample_weight)
    elif loss_name in ["F1", "f1"]:
        def loss_fn(weight):
            final_probs = (probs * weight[:,None,None]).sum()
            final_preds = np.argmax(final_probs, -1)
            return -metrics.f1_score(valid_labels, final_probs, average="macro")
    return loss_fn


def main():
    ensemble_names = []
    loss_name = "f1"
    method = "Nelder-Mead"
    
    dt_now = datetime.datetime.now()
    dirpath = os.path.join("../ensembles", "{:02}_esm".format(dt_now.day))
    i = 1
    while os.path.exists(dirpath + str(i)):
        i += 1
    dirpath = dirpath + str(i)
    
    valid_probs_list, test_probs_list = [], []
    for name in ensemble_names:
        valid_probs_list.append(np.load(os.path.join("../results"), ensemble_names, "valid_probs.npy"))
        test_probs_list.append(np.load(os.path.join("../results"), ensemble_names, "test_aggregated_probs.npy"))
    valid_probs = np.array(valid_probs_list)
    valid_labels = df.read_csv("../data/train.csv")["jobflag"].values - 1
    test_probs = np.array(test_probs_list)
    class_weight = compute_class_weight("balanced", classes=np.arange(4), y=valid_labels)
    sample_weight = class_weights[valid_labels]
    loss_fn = get_loss_fn(valid_probs, valid_labels, sample_weight, loss_name)
    
    initial_w = np.ones(len(ensemble_names), dtype=float) / len(ensemble_names)
    cons = ({'type':'eq','fun':lambda w: 1-w.sum()})
    bounds = [(0,1)]*len(ensemble_names)
    results = minimize(log_loss_func, starting_values, method=method, bounds=bounds, constraints=cons)
    
    print("best {}: {:.2f}".format(loss_name, results["fun"]))
    np.save(os.path.join(dirpath, "results.npy"), {"results":results, "params":{"models":ensemble_names, "target":loss_name, "method":method}})
    weight = results["x"]
    
    labels_predicted = np.argmax((test_probs * weight[:,None,None]).sum(0), -1)
    pd.DataFrame(np.array([np.arange(1516, 3033), labels_predicted+1]).T).to_csv(os.path.join(dirpath, "submission.csv"), header=False, index=False)
    
    
    
if __name__ == "__main__":
    main()