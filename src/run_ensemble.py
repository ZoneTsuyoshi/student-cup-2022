import os, datetime, json
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.utils.class_weight import compute_class_weight
from scipy.optimize import minimize


def get_loss_fn(probs, valid_labels, sample_weight=None, loss_name="CE"):
    if loss_name in ["CE", "CEL"]:
        def loss_fn(weight):
            final_probs = (probs * weight[:,None,None]).sum(0)
            return metrics.log_loss(valid_labels, final_probs, sample_weight=sample_weight)
    elif loss_name in ["F1", "f1"]:
        def loss_fn(weight):
            final_probs = (probs * weight[:,None,None]).sum(0)
            final_preds = np.argmax(final_probs, -1)
            return -metrics.f1_score(valid_labels, final_preds, average="macro")
    return loss_fn


def main():
    ensemble_names = ["24_gs1/mnroberta-largebs8wd0.1e10mi5lFLaawpal1.0g1",
                     "24_gs1/mnmicrosoft/deberta-v3-largebs8wd0.01e10mi1lFLaNoneal1.0g2",
                     "25_gs1/mnxlnet-large-casedbs4wd0.01e5mi1aawpal0.1g0"]
    loss_name = "f1"
    method = "Nelder-Mead" # Nelder-Mead, Powell
    
    dt_now = datetime.datetime.now()
    dirpath = os.path.join("../ensembles", "{:02}_esm_{}_{}".format(dt_now.day, loss_name, method))
    i = 1
    while os.path.exists(dirpath + str(i)):
        i += 1
    dirpath = dirpath + str(i)
    os.mkdir(dirpath)
    print("dirpath: {}".format(dirpath))
    
    valid_probs_list, test_probs_list = [], []
    for name in ensemble_names:
        valid_probs_list.append(np.load(os.path.join("../results", name, "valid_probs.npy")))
        test_probs_list.append(np.load(os.path.join("../results", name, "test_aggregated_probs.npy")))
    valid_probs = np.array(valid_probs_list)
    valid_labels = pd.read_csv("../data/train.csv")["jobflag"].values - 1
    test_probs = np.array(test_probs_list)
    class_weight = compute_class_weight("balanced", classes=np.arange(4), y=valid_labels)
    sample_weight = class_weight[valid_labels]
    loss_fn = get_loss_fn(valid_probs, valid_labels, sample_weight, loss_name)
    
    initial_w = np.ones(len(ensemble_names), dtype=float) / len(ensemble_names)
    cons = ({'type':'eq','fun':lambda w: 1-w.sum()})
    bounds = [(0,1)]*len(ensemble_names)
    results = minimize(loss_fn, initial_w, method=method, bounds=bounds, constraints=cons)
    
<<<<<<< HEAD
    weight = results["x"]
    f1_score = -get_loss_fn(valid_probs, valid_labels, sample_weight, "f1")(weight)
    print("best f1: {:.3f}".format(f1_score))
    result_dict = {"models":ensemble_names, "target":loss_name, "method":method, "weight":results["x"].tolist(), loss_name:results["fun"], "F1":f1_score}
=======
    print("best {}: {:.2f}".format(loss_name, results["fun"]))
    result_dict = {"models":ensemble_names, "target":loss_name, "method":method, "weight":results["x"].tolist(), "f1":-results["fun"]}
>>>>>>> 166a8d6e438837f5bd78ee1d49198f9c68637e23
    with open(os.path.join(dirpath, "results.json"), "w") as f:
        json.dump(result_dict, f, indent=4, ensure_ascii=False)
    # np.save(os.path.join(dirpath, "results.npy"), {"results":results, "params":{"models":ensemble_names, "target":loss_name, "method":method}})
    
    labels_predicted = np.argmax((test_probs * weight[:,None,None]).sum(0), -1)
    pd.DataFrame(np.array([np.arange(1516, 3033), labels_predicted+1]).T).to_csv(os.path.join(dirpath, "submission.csv"), header=False, index=False)
    
    
    
if __name__ == "__main__":
    main()