import argparse, json, os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForMaskedLM, DataCollatorForLanguageModeling, TrainingArguments, Trainer
# from transformers.integrations import CometCallbacks
from utils_data import get_train_data_for_mlm


def main(config):
    gpu = config["base"]["gpu"]
    seed = config["base"]["seed"]
    kfolds = config["base"]["kfolds"]
    mask_ratio = config["base"]["mask_ratio"]
    model_name = config["base"]["model_name"]
    batch_size = config["base"]["batch_size"]
    
    dirpath = f"../pretrained_models/mlm-k{kfolds}-s{seed}-"
    if "/" in model_name:
        dirpath += model_name.rsplit("/", 1)[1]
    else:
        dirpath += model_name
    
    i = 1
    while os.path.exists(dirpath + str(i)):
        i += 1
    dirpath += str(i)
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)
    with open(os.path.join(dirpath, "config_mlm.json"), "w") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    print("set result directory to {}".format(dirpath))
    
    cuda = torch.cuda.is_available()
    print("cuda is avaiable: {}".format(cuda))
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
        device = torch.device('cuda', gpu)
    np.random.seed(seed)
    
    os.environ["COMET_WORKSPACE"] = "zonetsuyoshi"
    os.environ["COMET_PROJECT_NAME"] = "student-cup-2022-mlm"
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    
    train_dataset_list, valid_dataset_list = get_train_data_for_mlm(config)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=mask_ratio)
    mlm_config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)
        
    for i, (train_dataset, valid_dataset) in enumerate(zip(train_dataset_list, valid_dataset_list)):
        if valid_dataset is not None:
            fold_id = i
            evaluation_strategy = "epoch"
        else:
            fold_id = "A"
            evaluation_strategy = "no"
        training_args = TrainingArguments(**config["training"], output_dir=dirpath, evaluation_strategy=evaluation_strategy, save_strategy="no",
                                     report_to="comet_ml", seed=seed, per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size)
        model = AutoModelForMaskedLM.from_pretrained(model_name, config=mlm_config)
        trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=valid_dataset, data_collator=data_collator)
        trainer.train()
        trainer.model.save_pretrained(os.path.join(dirpath, f"fold{fold_id}"))
    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='mlm', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default='config_mlm.json', help='specify the config file')
    args = parser.parse_args()
    
    f = open(args.config, "r")
    config = json.load(f)
    f.close()
    
    main(config)