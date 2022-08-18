import os, sys, json, glob, time, shutil, copy, subprocess, math, datetime, argparse, itertools
import numpy as np
from utils import get_gpu_info

    
def gs_main(config):
    flatten_config = {}
    for key in config.keys():
        flatten_config.update(config[key])

    model_name_list = ["roberta", "microsoft/deberta-v3", "microsoft/deberta"]
    model_size_list = ["base", "large"]
    model_list = [i+"-"+j for (j,i) in itertools.product(model_size_list, model_name_list)]
    bs_list = np.repeat(np.array([8,4]), 3).tolist()
    wd_list = [0.1, 0.01, 0.01, 0.1, 0.01, 0.01]
    gs_dict = {"mix":{"model_name":model_list,
                     "batch_size":bs_list,
                     "weight_decay":wd_list},
               "mask_ratio":[0.1, 0.15],
               "num_train_epochs":[10, 20]}
    # print(type(model_list), type(bs_list))

    gs_key = list(gs_dict.keys()) # list of keys for grid search
    gs_length = len(gs_dict)
    gs_key2 = []
    for key in gs_key:
        # if dictionary has hierarchical structure, add hierarchical keys to gs_key2 list.
        if type(gs_dict[key])==list:
            gs_key2.append(key)
        elif type(gs_dict[key])==dict:
            gs_key2 += list(gs_dict[key].keys())
    
    config_list = []
        
        
    def generate_queue_flatten_config(old_config, depth):
        key = gs_key[depth]
        
        if type(gs_dict[key])==list:
            for i, value in enumerate(gs_dict[key]):
                new_config = copy.deepcopy(old_config)
                new_config[key] = value
                if depth+1 < gs_length:
                    generate_queue_flatten_config(new_config, depth+1)
                else:
                    config_list.append(new_config)
        elif type(gs_dict[key])==dict:
            interlocking_key = list(gs_dict[key].keys())
            min_length = 10
            for ikey in interlocking_key:
                min_length = len(gs_dict[key][ikey]) if len(gs_dict[key][ikey]) < min_length else min_length
            for i in range(min_length):
                new_config = copy.deepcopy(old_config)
                for ikey in interlocking_key:
                    new_config[ikey] = gs_dict[key][ikey][i]
                if depth+1 < gs_length:
                    generate_queue_flatten_config(new_config, depth+1)
                else:
                    config_list.append(new_config)
        else:
            raise ValueError("elements must be a list type object or a dict type object")
            
    
    def flatten_config_to_parse(config):
        parse_list = []
        for key in config.keys():
            parse_list.append("--{}".format(key))
            if type(config[key])==list:
                parse_list += [str(value) for value in config[key]]
            else:
                parse_list.append(str(config[key]))
        return parse_list
    
            
            
    generate_queue_flatten_config(flatten_config, 0)
    total_parse_list = []
    for config_element in config_list:
        total_parse_list.append(["python", "parse_mlm.py"] + flatten_config_to_parse(config_element))
    
    for parse_element in total_parse_list:
        subprocess.run(parse_element)
            
        
        
        
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='gs for mlm', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-c", "--config", type=str, default='config_mlm.json', help='specify the config file')
    args = parser.parse_args()
    
    f = open(args.config, "r")
    config = json.load(f)
    f.close()
    
    gs_main(config)