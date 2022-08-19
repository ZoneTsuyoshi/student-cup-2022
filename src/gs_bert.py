import os, sys, json, glob, time, shutil, copy, subprocess, math, datetime, argparse
import multiprocessing as mp
import numpy as np
from utils import get_gpu_info

    
def gs_main(config, parallel_strategy_on=False, max_parallel_queues=3, minimum_memory=1500):
    flatten_config = {}
    for key in config.keys():
        flatten_config.update(config[key])

    dt_now = datetime.datetime.now()
    number_of_date = config["train"]["number_of_date"]
    gpu_id = config["train"]["gpu"]
    
    model_list = ["roberta-base", "microsoft/deberta-v3-base", "microsoft/deberta-base", "xlnet-base-cased",
                 "roberta-large", "microsoft/deberta-v3-large", "microsoft/deberta-large"]
    bs_list = [16, 16, 16, 16, 8, 8, 4]
    wd_list = [0.1, 0.01, 0.01, 0.01, 0.1, 0.01, 0.01]
    mi_list = [5, 6, 5, 1, 5, 1, 1, 1]
    gs_dict = {"mix":{"model_name":model_list, "batch_size":bs_list, "weight_decay":wd_list, "mlm_id":mi_list},
              "using_mlm":[True, False],
              "mix2":{"loss":["FL", "DL"], "gamma":[2, 1]},
              "mix3":{"at":["awp", "fgm", None], "adv_lr":[1., 0.1, 1.], "gpu":[0,1,2]}}


    gs_key = list(gs_dict.keys()) # list of keys for grid search
    gs_length = len(gs_dict)
    gs_key2 = []
    for key in gs_key:
        # if dictionary has hierarchical structure, add hierarchical keys to gs_key2 list.
        if type(gs_dict[key])==list:
            gs_key2.append(key)
        elif type(gs_dict[key])==dict:
            gs_key2 += list(gs_dict[key].keys())
    
    dir_name = "../results/{:02}_gs{}/".format(dt_now.day, number_of_date)
    if not os.path.exists(os.path.join("../results", dir_name)):
        os.mkdir(os.path.join("../results", dir_name))
    name_list = []
    config_list = []
        
        
    def generate_queue_flatten_config(old_config, name, depth):
        key = gs_key[depth]
        
        if type(gs_dict[key])==list:
            for i, value in enumerate(gs_dict[key]):
                new_name = name
                new_config = copy.deepcopy(old_config)
                new_config[key] = value
                abbrev_list = key.split("_")
                for abbrev in abbrev_list:
                    new_name += abbrev[0]
                if key=="model_name":
                    if "/" in value:
                        new_name += value.rsplit("/",1)[1]
                    else:
                        new_name += value
                else:
                    new_name += str(value)
                if depth+1 < gs_length:
                    generate_queue_flatten_config(new_config, new_name, depth+1)
                else:
                    config_list.append(new_config)
                    name_list.append(new_name)
        elif type(gs_dict[key])==dict:
            interlocking_key = list(gs_dict[key].keys())
            min_length = 10
            for ikey in interlocking_key:
                min_length = len(gs_dict[key][ikey]) if len(gs_dict[key][ikey]) < min_length else min_length
            for i in range(min_length):
                new_name = name
                new_config = copy.deepcopy(old_config)
                for ikey in interlocking_key:
                    new_config[ikey] = gs_dict[key][ikey][i]
                    abbrev_list = ikey.split("_")
                    for abbrev in abbrev_list:
                        new_name += abbrev[0]
                    new_name += str(gs_dict[key][ikey][i])
                if depth+1 < gs_length:
                    generate_queue_flatten_config(new_config, new_name, depth+1)
                else:
                    config_list.append(new_config)
                    name_list.append(new_name)
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
    
            
            
    generate_queue_flatten_config(flatten_config, dir_name, 0)
    total_parse_list = []
    for config_element, name_element in zip(config_list, name_list):
        total_parse_list.append(["python", "parse_bert.py"] + flatten_config_to_parse(config_element) + ["--name", name_element])
    
    
    if parallel_strategy_on:
        for i in range((len(name_list)-1)//max_parallel_queues+1):
            p = mp.Pool(min(mp.cpu_count(), max_parallel_queues))
            p.map(subprocess.run, total_parse_list[max_parallel_queues*i:max_parallel_queues*(i+1)])
            p.close()
            if "gpu" in gs_key:
                gpu_ids = gs_dict["gpu"]
                memory_used = [int(get_gpu_info()[gpu_id]["memory.used"]) for gpu_id in gpu_ids]
                while max(memory_used) > minimum_memory:
                    print("waiting in {}-th parallel computation".format(i+1))
                    time.sleep(10)
                    memory_used = [int(get_gpu_info()[gpu_id]["memory.used"]) for gpu_id in gpu_ids]
            elif type(gpu_id)==int:
                memory_used = int(get_gpu_info()[gpu_id]["memory.used"])
                while memory_used > minimum_memory:
                    print("waiting in {}-th parallel computation".format(i+1))
                    time.sleep(10)
                    memory_used = int(get_gpu_info()[gpu_id]["memory.used"])
    else:
        for parse_element in total_parse_list:
            subprocess.run(parse_element)
    
    
            
def get_gpu_info(nvidia_smi_path='nvidia-smi', keys=("index", "uuid", "name", "timestamp", "memory.total", "memory.free", "memory.used", "utilization.gpu", "utilization.memory"), no_units=True):
    nu_opt = '' if not no_units else ',nounits'
    cmd = '%s --query-gpu=%s --format=csv,noheader%s' % (nvidia_smi_path, ','.join(keys), nu_opt)
    output = subprocess.check_output(cmd, shell=True)
    lines = output.decode().split('\n')
    lines = [ line.strip() for line in lines if line.strip() != '' ]

    return [ { k: v for k, v in zip(keys, line.split(', ')) } for line in lines ]
        
        
        
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Main Routine for Swithing Trajectory Network', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-c", "--config", type=str, default='config_bert.json', help='specify the config file')
    parser.add_argument("-p", "--parallel", type=bool, default=True, help="parallel")
    parser.add_argument("-q", "--queue", type=int, default=3, help="max queue")
    parser.add_argument("-m", "--memory", type=int, default=2000, help="minimum memory")
    args = parser.parse_args()
    
    f = open(args.config, "r")
    config = json.load(f)
    f.close()
    
    gs_main(config, args.parallel, args.queue, args.memory)